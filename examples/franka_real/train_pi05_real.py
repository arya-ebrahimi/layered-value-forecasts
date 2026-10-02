#!/usr/bin/env python3
"""
Franka Robot Inference Script for Real-Time Kernel

This script runs on the real-time kernel machine and:
1. Uses franky to read robot state (qpos + gripper)
2. Uses franky to control robot with end-effector velocity control
3. Uses RealSense cameras to capture images
4. Converts BGR to RGB
5. Connects to policy server (model runs on separate server)

Setup:
- Install: pip install pyrealsense2 franky-control
- Install openpi-client: cd packages/openpi-client && pip install -e .
"""
import cv2
import logging
import time
from franky import *
from typing import Optional, Tuple
# from franka_robot import FrankaRobot, RobotInputs
import numpy as np
import pyrealsense2 as rs
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import threading
import queue
import sys
import termios
import tty
import select

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MotionAborted(Exception):
    """Raised when a franky motion command is aborted mid-execution (e.g. the arm/
    gripper stalls against an obstacle and libfranka's reflex trips the control
    loop). Distinguished from other exceptions so the main loop can report THIS
    specific failure mode as an episode failure instead of crashing the process."""


class MotionDiscontinuity(MotionAborted):
    """A libfranka *command-validity* rejection rather than a contact reflex.

    libfranka checks every commanded setpoint against its acceleration/jerk limits and
    aborts with e.g. `cartesian_motion_generator_acceleration_discontinuity` when the new
    setpoint is too far from the current one. That is a property of HOW we command the arm
    -- execute_cartesion_vel_action issues a fresh CartesianVelocityMotion every control
    step with asynchronous=True, so each 10 Hz command preempts the one still executing --
    and NOT a statement about the scene or the policy.

    Treating it like a snagged book (end the episode, reward 0) was throwing away roughly
    half the episodes and blaming the policy for a controller artifact. It subclasses
    MotionAborted so that any handler that does not care about the distinction still
    catches it; handlers that DO care must be ordered before the MotionAborted one.

    The cost of ignoring it, stated plainly: the aborted command did not execute, but the
    trainer still records that action against the next observation. That is one mildly
    off-policy transition per occurrence, which is a far smaller error than half the
    episodes terminating early at an arbitrary step with a fabricated reward of 0."""


# Substrings identifying the command-validity family. Deliberately narrow: a genuine
# contact reflex (cartesian_reflex, joint_position_limits_violation, ...) must still end
# the episode, because there the scene really has changed.
DISCONTINUITY_MARKERS = (
    "discontinuity",
    "velocity_violation",
    "velocity_discontinuity",
    "acceleration_discontinuity",
)


def _motion_error(e) -> MotionAborted:
    """MotionDiscontinuity for the command-validity family, MotionAborted otherwise."""
    msg = str(e)
    low = msg.lower()
    if any(m in low for m in DISCONTINUITY_MARKERS):
        return MotionDiscontinuity(msg)
    return MotionAborted(msg)

# ============================================================================
# Configuration
# ============================================================================

# Policy server (model runs here)
SERVER_IP = "localhost"  # IP address of the machine running the policy server
SERVER_PORT = 8000

# RealSense cameras (on real-time kernel machine)
# IMPORTANT: Set these to specify which camera is base vs wrist
# 
# Base camera: Usually the third-person/external camera viewing the scene
# Wrist camera: Usually mounted on the robot's end-effector/wrist
#
# To find your camera serial numbers:
#   1. Run: rs-enumerate-devices
#   2. Or uncomment list_realsense_cameras() in main() and run the script
#
# Then set them here:
BASE_CAMERA_SN = "234322305598"  # Serial number of BASE camera (third-person view)
# BASE_CAMERA_SN = "241122302482"  # Serial number of BASE camera (third-person view)
WRIST_CAMERA_SN = "241122306284"  # Serial number of WRIST camera (mounted on robot)
#
# If both are None, the code will auto-detect:
#   - First camera found = base camera
#   - Second camera found = wrist camera

# Franka robot (on real-time kernel machine)
FRANKA_IP = "192.168.1.10"  # IP address of your Franka robot

# Task instruction
TASK_INSTRUCTION = "pick up the carrot and place it in the pot"

# Control parameters
MAX_STEPS = 500  # Maximum number of control steps

# Control frequency (Hz) - How often to execute actions on the robot
# Common values:
#   - 10 Hz  = 0.1s per step  (slower, more stable)
#   - 20 Hz  = 0.05s per step (recommended, good balance)
#   - 30 Hz  = 0.033s per step (faster, requires good real-time performance)
#   - 50 Hz  = 0.02s per step  (very fast, needs excellent real-time kernel)
#
# Note: This should match your training data frequency if possible.
# Franka robots typically support 20-50 Hz control.
CONTROL_FREQUENCY = 30.0  # Control frequency in Hz (20 Hz = 0.05s per step)

# Velocity limits (adjust based on your needs)
MAX_LINEAR_VELOCITY = 0.8  # m/s for x, y, z
MAX_ANGULAR_VELOCITY = 1.0 # rad/s for roll, pitch, yaw
GRIPPER_SPEED = 0.05  # m/s
GRIPPER_FORCE = 50  # N, grasp_async clamping force
"""Grip force for grasp_async -- the long-standing value, unchanged.

Named rather than repeated at the two call sites in execute_gripper_action_hold /
_toggle, which previously both hardcoded 50 by coincidence of editing rather than by
construction and could silently drift apart.

It was briefly raised to 70 while chasing a session where the book slipped out of the
fingers as they closed. That turned out to be robot state, not grip force -- a reboot
fixed it, with the robot still running 50 -- so 70 was never validated on hardware and
the value is back where it was. If a grasp really is slipping, check the robot's state
first (see the reboot note), then whether the fingers are closing on an edge rather
than a face; force is the last thing to reach for, and the Franka Hand's documented
continuous maximum is the ceiling."""
MAX_GRIPPER_WIDTH = 0.1  # meters (8cm)
MAX_JOINT_VELOCITY = 0.2
MAX_JOINT_POS_CHANGE = 0.1
MIN_HEIGHT = 0.04


JOINT_CONTROL = 'cartesian_vel' # cartesian_vel or joint_vel
GRIPPER_CONTROL = 'toggle' # toggle or hold

# Give up only if the arm fails this many times in a row WITHOUT a successful step in
# between. Individual ControlExceptions are normal (the reflex trips whenever the book
# snags) and are reported as episode failures so RL training continues; a run of them
# with no progress means the robot is stuck in an unrecoverable state and spinning on it
# would silently feed garbage failures into the replay buffer.
MAX_CONSECUTIVE_ERRORS = 10

# Command-validity aborts (MotionDiscontinuity) are NOT episode failures -- see that
# class. They were ending roughly half of all episodes early, each one recorded as a
# reward-0 failure at an arbitrary step, which is far more damaging to the replay buffer
# than the one skipped command that ignoring them costs. Set False to restore the old
# behaviour (every abort ends the episode).
IGNORE_MOTION_DISCONTINUITY = True
# ...but not forever: if the controller rejects this many commands in a row it is not a
# transient, and continuing would feed the trainer a long run of actions the arm never
# executed. Escalates to a normal episode failure at that point.
MAX_CONSECUTIVE_DISCONTINUITIES = 20
# Settle time after recovery, INSIDE an episode. Keep this small: the arm is stopped
# while it waits, so a long pause silently stretches the wall-clock gap between commands
# (execute_cartesion_vel_action issues velocity motions with no duration, so the trainer's
# step rate is part of the dynamics). 0 disables.
DISCONTINUITY_SETTLE_S = 0.05

# Pause after the arm reaches home, before the next episode's first observation is sent,
# so the scene can be reset by hand (book back on the table, etc.). The trainer is blocked
# on this robot's reply for the whole pause, so no stale post-episode frame gets recorded
# as the new episode's starting state. Set to 0 to disable.
SCENE_RESET_DELAY = 3.0  # seconds

# ============================================================================
# Helper Functions
# ============================================================================

def list_realsense_cameras():
    """List all available RealSense cameras and their serial numbers.
    
    Run this function to find your camera serial numbers, then set
    BASE_CAMERA_SN and WRIST_CAMERA_SN above.
    
    The function will help you identify which camera should be base vs wrist.
    """
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        print("\n" + "=" * 70)
        print("Available RealSense Cameras:")
        print("=" * 70)
        for i, device in enumerate(devices):
            sn = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"\n  Camera {i+1}:")
            print(f"    Name: {name}")
            print(f"    Serial Number: {sn}")
            print(f"    Usage: {'BASE (third-person view)' if i == 0 else 'WRIST (robot-mounted)' if i == 1 else 'Available'}")
        
        print("\n" + "=" * 70)
        print("To configure cameras, set in the configuration section:")
        print("=" * 70)
        if len(devices) >= 1:
            print(f"\n  BASE_CAMERA_SN = \"{devices[0].get_info(rs.camera_info.serial_number)}\"  # Camera 1")
        if len(devices) >= 2:
            print(f"  WRIST_CAMERA_SN = \"{devices[1].get_info(rs.camera_info.serial_number)}\"  # Camera 2")
        print("\nNote: You can swap these if Camera 1 should be wrist and Camera 2 should be base.")
        print("=" * 70 + "\n")
        return [d.get_info(rs.camera_info.serial_number) for d in devices]
    except Exception as e:
        print(f"Error listing cameras: {e}")
        return []


# ============================================================================
# RealSense Camera Interface
# ============================================================================

class RealSenseCameras:
    """Manages two RealSense cameras for base and wrist views."""
    
    def __init__(self, base_sn: Optional[str] = None, wrist_sn: Optional[str] = None):
        """Initialize RealSense cameras.
        
        Args:
            base_sn: Serial number of base camera (None to auto-detect)
            wrist_sn: Serial number of wrist camera (None to auto-detect)
        """
        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        
        if len(devices) < 2:
            raise RuntimeError(f"Found {len(devices)} RealSense device(s), need at least 2")
        
        device_sns = [d.get_info(rs.camera_info.serial_number) for d in devices]
        logger.info(f"Found RealSense devices: {device_sns}")
        
        # Assign cameras based on configuration
        # BASE_CAMERA_SN and WRIST_CAMERA_SN determine which camera is which
        if base_sn is None:
            base_sn = device_sns[0]
            logger.info(f"Auto-detected base camera: {base_sn} (first camera found)")
        if wrist_sn is None:
            wrist_sn = device_sns[1] if len(device_sns) > 1 else device_sns[0]
            logger.info(f"Auto-detected wrist camera: {wrist_sn} (second camera found)")
        
        if base_sn not in device_sns:
            raise RuntimeError(f"Base camera SN {base_sn} not found. Available cameras: {device_sns}")
        if wrist_sn not in device_sns:
            raise RuntimeError(f"Wrist camera SN {wrist_sn} not found. Available cameras: {device_sns}")
        
        if base_sn == wrist_sn:
            raise RuntimeError(f"Base and wrist cameras cannot be the same! Both set to: {base_sn}")
        
        self.base_sn = base_sn
        self.wrist_sn = wrist_sn
        
        logger.info(f"Camera assignment:")
        logger.info(f"  BASE camera (third-person view): SN {base_sn}")
        logger.info(f"  WRIST camera (robot-mounted): SN {wrist_sn}")
        
        # Configure and start pipelines
        self.base_pipeline = rs.pipeline()
        self.wrist_pipeline = rs.pipeline()
        
        base_config = rs.config()
        base_config.enable_device(base_sn)
        base_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        wrist_config = rs.config()
        wrist_config.enable_device(wrist_sn)
        wrist_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        logger.info(f"Starting base camera (SN: {base_sn})")
        self.base_pipeline.start(base_config)
        
        logger.info(f"Starting wrist camera (SN: {wrist_sn})")
        self.wrist_pipeline.start(wrist_config)
        
        # Warm up cameras
        logger.info("Warming up cameras...")
        for _ in range(10):
            self.base_pipeline.wait_for_frames()
            self.wrist_pipeline.wait_for_frames()
        logger.info("Cameras ready!")
    
    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture images from both cameras.
        
        Returns:
            Tuple of (base_image, wrist_image) as numpy arrays (H, W, 3) in RGB format.
            RealSense outputs BGR, so we convert to RGB here.
        """
        # Get frames
        base_frames = self.base_pipeline.wait_for_frames()
        wrist_frames = self.wrist_pipeline.wait_for_frames()
        
        # Extract color frames
        base_color_frame = base_frames.get_color_frame()
        wrist_color_frame = wrist_frames.get_color_frame()
        if not base_color_frame or not wrist_color_frame:
            raise RuntimeError("Failed to get color frames")
        
        # Convert to numpy arrays (RealSense outputs BGR format)
        base_image = np.asanyarray(base_color_frame.get_data())
        wrist_image = np.asanyarray(wrist_color_frame.get_data())
        
        # Convert BGR to RGB (RealSense outputs BGR, but model expects RGB)
        base_image = base_image[..., ::-1]
        wrist_image = wrist_image[..., ::-1]
        
        cv2.imshow("Base Camera", base_image[..., ::-1])
        cv2.imshow("Wrist Camera", wrist_image[..., ::-1])
        cv2.waitKey(1)  # important for window refresh
        
        return base_image, wrist_image
    
    def stop(self):
        """Stop camera pipelines."""
        self.base_pipeline.stop()
        self.wrist_pipeline.stop()
        logger.info("Cameras stopped")


# ============================================================================
# Franka Robot Interface using franky
# ============================================================================

class FrankaRobot:
    """Manages Franka robot using franky library.
    
    - Reads robot state (qpos + gripper) using franky
    - Controls robot with end-effector velocity control using franky
    """
    
    def __init__(self, robot_ip: str):
        """Initialize Franka robot connection.
        
        Args:
            robot_ip: IP address of the Franka robot
        """
        try:
            import franky
            self.franky = franky
            logger.info("Using franky for Franka control")
            
            # Initialize robot and gripper
            self.robot = franky.Robot(robot_ip)
            self.gripper = franky.Gripper(robot_ip)
            self.gripper_closed = False
            self.gripper_speed = GRIPPER_SPEED
            self._gripper_target = None
            self._gripper_future = None
            # self.close_gripper = False
            # self.open_gripper = False
            # self.gripper_open = True
            logger.info(f"Connected to Franka robot at {robot_ip}")
        except ImportError:
            raise RuntimeError(
                "franky library not found. Install it with: pip install franky"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Franka robot at {robot_ip}: {e}")
        
    def go_home(self) -> bool:
        """Move the robot to the pre-set home position. Returns True on success.

        go_home() is called EXACTLY when the robot is most likely to be in a
        reflex-latched error state (the reset that follows a failed episode). After a
        reflex trip libfranka rejects every subsequent motion with "move command aborted
        ... motion aborted by reflex" until automaticErrorRecovery() runs -- so this must
        clear the latch BEFORE commanding the homing motion, not assume a clean state.
        The previous version went straight to move(), caught the resulting exception, and
        logged "Failed to go home" while returning normally, so the caller believed the
        arm was homed when it had not moved at all.
        """
        time.sleep(2)
        home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        # Clear any latched error first -- a no-op when the robot is already healthy.
        self.recover_from_errors()
        for attempt in (1, 2):
            try:
                logger.info(f"Moving home (attempt {attempt}/2)")

                # moving speed
                self.robot.relative_dynamics_factor = 0.2
                motion = self.franky.JointMotion(home_joints, ReferenceType.Absolute)
                self.robot.move(motion)

                # wait for any in-flight async gripper op to finish before issuing
                # the blocking open — otherwise grasp_async can complete *after*
                # move() and leave the gripper closed at the start of the next episode
                if self._gripper_future is not None:
                    try:
                        self._gripper_future.wait()
                    except Exception:
                        pass
                    self._gripper_future = None
                self._gripper_target = None
                self.gripper.move(width=self.gripper.max_width, speed=0.05)
                self.gripper_closed = False
                logger.info("Robot reached home position")
                if SCENE_RESET_DELAY > 0:
                    logger.info(f">>> RESET THE SCENE — resuming in {SCENE_RESET_DELAY:.0f}s <<<")
                    time.sleep(SCENE_RESET_DELAY)
                return True
            # NOTE both clauses are required. franky's ControlException comes from pybind11
            # and is NOT guaranteed to derive from Python's Exception -- when it derives
            # from BaseException instead, a bare `except Exception` does not catch it and
            # the traceback escapes straight out of robot.move(). Listing the class
            # explicitly (as the execute_*/get_state guards already do) is what actually
            # catches a reflex trip here.
            except (self.franky.ControlException, Exception) as e:
                logger.error(f"go_home attempt {attempt}/2 failed: {e}")
                self.recover_from_errors()
                time.sleep(1.0)
        logger.error(
            "go_home FAILED after recovery + retry — the arm did not reach home. The next "
            "episode would start from an arbitrary pose, so this is reported to the caller."
        )
        return False

    def get_state(self) -> np.ndarray:
        """Get current robot state using franky.
        
        Returns:
            Robot state as numpy array: [qpos (7), gripper_width (1)]
            - qpos: 7 joint positions
            - gripper_width: gripper opening width
        """
        # Use franky to read robot stategripper_action.
        # NOTE this is a common place for a ControlException to SURFACE even though
        # nothing here commands motion: execute_*_action() issues its motions with
        # asynchronous=True, so a reflex trip during that motion is not raised at the
        # move() call -- franky reports it on a subsequent interaction, typically the
        # next state query. Convert it to MotionAborted like the motion methods do, so
        # the caller can treat it as an episode failure instead of crashing.
        try:
            eef_pose = self.robot.current_cartesian_state.pose.end_effector_pose
            eef_pos = eef_pose.translation
            eef_quat = eef_pose.quaternion
            gripper_qpos = np.array([self.gripper.width])  # Gripper width
        except self.franky.ControlException as e:
            logger.error(f"Robot error surfaced while reading state (treating as episode failure): {e}")
            raise _motion_error(e) from e
        except Exception as e:
            logger.error(f"Robot error surfaced while reading state (treating as episode failure): {e}")
            raise _motion_error(e) from e
        return eef_pos, eef_quat, gripper_qpos
    
    def execute_gripper_action_hold(self, gripper_action):
        try:
            if gripper_action < 0:  # close
                if self._gripper_target != 'close':
                    self._gripper_target = 'close'
                    self.gripper_closed = True
                    self._gripper_future = self.gripper.grasp_async(width=0.0, speed=self.gripper_speed, force=GRIPPER_FORCE, epsilon_outer=1.0)
            else:  # open
                if self._gripper_target != 'open':
                    self._gripper_target = 'open'
                    self.gripper_closed = False
                    self._gripper_future = self.gripper.open_async(self.gripper_speed)
        except self.franky.ControlException as e:
            logger.error(f"Gripper control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
        except Exception as e:
            logger.error(f"Gripper control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e

    def execute_gripper_action_toggle(self, gripper_action):
        gripper_cmd = 0
        if gripper_action < -0.5:  # close
            gripper_cmd = -1
        elif gripper_action > 0.5:  # open
            gripper_cmd = 1

        try:
            if gripper_cmd == -1 and self._gripper_target != 'close':
                self._gripper_target = 'close'
                self.gripper_closed = True
                self._gripper_future = self.gripper.grasp_async(width=0.0, speed=self.gripper_speed, force=GRIPPER_FORCE, epsilon_outer=1.0)
            elif gripper_cmd == 1 and self._gripper_target != 'open':
                self._gripper_target = 'open'
                self.gripper_closed = False
                self._gripper_future = self.gripper.open_async(self.gripper_speed)
        except self.franky.ControlException as e:
            logger.error(f"Gripper control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
        except Exception as e:
            logger.error(f"Gripper control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e

    
    def execute_cartesion_vel_action(self, action: np.ndarray, dt: float = 0.05):
        """Execute action on robot using franky end-effector velocity control.
        
        Args:
            action: Action array of shape (7,): [x_vel, y_vel, z_vel, roll_vel, pitch_vel, yaw_vel, gripper]
                   All velocities are in normalized [-1, 1] range
            dt: Time step for execution (default 0.05s for 20Hz)
        """
        # if len(action) < 8:
        #     raise ValueError(f"Action must have 7 dimensions, got {len(action)}")
        
        # Parse action: [x, y, z, roll, pitch, yaw velocities, gripper]
        # detla_joint_angle = action[:7]  # End-effector velocities
        # gripper_action = action[7] # Gripper command
        # # Clip actions to safe ranges
        # ee_velocities = np.clip(ee_velocities, -1.0, 1.0)
        # gripper_action = np.clip(gripper_action, -1.0, 1.0)
        
        # Scale from normalized [-1, 1] to actual velocities
        linear_vel = action[:3] * MAX_LINEAR_VELOCITY  # x, y, z in m/s
        angular_vel = action[3:6] * MAX_ANGULAR_VELOCITY  # roll, pitch, yaw in rad/s
        # # linear_vel = ee_velocities[:3]  # x, y, z in m/s
        # # angular_vel = ee_velocities[3:6]  # roll, pitch, yaw in rad/s
        # Combine into 6D twist vector [vx, vy, vz, wx, wy, wz]

        # Execute end-effector velocity control using franky.
        # NOTE the try MUST cover the workspace-constraint STATE QUERY below, not just the
        # move() call: motions are issued with asynchronous=True, so a reflex trip during
        # the PREVIOUS step's motion is not raised by that move() -- franky surfaces it at
        # the next robot interaction, which is this current_cartesian_state read. It used
        # to sit one line ABOVE the try, leaving the single most likely raise site in the
        # whole loop unguarded; the ControlException then escaped as-is (never becoming a
        # MotionAborted) and killed the main loop instead of failing just the episode.
        try:
            # constrain workspace
            z = self.robot.current_cartesian_state.pose.end_effector_pose.translation[-1]
            delta_z = linear_vel[-1] * (1 / CONTROL_FREQUENCY)
            if delta_z < 0.0 and z + delta_z <= MIN_HEIGHT:
                linear_vel[-1] = 0.0
                print("CONSTRAINED")

            twist = np.concatenate([linear_vel, angular_vel])
            print("actions: ", twist)
            vel = RobotVelocity(Twist(twist[:3],twist[3:]),)
            # detla_joint_angle = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            # motion = self.franky.JointMotion(detla_joint_angle, ReferenceType.Relative, return_when_finished=True)
            # # motion = self.franky.CartesianMotion(Affine(linear_vel), ReferenceType.Relative)
            motion = self.franky.CartesianVelocityMotion(vel, relative_dynamics_factor=0.1)
            self.robot.relative_dynamics_factor = 0.1
            # self.robot.relative_dynamics_factor = RelativeDynamicsFactor(0.05, 0.05, 0.05)
            self.robot.move(motion, asynchronous=True)
            time.sleep(0.0005)
        except self.franky.ControlException as e:
            logger.error(f"Cartesian velocity control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
        except Exception as e:
            logger.error(f"Cartesian velocity control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
        
    def execute_joint_vel_action(self, action: np.ndarray, dt: float = 0.05):
        """Execute action on robot using franky joint velocity control.
        
        Args:
            action: Action array of shape (8,): [7d joint velocity, gripper]
                   All velocities are in normalized [-1, 1] range
            dt: Time step for execution (default 0.05s for 20Hz)
        """
        
        # Parse action
        action = action * MAX_JOINT_VELOCITY

        # # constrain workspace (NOTE: not certain if math is correct, uncomment to try)
        # z = self.robot.current_cartesian_state.pose.end_effector_pose.translation[-1]
        # J_z = self.robot.model.body_jacobian(Frame.EndEffector, self.robot.state)[2, :]
        # z_vel = J_z @ action
        # delta_z = z_vel * (1 / CONTROL_FREQUENCY)
        # if delta_z < 0.0 and z + delta_z <= MIN_HEIGHT: 
        #     action = action - (z_vel / (J_z @ J_z)) * J_z
        #     print("CONSTRAINED")

        print("actions: ", action)
        # Execute joint velocity control using franky
        try:
            motion = self.franky.JointVelocityMotion(action)
            self.robot.relative_dynamics_factor = RelativeDynamicsFactor(0.05, 0.05, 0.05)
            self.robot.move(motion, asynchronous=True)
            time.sleep(0.0005)
        except self.franky.ControlException as e:
            logger.error(f"Joint velocity control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
        except Exception as e:
            logger.error(f"Joint velocity control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
        
    def execute_joint_pos_action(self, action: np.ndarray, dt: float = 0.05):
        """Execute action on robot using franky joint position control.
        
        Args:
            action: Action array of shape (8,): [7d joint position, gripper]
                   All velocities are in normalized [-1, 1] range
            dt: Time step for execution (default 0.05s for 20Hz)
        """
        
        # Parse action
        joint_positions = action[:7] * MAX_JOINT_POS_CHANGE
        gripper_action = action[7]
        
        print("actions: ", joint_positions)
        self.execute_gripper_action(gripper_action)
        # Execute joint velocity control using franky
        try:
            motion = self.franky.JointMotion(joint_positions)
            self.robot.relative_dynamics_factor = RelativeDynamicsFactor(0.05, 0.05, 0.05)
            self.robot.move(motion, asynchronous=True)
            time.sleep(0.0005)
        except self.franky.ControlException as e:
            logger.error(f"Joint position control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
        except Exception as e:
            logger.error(f"Joint position control aborted (treating as episode failure): {e}")
            raise _motion_error(e) from e
    
    def recover_from_errors(self):
        """Clear libfranka's reflex/error state after a MotionAborted so the NEXT
        move() command (e.g. go_home()) doesn't immediately fail with the same
        error. NOTE: verify this method name against your installed franky version
        (`python -c "import franky; print([m for m in dir(franky.Robot) if 'recover' in m.lower() or 'error' in m.lower()])"`)
        -- this mirrors libfranka's automaticErrorRecovery(), which franky
        typically exposes as Robot.recover_from_errors(), but bindings differ
        across versions."""
        try:
            self.robot.recover_from_errors()
            logger.info("Recovered from robot error/reflex state")
        except (self.franky.ControlException, Exception) as e:
            # Both clauses required -- see the note in go_home(): ControlException may not
            # be an Exception subclass. This method is the LAST line of defence, so it must
            # never itself propagate.
            logger.warning(f"recover_from_errors() failed or unsupported: {e}")

    def stop(self):
        """Stop robot and gripper."""
        # Both clauses required in each -- see go_home(): ControlException may not derive
        # from Exception. stop() runs in the `finally` of main(), so an escape here would
        # replace the real error with a shutdown traceback.
        if self.robot is not None:
            try:
                self.robot.stop()
            except (self.franky.ControlException, Exception) as e:
                logger.warning(f"Error stopping robot: {e}")
        if hasattr(self, 'gripper') and self.gripper is not None:
            try:
                self.gripper.stop()
            except (self.franky.ControlException, Exception) as e:
                logger.warning(f"Error stopping gripper: {e}")
        logger.info("Robot stopped")

class KeyboardInputManager:
    def __init__(self):
        self.reward_queue = queue.Queue()
        self.done_queue = queue.Queue()
        self.pause_queue = queue.Queue()
        self.running = True

        self.thread = threading.Thread(
            target=self._keyboard_listener,
            daemon=True
        )
        self.thread.start()

    def _keyboard_listener(self):
        print("Keyboard listener started.")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while self.running:
                key = sys.stdin.read(1)
                print('KEY', repr(key))

                if key == '\n': # enter
                    self.reward_queue.put(1.0)
                    self.done_queue.put(True)
                elif key == '\x7f': # backspace
                    self.done_queue.put(True)
                elif key == ' ': # space
                    self.pause_queue.put(True)

        finally:
            termios.tcsetattr(
                fd,
                termios.TCSADRAIN,
                old_settings
            )


    def get_reward(self):
        reward = 0.0
        while not self.reward_queue.empty():
            reward = self.reward_queue.get()
        return reward
    
    def get_done(self):
        done = False
        while not self.done_queue.empty():
            done = self.done_queue.get()
        return done
    
    def get_pause(self):
        pause = False
        while not self.pause_queue.empty():
            pause = self.pause_queue.get()
        return pause



# ============================================================================
# Main Inference Loop
# ============================================================================

def main():
    """Main inference loop running on real-time kernel."""
    
    logger.info("=" * 60)
    logger.info("Franka Robot Inference - Real-Time Kernel")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # 1. Initialize RealSense cameras
    cameras = RealSenseCameras(BASE_CAMERA_SN, WRIST_CAMERA_SN)
    
    # 2. Initialize Franka robot (using franky)
    robot = FrankaRobot(FRANKA_IP)
    
    # 3. Initialize policy client (connects to server)
    logger.info("=" * 60)
    logger.info(f"Connecting to policy server at {SERVER_IP}:{SERVER_PORT}")
    logger.info("=" * 60)
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_IP, port=SERVER_PORT)
        logger.info("✓ WebSocket client created")
    except Exception as e:
        logger.error(f"✗ Failed to create WebSocket client: {e}")
        raise
    
    # Verify connection and get server metadata
    try:
        logger.info("Requesting server metadata...")
        metadata = client.get_server_metadata()
        logger.info("✓ Successfully connected to policy server!")
        logger.info(f"Server metadata: {metadata}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"✗ Failed to connect to policy server: {e}")
        logger.error("Make sure the policy server is running on the server machine.")
        logger.error(f"  Server IP: {SERVER_IP}")
        logger.error(f"  Server Port: {SERVER_PORT}")
        raise

    # The robot_bridge_server's reset_to() pushes a "reset" command and then
    # blocks on the NEXT message this client sends, using it as the settled
    # post-reset observation. Without sending this ready ping first, the main
    # loop's first real infer() call (a real, possibly stale/un-homed
    # observation -- e.g. wherever the arm was left after the previous
    # episode) gets consumed as that reset acknowledgment instead, and the
    # action computed from it comes back as an ordinary "step" command -- so
    # it gets executed directly, without the arm ever going home first. This
    # is what caused the gripper to slam shut at the very start of a run.
    logger.info("Sending ready ping to bridge server...")
    ready_result = client.infer(None, training=True, reward=None, done=None)
    if ready_result.get('type') == 'reset':
        if not robot.go_home():
            # Refuse to start a run from an unknown pose -- the first episode's data would
            # be garbage and the arm may still be latched in an error state.
            raise MotionAborted("initial go_home failed — robot not at a known start pose")
    else:
        logger.warning(f"Expected 'reset' in response to ready ping, got: {ready_result.get('type')}")

    input_manager = KeyboardInputManager()

    # Control variables
    action = None
    step_duration = 1.0 / CONTROL_FREQUENCY

    last_send_time = 0.0
    
    logger.info(f"Starting inference loop")
    logger.info(f"Control frequency: {CONTROL_FREQUENCY} Hz")
    logger.info("-" * 60)
    logger.info("Server Communication:")
    logger.info(f"  - Server IP: {SERVER_IP}")
    logger.info(f"  - Server Port: {SERVER_PORT}")
    logger.info("-" * 60)

    step = 0
    consecutive_errors = 0  # reset by any step that completes; see MAX_CONSECUTIVE_ERRORS
    consecutive_discontinuities = 0  # see MAX_CONSECUTIVE_DISCONTINUITIES
    try:
        while True:
            step_start_time = time.time()
            

            try:
                base_image, wrist_image = cameras.get_images()

                # 2. Get robot state using franky (qpos + gripper)
                eef_pos, eef_quat, gripper_qpos = robot.get_state()
            except MotionDiscontinuity as e:
                # Command-validity rejection, not a contact reflex: recover and keep the
                # episode running. Ordered BEFORE `except MotionAborted` -- it is a
                # subclass, so the broader handler would otherwise swallow it first.
                if not IGNORE_MOTION_DISCONTINUITY:
                    raise
                consecutive_discontinuities += 1
                if consecutive_discontinuities >= MAX_CONSECUTIVE_DISCONTINUITIES:
                    logger.error(
                        f"{consecutive_discontinuities} consecutive command-validity aborts — "
                        f"not a transient; failing the episode rather than reporting actions "
                        f"the arm never executed."
                    )
                    raise MotionAborted(str(e)) from e
                if consecutive_discontinuities == 1 or consecutive_discontinuities % 10 == 0:
                    logger.warning(
                        f"Motion discontinuity #{consecutive_discontinuities} (ignored, "
                        f"episode continues): {e}"
                    )
                robot.recover_from_errors()
                if DISCONTINUITY_SETTLE_S > 0:
                    time.sleep(DISCONTINUITY_SETTLE_S)
                continue

            except MotionAborted as e:
                # A reflex trip from the PREVIOUS step's async motion surfaces here (see
                # get_state). Recover and report the episode as a failure -- exactly like
                # the execute-block's handler below -- instead of letting it reach the
                # outer `except Exception`, which exits the loop and ends training.
                logger.warning(f"Robot error before inference -- auto-reporting as failure: {e}")
                robot.recover_from_errors()
                input_manager.done_queue.put(True)
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        f"{consecutive_errors} consecutive robot errors with no successful step "
                        f"in between — the arm is not recovering; stopping rather than spinning."
                    )
                    raise
                time.sleep(1.0)  # let the controller settle before re-arming
                continue

            # Prepare observation
            observation = {
                "agentview_image": base_image[::-1, ::-1],
                "robot0_eye_in_hand_image": wrist_image[::-1, ::-1],
                "robot0_eef_pos": eef_pos,
                "robot0_eef_quat": eef_quat,
                "robot0_gripper_qpos": gripper_qpos,
                # "prompt": TASK_INSTRUCTION,
            }
            #     observation = {
            #         "observation/exterior_image_1_left": base_image[::-1, ::-1],
            #         "observation/wrist_image_left": wrist_image[::-1, ::-1],
            #         "observation/joint_position": np.concatenate([eef_pos, eef_quat]),  # qpos (unnormalized, server handles normalization)
            #         "observation/gripper_position": gripper_qpos,
            #         # "prompt": TASK_INSTRUCTION,
            #     }
            
            inference_start = time.time()
            try:
                reward = input_manager.get_reward()
                done = input_manager.get_done()
                print('reward: ', reward)
                result = client.infer(
                    observation,
                    training=True,
                    reward=reward,
                    done=done
                )
                # print(result["actions"])
                inference_time = (time.time() - inference_start) * 1000  # ms
                logger.info(f"✓ Received response from server (took {inference_time:.1f}ms)")

                pause = input_manager.get_pause()
                if pause:
                    while not input_manager.get_pause():
                        time.sleep(0.5)
            except Exception as e:
                logger.error(f"✗ Failed to get response from server: {e}")
                raise
            
            if result['type'] == 'reset':
                if robot.go_home():
                    # Homed cleanly, so the arm is healthy again regardless of what
                    # failed before this reset.
                    consecutive_errors = 0
                    consecutive_discontinuities = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            f"go_home failed {consecutive_errors} times in a row — the arm is not "
                            f"recovering (check for a physical obstruction or an E-stop); stopping."
                        )
                        raise MotionAborted("go_home repeatedly failed")
                    logger.warning("continuing after failed go_home; will retry on the next reset")
                continue
            action = result["actions"]
            
            # Validate action dimensions
            if JOINT_CONTROL == 'cartesian_vel' and action.shape != (7,):
                logger.warning(
                    f"⚠ Expected action dimension 7 (x,y,z,roll,pitch,yaw,gripper), "
                    f"got {action.shape}."
                )
            elif JOINT_CONTROL == 'joint_vel' and action.shape[-1] != (8,):
                logger.warning(
                    f"⚠ Expected action dimension 8 (7d joint velocity + gripper), "
                    f"got {action.shape[-1]}."
                )
            else:
                logger.info(
                    f"✓ Action received: shape {action.shape}, "
                    f"inference time: {inference_time:.1f}ms"
                )
            
            # 5. Execute action on robot using franky (end-effector velocity control)
            
            # now = time.time()
            # if now - last_send_time >= 0.2:
            joint_action = action[:-1]
            gripper_action = action[-1]
            # Gripper diagnostic: execute_gripper_action_* only fires the hardware
            # call when this RAW value crosses -+0.5, so a policy whose gripper
            # channel is being squashed upstream (normalization / RL ActionSpace
            # round-trip) shows up here as a value pinned near 0 that never
            # triggers. Log the value and the decision every step so "the arm
            # reaches the object but never grasps" is directly attributable.
            _g_decision = "CLOSE" if gripper_action < -0.5 else ("OPEN" if gripper_action > 0.5 else "no-op")
            logger.info(
                f"gripper raw={gripper_action:+.4f} -> {_g_decision} "
                f"(|v|/threshold={abs(gripper_action) / 0.5:.2f}, target={robot._gripper_target})"
            )
            try:
                if JOINT_CONTROL == 'cartesian_vel':
                    robot.execute_cartesion_vel_action(joint_action, dt=step_duration)
                    # last_send_time = now
                elif JOINT_CONTROL == 'joint_vel':
                    robot.execute_joint_vel_action(action, dt=step_duration)

                if GRIPPER_CONTROL == 'toggle':
                    robot.execute_gripper_action_toggle(gripper_action)
                elif GRIPPER_CONTROL == 'hold':
                    robot.execute_gripper_action_hold(gripper_action)
            except MotionDiscontinuity as e:
                # Command-validity rejection, not a contact reflex: recover and keep the
                # episode running. Ordered BEFORE `except MotionAborted` -- it is a
                # subclass, so the broader handler would otherwise swallow it first.
                if not IGNORE_MOTION_DISCONTINUITY:
                    raise
                consecutive_discontinuities += 1
                if consecutive_discontinuities >= MAX_CONSECUTIVE_DISCONTINUITIES:
                    logger.error(
                        f"{consecutive_discontinuities} consecutive command-validity aborts — "
                        f"not a transient; failing the episode rather than reporting actions "
                        f"the arm never executed."
                    )
                    raise MotionAborted(str(e)) from e
                if consecutive_discontinuities == 1 or consecutive_discontinuities % 10 == 0:
                    logger.warning(
                        f"Motion discontinuity #{consecutive_discontinuities} (ignored, "
                        f"episode continues): {e}"
                    )
                robot.recover_from_errors()
                if DISCONTINUITY_SETTLE_S > 0:
                    time.sleep(DISCONTINUITY_SETTLE_S)
                continue

            except MotionAborted as e:
                # e.g. the book snags on the bookholder's edge and libfranka's reflex
                # trips -- report this exactly like a manual "backspace" failure
                # keypress instead of crashing: reward stays 0.0 (default, nobody
                # pushes to reward_queue), done=True gets picked up by the NEXT
                # client.infer() call, which should get back a `reset` command,
                # handled by the existing `result['type'] == 'reset'` branch above.
                logger.warning(f"Motion aborted mid-episode -- auto-reporting as failure: {e}")
                robot.recover_from_errors()
                input_manager.done_queue.put(True)
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        f"{consecutive_errors} consecutive robot errors with no successful step "
                        f"in between — the arm is not recovering; stopping rather than spinning."
                    )
                    raise
                time.sleep(1.0)  # let the controller settle before re-arming
                continue

            # The step completed end-to-end, so whatever tripped earlier has cleared.
            consecutive_errors = 0
            consecutive_discontinuities = 0

            # 6. Maintain control frequency
            elapsed = time.time() - step_start_time
            sleep_time = max(0, step_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if step % 50 == 0:
                logger.info(f"Step {step}/{MAX_STEPS} completed")
                logger.info(f"  - Actions executed: {step}")
            step += 1
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
    finally:
        logger.info("Stopping components...")
        cameras.stop()
        robot.stop()
        logger.info("Inference loop ended")


if __name__ == "__main__":
    # # Uncomment the line below to list available cameras and their serial numbers
    # list_realsense_cameras()
    # exit()
    
    main()