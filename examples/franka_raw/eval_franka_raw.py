#!/usr/bin/env python3
"""
Evaluate the raw franka_raw SFT checkpoint (no RL fine-tuning) on the real robot.

This talks to a plain `scripts/serve_policy.py` inference server -- NOT the RL bridge
(rlt/robot_bridge.py / train_rlt_libero.py --real_robot_ports). No reward/done/reset
protocol here: just observation in, action out, on a loop. Use this to sanity-check the
SFT policy's raw behavior before layering RLT/TD3 on top.

State/action convention matches training exactly (see
examples/franka_raw/convert_franka_raw_to_lerobot.py and src/openpi/policies/franka_policy.py):
  observation/state: 7-dim = ee_pos(3) + axis-angle(3) + gripper_width(1)
  action: 7-dim = [vx, vy, vz, wx, wy, wz, gripper_trigger] (cartesian velocity + trigger)

Setup (same as your RL client):
  pip install pyrealsense2 franky-control
  cd packages/openpi-client && pip install -e .

Usage:
  python eval_franka_raw.py  # edit SERVER_IP/FRANKA_IP/camera serials below first
"""

import logging
import time

from franky import Gripper
from franky import Robot
import numpy as np
import pyrealsense2 as rs
from openpi_client import websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration -- edit these for your setup
# ============================================================================
SERVER_IP = "192.168.1.20"  # machine running `uv run scripts/serve_policy.py ...`
SERVER_PORT = 8000

BASE_CAMERA_SN = "234322305598"  # third-person view
WRIST_CAMERA_SN = "241122306284"  # robot-mounted

FRANKA_IP = "192.168.1.10"

TASK_INSTRUCTION = "pick up the book and place it in the book holder"

MAX_STEPS = 500
CONTROL_FREQUENCY = 30.0  # Hz -- for logging/pacing only; the model doesn't need to match this exactly

MAX_LINEAR_VELOCITY = 0.1  # m/s
MAX_ANGULAR_VELOCITY = 2.5  # rad/s
GRIPPER_SPEED = 0.05  # m/s
MAX_GRIPPER_WIDTH = 0.1  # m
MIN_HEIGHT = 0.04  # m -- simple workspace floor guard


def _quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Same formula as src/openpi/training/rlt/envs.py's `_quat2axisangle` and
    examples/franka_raw/convert_franka_raw_to_lerobot.py's `_quat_wxyz_to_axisangle`
    (verified numerically identical). franky's `.quaternion` is already (x, y, z, w) --
    no reordering needed here (unlike our stored hdf5 data, which is w,x,y,z)."""
    q3 = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - q3 * q3, 0.0))
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(q3)
    return (quat_xyzw[:3] * (angle / den)).astype(np.float32)


class RealSenseCameras:
    def __init__(self, base_sn: str, wrist_sn: str):
        self.base_pipeline = rs.pipeline()
        self.wrist_pipeline = rs.pipeline()
        base_config = rs.config()
        base_config.enable_device(base_sn)
        base_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        wrist_config = rs.config()
        wrist_config.enable_device(wrist_sn)
        wrist_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.base_pipeline.start(base_config)
        self.wrist_pipeline.start(wrist_config)
        logger.info("Warming up cameras...")
        for _ in range(10):
            self.base_pipeline.wait_for_frames()
            self.wrist_pipeline.wait_for_frames()
        logger.info("Cameras ready!")

    def get_images(self) -> tuple[np.ndarray, np.ndarray]:
        base = self.base_pipeline.wait_for_frames().get_color_frame()
        wrist = self.wrist_pipeline.wait_for_frames().get_color_frame()
        base_image = np.asanyarray(base.get_data())[..., ::-1]  # BGR -> RGB
        wrist_image = np.asanyarray(wrist.get_data())[..., ::-1]
        return base_image, wrist_image

    def stop(self):
        self.base_pipeline.stop()
        self.wrist_pipeline.stop()


class FrankaRobot:
    def __init__(self, robot_ip: str):
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)
        self.gripper_closed = False

    def go_home(self):
        home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.robot.relative_dynamics_factor = 0.2
        self.robot.move(__import__("franky").JointMotion(home_joints))
        self.gripper.move(width=self.gripper.max_width, speed=GRIPPER_SPEED)
        self.gripper_closed = False

    def get_state(self) -> np.ndarray:
        """7-dim: ee_pos(3) + axis-angle(3) + gripper_width(1) -- matches training exactly."""
        pose = self.robot.current_cartesian_state.pose.end_effector_pose
        ee_pos = np.asarray(pose.translation, dtype=np.float32)
        ee_quat_xyzw = np.asarray(pose.quaternion, dtype=np.float32)
        axis_angle = _quat_to_axisangle(ee_quat_xyzw)
        gripper_width = np.float32(self.gripper.width)
        return np.concatenate([ee_pos, axis_angle, [gripper_width]])

    def execute_action(self, action: np.ndarray):
        """action: [vx, vy, vz, wx, wy, wz, gripper_trigger] (all in [-1, 1])."""
        import franky

        linear_vel = action[:3] * MAX_LINEAR_VELOCITY
        angular_vel = action[3:6] * MAX_ANGULAR_VELOCITY
        gripper_trigger = action[6]

        z = self.robot.current_cartesian_state.pose.end_effector_pose.translation[-1]
        delta_z = linear_vel[-1] * (1.0 / CONTROL_FREQUENCY)
        if delta_z < 0.0 and z + delta_z <= MIN_HEIGHT:
            linear_vel[-1] = 0.0
            logger.warning("workspace floor guard: z-velocity clamped")

        twist = franky.Twist(linear_vel, angular_vel)
        motion = franky.CartesianVelocityMotion(franky.RobotVelocity(twist), relative_dynamics_factor=0.1)
        self.robot.relative_dynamics_factor = 0.1
        self.robot.move(motion, asynchronous=True)

        if gripper_trigger < -0.5 and not self.gripper_closed:
            self.gripper.grasp(width=0.0, speed=GRIPPER_SPEED, force=50, epsilon_outer=1.0)
            self.gripper_closed = True
        elif gripper_trigger > 0.5 and self.gripper_closed:
            self.gripper.move(width=MAX_GRIPPER_WIDTH, speed=GRIPPER_SPEED)
            self.gripper_closed = False

    def stop(self):
        self.robot.stop()
        self.gripper.stop()


def main():
    cameras = RealSenseCameras(BASE_CAMERA_SN, WRIST_CAMERA_SN)
    robot = FrankaRobot(FRANKA_IP)
    robot.go_home()

    client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_IP, port=SERVER_PORT)
    logger.info(f"Connected. Server metadata: {client.get_server_metadata()}")

    step_duration = 1.0 / CONTROL_FREQUENCY
    try:
        for step in range(MAX_STEPS):
            step_start = time.time()
            base_image, wrist_image = cameras.get_images()
            state = robot.get_state()

            observation = {
                "observation/image": base_image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "prompt": TASK_INSTRUCTION,
            }

            infer_start = time.time()
            result = client.infer(observation)  # plain inference -- no training/reward/done
            infer_ms = (time.time() - infer_start) * 1000

            actions = result["actions"]  # [action_horizon, 7]
            action = actions[0]  # execute one chunk-step per control tick
            robot.execute_action(action)

            if step % 20 == 0:
                logger.info(f"step {step}/{MAX_STEPS}  infer={infer_ms:.1f}ms  action={np.round(action, 3)}")

            elapsed = time.time() - step_start
            time.sleep(max(0.0, step_duration - elapsed))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cameras.stop()
        robot.stop()


if __name__ == "__main__":
    main()
