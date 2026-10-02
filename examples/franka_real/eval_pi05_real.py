#!/usr/bin/env python3
"""
Franka Robot EVALUATION client for the real-time kernel.

Robot-side counterpart of scripts/eval_rlt_real.py. Speaks the identical bridge
protocol as train_pi05_real.py (ready ping -> reset/step commands -> obs + reward/done
reports), so the server cannot tell the two apart -- which is the point: eval must
exercise the same transport and the same action pipeline as training, or the number it
produces is not comparable to the training log's `EVAL rlt` line.

Differences from train_pi05_real.py, all of them robot-side bookkeeping:
  * tracks per-episode outcomes and prints a running success rate + final summary
  * writes a CSV of episode results (--results_csv) for later analysis
  * no pause/resume key (eval runs unattended between resets)
The POLICY differences (deterministic actor mean, no exploration noise) live entirely
on the server -- this script just relays whatever actions it is sent.

Operator keys, same as training:
  ENTER      -> episode SUCCEEDED  (reward 1.0, done)
  BACKSPACE  -> episode FAILED     (reward 0.0, done)
A MotionAborted (libfranka contact reflex) auto-reports a failure, exactly as in
training. A MotionDiscontinuity (command-validity rejection) is ignored and the
episode continues -- also as in training, and for the same reason: it measures the
controller, not the policy.

Usage:
  1. start the server:  uv run scripts/eval_rlt_real.py ... --real_robot_ports 8000
  2. then on the robot: python eval_pi05_real.py --episodes 20
"""

import argparse
import csv
import logging
import queue
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Optional, Tuple

import cv2
from franky import *
import numpy as np
import pyrealsense2 as rs
from openpi_client import websocket_client_policy

# Reuse the robot/camera/keyboard plumbing verbatim so eval and training cannot drift
# apart (same velocity scaling, same +-0.5 gripper threshold, same reflex recovery,
# same go_home + SCENE_RESET_DELAY).
from train_pi05_real import (
    BASE_CAMERA_SN,
    CONTROL_FREQUENCY,
    FRANKA_IP,
    GRIPPER_CONTROL,
    JOINT_CONTROL,
    MAX_CONSECUTIVE_ERRORS,
    SERVER_IP,
    SERVER_PORT,
    WRIST_CAMERA_SN,
    FrankaRobot,
    KeyboardInputManager,
    DISCONTINUITY_SETTLE_S,
    IGNORE_MOTION_DISCONTINUITY,
    MAX_CONSECUTIVE_DISCONTINUITIES,
    MotionAborted,
    MotionDiscontinuity,
    RealSenseCameras,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval -- correct near 0 and 1, where the normal approximation
    produces bounds outside [0, 1] (at the n=20 typical here that matters)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * float(np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20, help="stop after this many completed episodes")
    ap.add_argument("--results_csv", type=str, default="", help="optional per-episode CSV output")
    ap.add_argument(
        "--save_frames_dir",
        type=str,
        default="",
        help="write camera frames to <dir>/ep_<n>_<success|failure>/ (off by default). "
        "Nothing downstream keeps pixels -- the server only stores z_rl/proprio/action/Q -- "
        "so frames for figures MUST be captured here, at run time; they cannot be recovered later.",
    )
    ap.add_argument(
        "--save_frames_every",
        type=int,
        default=1,
        help="stride in control steps between saved frames (1 = every step). At 10 Hz and "
        "~400-step episodes, stride 1 is ~800 PNGs per episode across both cameras.",
    )
    ap.add_argument("--server_ip", type=str, default=SERVER_IP)
    ap.add_argument("--server_port", type=int, default=SERVER_PORT)
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("Franka EVALUATION client — Real-Time Kernel")
    logger.info(f"Target: {args.episodes} episodes")
    logger.info("  ENTER = success   BACKSPACE = failure")
    logger.info("=" * 60)

    cameras = RealSenseCameras(BASE_CAMERA_SN, WRIST_CAMERA_SN)
    robot = FrankaRobot(FRANKA_IP)

    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=args.server_ip, port=args.server_port)
        metadata = client.get_server_metadata()
        logger.info(f"Connected to eval server. Metadata: {metadata}")
    except Exception as e:
        logger.error(f"Failed to connect to eval server at {args.server_ip}:{args.server_port}: {e}")
        raise

    # Ready ping: the bridge's reset_to() pushes a reset command and then consumes the
    # NEXT message as the settled post-reset observation. Without this the first real
    # observation would be swallowed as that ack and the arm would never home first.
    logger.info("Sending ready ping...")
    ready = client.infer(None, training=True, reward=None, done=None)
    if ready.get("type") == "reset":
        robot.go_home()
    else:
        logger.warning(f"Expected 'reset' in response to ready ping, got: {ready.get('type')}")

    input_manager = KeyboardInputManager()
    step_duration = 1.0 / CONTROL_FREQUENCY

    episodes: list[dict] = []  # one record per finished episode
    ep_steps = 0
    ep_start = time.time()
    consecutive_errors = 0
    consecutive_discontinuities = 0
    pending_outcome: Optional[float] = None  # reward of the episode just finished

    frames_root = Path(args.save_frames_dir) if args.save_frames_dir else None
    if frames_root is not None:
        frames_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving camera frames to {frames_root} (every {args.save_frames_every} step(s))")

    def _episode_frame_dir() -> Optional[Path]:
        """Directory for the episode currently in flight. Named by index only, because
        the outcome is not known until it ends; _finish_episode renames it."""
        if frames_root is None:
            return None
        return frames_root / f"ep_{len(episodes) + 1:04d}"

    def _save_frames(base_rgb, wrist_rgb, step: int) -> None:
        d = _episode_frame_dir()
        if d is None or step % max(args.save_frames_every, 1) != 0:
            return
        d.mkdir(parents=True, exist_ok=True)
        # get_images() returns RGB; cv2.imwrite expects BGR, so convert or the saved
        # frames come out with red and blue swapped.
        for name, img in (("base", base_rgb), ("wrist", wrist_rgb)):
            cv2.imwrite(str(d / f"step_{step:05d}_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _finish_episode(reward: float, reason: str):
        nonlocal ep_steps, ep_start
        idx = len(episodes) + 1
        d = _episode_frame_dir()
        if d is not None and d.exists():
            # Stamp the outcome into the directory name so successes and failures sort
            # apart when picking figure frames later.
            d.rename(d.with_name(f"{d.name}_{'success' if reward > 0 else 'failure'}"))
        rec = {
            "episode": idx,
            "success": int(reward > 0),
            "steps": ep_steps,
            "seconds": round(time.time() - ep_start, 1),
            "reason": reason,
        }
        episodes.append(rec)
        n_succ = sum(e["success"] for e in episodes)
        lo, hi = _wilson_ci(n_succ, len(episodes))
        logger.info(
            f"=== EPISODE {idx}/{args.episodes}: {'SUCCESS' if reward > 0 else 'FAILURE'} ({reason}) "
            f"| steps={ep_steps} | running {n_succ}/{len(episodes)} = {n_succ / len(episodes):.3f} "
            f"[95% CI {lo:.2f}-{hi:.2f}] ==="
        )
        ep_steps = 0
        ep_start = time.time()

    try:
        while len(episodes) < args.episodes:
            step_start = time.time()

            base_image, wrist_image = cameras.get_images()
            eef_pos, eef_quat, gripper_qpos = robot.get_state()

            # Save the ROTATED views -- the same pixels the policy is handed, and the
            # right way up for a figure.
            _save_frames(base_image[::-1, ::-1], wrist_image[::-1, ::-1], ep_steps)

            observation = {
                "agentview_image": base_image[::-1, ::-1],
                "robot0_eye_in_hand_image": wrist_image[::-1, ::-1],
                "robot0_eef_pos": eef_pos,
                "robot0_eef_quat": eef_quat,
                "robot0_gripper_qpos": gripper_qpos,
            }

            reward = input_manager.get_reward()
            done = input_manager.get_done()
            if done and pending_outcome is None:
                pending_outcome = reward

            try:
                result = client.infer(observation, training=True, reward=reward, done=done)
            except Exception as e:
                logger.error(f"Failed to get response from eval server: {e}")
                raise

            if result["type"] == "reset":
                # The server acks the episode end by commanding a reset; the outcome we
                # just reported is what it recorded, so book it on the same signal.
                _finish_episode(
                    pending_outcome if pending_outcome is not None else 0.0,
                    "operator" if pending_outcome is not None else "server-reset",
                )
                pending_outcome = None
                if len(episodes) >= args.episodes:
                    break
                robot.go_home()
                continue

            # The server piggybacks per-chunk critic/z_rl telemetry on the first step
            # command of each chunk (see _BridgeSlotProxy.set_telemetry). It arrives
            # pre-rendered as tel["line"] so this machine needs no openpi install --
            # it has only openpi_client. Absent when the server runs --no-monitor or
            # is an older build, hence the .get() chain.
            tel = result.get("telemetry")
            if tel and tel.get("line"):
                logger.info(f"  {tel['line']}")

            action = result["actions"]
            joint_action = action[:-1]
            gripper_action = action[-1]
            g_decision = "CLOSE" if gripper_action < -0.5 else ("OPEN" if gripper_action > 0.5 else "no-op")

            try:
                if JOINT_CONTROL == "cartesian_vel":
                    robot.execute_cartesion_vel_action(joint_action, dt=step_duration)
                elif JOINT_CONTROL == "joint_vel":
                    robot.execute_joint_vel_action(action, dt=step_duration)

                if GRIPPER_CONTROL == "toggle":
                    robot.execute_gripper_action_toggle(gripper_action)
                elif GRIPPER_CONTROL == "hold":
                    robot.execute_gripper_action_hold(gripper_action)
                consecutive_errors = 0
                consecutive_discontinuities = 0
            except MotionDiscontinuity as e:
                # A libfranka command-validity rejection, not a contact reflex. Scoring
                # it as a failed episode would be scoring the controller, not the policy
                # -- on book_placement it fired on roughly half the episodes, which would
                # dominate any success rate measured here. Mirrors train_pi05_real.py's
                # main loop; ordered first because it subclasses MotionAborted.
                if not IGNORE_MOTION_DISCONTINUITY:
                    raise
                consecutive_discontinuities += 1
                if consecutive_discontinuities >= MAX_CONSECUTIVE_DISCONTINUITIES:
                    logger.error(
                        f"{consecutive_discontinuities} consecutive command-validity aborts — "
                        f"not a transient; failing the episode."
                    )
                    consecutive_errors += 1
                    robot.recover_from_errors()
                    pending_outcome = 0.0
                    input_manager.done_queue.put(True)
                    continue
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
                consecutive_errors += 1
                logger.warning(
                    f"Motion aborted ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}) "
                    f"— auto-reporting episode failure: {e}"
                )
                robot.recover_from_errors()
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error("Too many consecutive control errors — aborting eval.")
                    break
                pending_outcome = 0.0
                input_manager.done_queue.put(True)
                continue

            ep_steps += 1
            if ep_steps % 50 == 0:
                logger.info(f"  ep {len(episodes) + 1}: step {ep_steps} | gripper {gripper_action:+.3f} {g_decision}")

            elapsed = time.time() - step_start
            if step_duration - elapsed > 0:
                time.sleep(step_duration - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
    finally:
        n = len(episodes)
        n_succ = sum(e["success"] for e in episodes)
        logger.info("=" * 60)
        if n:
            lo, hi = _wilson_ci(n_succ, n)
            mean_steps = float(np.mean([e["steps"] for e in episodes]))
            logger.info(f"EVAL RESULT: {n_succ}/{n} = {n_succ / n:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
            logger.info(f"  mean episode length: {mean_steps:.0f} steps")
            aborts = sum(1 for e in episodes if e["reason"] == "motion-aborted")
            if aborts:
                logger.info(f"  motion-aborted episodes: {aborts}/{n}")
        else:
            logger.info("EVAL RESULT: no episodes completed")
        logger.info("=" * 60)

        if args.results_csv and episodes:
            with open(args.results_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
                w.writeheader()
                w.writerows(episodes)
            logger.info(f"Per-episode results -> {args.results_csv}")

        cameras.stop()
        robot.stop()
        logger.info("Evaluation client stopped")


if __name__ == "__main__":
    main()
