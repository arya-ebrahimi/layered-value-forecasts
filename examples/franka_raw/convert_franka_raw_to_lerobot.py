"""
Script to convert the <hf-user>/franka_raw hdf5 data to the LeRobot dataset v2.0 format.

Raw episode schema (single-arm Franka Panda, ~10 Hz control loop, no depth actually recorded):
  action                          (T, 7) float32   -- raw command, ~[-1, 1]
  observations/qpos               (T, 7) float32   -- arm joint angles (rad) -- NOT used for
                                                        state (see below); kept available.
  observations/qvel               (T, 7) float32   -- arm joint velocities
  observations/gpos               (T, 1) float32   -- gripper width (m), ~0.014-0.08
  observations/ee_pos_t            (T, 3) float32   -- end-effector position (m)
  observations/ee_pos_q            (T, 4) float32   -- end-effector orientation quaternion,
                                                        (x, y, z, w) order (scipy/scalar-last).
                                                        Verified against the also-stored
                                                        ee_pos_rpy: scipy Rotation.from_quat(ee_pos_q)
                                                        .as_euler('xyz') reproduces ee_pos_rpy exactly;
                                                        the data-collection code's own
                                                        quaternion_array_to_rpy docstring confirms
                                                        franky's pose.quaternion is (x,y,z,w).
  observations/images/ext1        (T, 480, 640, 3) uint8
  observations/images/wrist       (T, 480, 640, 3) uint8
  tm                              (T, 1) float32   -- per-step dt (s); first entry is a startup delay, not a control dt

Not currently ported (available in the raw hdf5 if needed later): ee_pos_rpy, ee_twist_lin/ang,
tau_J, dtau_J, tau_ext_hat_filtered, O_F_ext_hat_K, J, T_ee, elbow_jnt3_pos, elbow_jnt4_flip.

`action` is the raw 7-dim command as recorded, dims 0-5 are small arm deltas (dims 3,4 are
always exactly 0 across the whole dataset -- unused axes for this task) and dim 6 is NOT a
7th joint: verified against gpos (gripper width) that it's a discrete gripper trigger
(-1=close, 0=no-op, +1=open), appearing as brief 1-2 frame pulses in all 100/100 episodes,
each followed shortly after by gpos actually decreasing (close) or increasing (open).

--gripper-encoding controls how that column is written out:
  "held" (DEFAULT): latch the trigger into a persistent state command (-1=closed, +1=open
      at EVERY frame), matching LIBERO's convention. Strongly preferred for RL fine-tuning.
      As recorded, the trigger is nonzero in only ~1.5% of frames (std 0.124), which (a)
      collapses the 1st/99th percentile window so quantile norm is unusable, (b) collapses
      the RLT ActionSpace calibration to its 0.05 floor -- crushing the close command ~160x
      so the gripper can never close once rollouts stop executing the raw reference, and
      (c) makes the BC/MSE target a sparse spike that regression pulls toward its ~0 mean,
      so the TD3 actor never learns to emit it. Latching makes the channel dense (nonzero
      100% of frames, std ~1), which removes all three. LIBERO measures exactly this:
      nonzero in 100.00% of frames, std 0.991.
  "trigger": preserve the raw pulses as recorded (previous behavior).
The real-robot client needs NO change either way: execute_gripper_action_toggle() latches on
its own `_gripper_target`, firing the hardware call only when the commanded state CHANGES, so
a 1-frame pulse and a held level both produce exactly one grasp at the same instant. Held is
additionally robust to a single weak/missed frame, which under "trigger" loses the grasp
permanently.

`observation.state` is end-effector pose + gripper, 7-dim: ee_pos_t(3) + axis-angle(ee_pos_q)(3)
+ gpos(1) -- NOT qpos. This matches the real-robot deployment convention hard-coded in
src/openpi/training/rlt/envs.py's `_make_state` (built for pi05_libero / LIBERO's own
eef-pose + axis-angle + gripper_qpos state, which the real-robot bridge and the Franka
teleop/inference client both already use), so a checkpoint trained on this convention can be
deployed through that bridge without touching any of its (tested) code. The axis-angle
conversion is bit-for-bit `workers._quat2axisangle`, applied directly to ee_pos_q with NO
reordering (an earlier version of this script incorrectly assumed ee_pos_q was (w,x,y,z) and
reordered it -- that was wrong and silently trained on garbled orientation state; e.g. a
frame whose true roll is ~pi rad was fed to the model as ~0.0008 rad. Caused visibly excessive
wrist rotation at real-robot deployment. Fixed and verified below.)

Example usage:
  uv run examples/franka_raw/convert_franka_raw_to_lerobot.py \
      --raw-dir $SCRATCH/datasets/franka_raw_hdf5 \
      --repo-id franka_raw \
      --root ${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot/franka_raw
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
# action dim 6 is a discrete gripper trigger, not a 7th joint -- see module docstring.
ACTION_NAMES = [*ARM_JOINTS[:6], "gripper_trigger"]
STATE_NAMES = ["x", "y", "z", "ax", "ay", "az", "gripper"]
CAMERAS = ["ext1", "wrist"]


def _quat_xyzw_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Bit-for-bit match of src/openpi/training/rlt/envs.py's `_quat2axisangle`,
    which expects (x, y, z, w) order -- our raw ee_pos_q is already (x, y, z, w) (verified
    against the independently-stored ee_pos_rpy via scipy, see module docstring), so NO
    reordering here. Vectorized over the leading (T,) axis.
    """
    w = quat_xyzw[:, 3]
    xyz = quat_xyzw[:, 0:3]
    q3 = np.clip(w, -1.0, 1.0)
    den = np.sqrt(np.clip(1.0 - q3 * q3, 0.0, None))
    angle = 2.0 * np.arccos(q3)
    out = np.zeros_like(xyz, dtype=np.float32)
    nonzero = ~np.isclose(den, 0.0)
    out[nonzero] = xyz[nonzero] * (angle[nonzero] / den[nonzero])[:, None]
    return out.astype(np.float32)


def _latch_gripper_trigger(trigger: np.ndarray, initial: float = 1.0) -> np.ndarray:
    """Edge-triggered gripper command -> held state command (see module docstring).

    Forward-fills the most recent nonzero trigger over the frames that follow it, so
    (0, 0, -1, 0, 0) becomes (+1, +1, -1, -1, -1). Frames before the episode's first
    pulse take `initial` (+1 = open, which is where go_home() leaves the gripper).
    """
    cmd = np.where(trigger < -0.5, -1.0, np.where(trigger > 0.5, 1.0, 0.0)).astype(np.float32)
    # Index of the most recent nonzero command at or before each frame (-1 if none yet).
    last = np.maximum.accumulate(np.where(cmd != 0, np.arange(len(cmd)), -1))
    return np.where(last >= 0, cmd[last], initial).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.05
    image_writer_processes: int = 8
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    root: Path,
    fps: int,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": [ACTION_NAMES],
        },
    }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if root.exists():
        shutil.rmtree(root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type="franka",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_episode_data(
    ep_path: Path,
    gripper_encoding: Literal["held", "trigger"] = "held",
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    with h5py.File(ep_path, "r") as ep:
        ee_pos_t = ep["/observations/ee_pos_t"][:]
        ee_pos_q = ep["/observations/ee_pos_q"][:]
        gpos = ep["/observations/gpos"][:]
        axis_angle = _quat_xyzw_to_axisangle(ee_pos_q)
        state = torch.from_numpy(np.concatenate([ee_pos_t, axis_angle, gpos], axis=-1).astype(np.float32))
        raw_action = ep["/action"][:].astype(np.float32)
        if gripper_encoding == "held":
            # Only the gripper column changes; the arm delta dims are untouched.
            raw_action[:, 6] = _latch_gripper_trigger(raw_action[:, 6])
        action = torch.from_numpy(raw_action)

        imgs_per_cam = {cam: ep[f"/observations/images/{cam}"][:] for cam in CAMERAS}

    return imgs_per_cam, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    gripper_encoding: Literal["held", "trigger"] = "held",
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action = load_raw_episode_data(ep_path, gripper_encoding=gripper_encoding)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_franka_raw(
    raw_dir: Path,
    repo_id: str,
    root: Path,
    task: str = "pick up the book and place it in the book holder",
    *,
    fps: int = 10,
    gripper_encoding: Literal["held", "trigger"] = "held",
    episodes: list[int] | None = None,
    # "video" requires a working PyAV install; this venv only has a stub `av` package
    # (no av.container), so default to "image" (per-frame PNGs) to match what actually works.
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {raw_dir}")

    dataset = create_empty_dataset(
        repo_id,
        root=root,
        fps=fps,
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
        gripper_encoding=gripper_encoding,
    )


if __name__ == "__main__":
    tyro.cli(port_franka_raw)
