"""Evaluate a fine-tuned Pi0 checkpoint on LIBERO simulation.

Loads the Pi0 model from a checkpoint, wraps it as a policy, and runs it
in the LIBERO OffScreenRenderEnv for N episodes per task, reporting
success rate.

The obs bridge:
  LIBERO env → agentview_image [H,W,3] uint8
             → robot0_eye_in_hand_image [H,W,3] uint8
             → robot0_joint_pos + robot0_gripper_qpos → state [8]
  Pi0 expects → observation/image [H,W,3] uint8
              → observation/wrist_image [H,W,3] uint8
              → observation/state [action_dim]
              → prompt: str

Usage:
    uv run scripts/eval_libero_sim.py \\
        --checkpoint_dir ./checkpoints/few_shot_sft \\
        --config_name pi05_libero \\
        --suite libero_10 \\
        --n_eval 20 \\
        --max_steps 600
"""

import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="shape requires ndarray", category=DeprecationWarning)

import math
import numpy as np
import tqdm
import tyro

# Dataset task_index offset per LIBERO suite (matches LIBERO_SUITE_TASK_INDICES in config.py)
SUITE_TASK_OFFSETS = {
    "libero_10":      0,
    "libero_goal":    10,
    "libero_object":  20,
    "libero_spatial": 30,
}

# ── LIBERO imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/libero"))
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

# ── openpi imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from openpi.training import config as train_config_lib
from openpi.policies import policy_config as _policy_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

SUITE_TASK_COUNTS = {
    "libero_10": 10,
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
}




# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, config_name: str, rlt_width: int = 0):
    """Load Pi0 model via create_trained_policy (identical to serve_policy path).

    Returns:
        policy               – Policy object with .infer(obs_dict) -> {"actions": ...}
        cfg                  – TrainConfig
    """
    import dataclasses

    cfg = train_config_lib.get_config(config_name)
    if rlt_width > 0:
        # Stage-1 RLT checkpoints are not all at the config default 2048 (libero_goal
        # t3 was trained at 512), and the loader does a strict pytree shape check, so a
        # mismatch fails at ['rlt']['dec_blocks'] before a single episode runs. The
        # RLT decoder plays no part in action sampling; this only lets the tree match.
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, rlt_width=rlt_width))
    policy = _policy_config.create_trained_policy(cfg, checkpoint_dir)
    logging.info(f"Loaded model from {checkpoint_dir}")
    return policy, cfg




LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ── Obs bridge ────────────────────────────────────────────────────────────────

def libero_obs_to_dict(obs: dict, prompt: str, resize_size: int = 224) -> dict:
    """Convert a raw LIBERO obs dict to the flat dict expected by Policy.infer().

    Matches examples/libero/main.py exactly: rotate 180°, resize, full gripper_qpos.
    Norm/unnorm is handled inside Policy.infer() via the config transforms.
    """
    from openpi_client import image_tools
    img   = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img   = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


# ── Video helpers ─────────────────────────────────────────────────────────────


def save_video(frames, path, fps=10):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()




# ── Main eval loop ────────────────────────────────────────────────────────────

def evaluate_task(
    policy,
    task,
    replan_steps: int,
    n_eval: int,
    max_steps: int,
    img_size: int,
    render_size: int,
    settle_steps: int,
    task_id: int,
    video_dir: Path | None = None,
    fps: int = 10,
) -> float:
    """Run n_eval episodes on one LIBERO task. Returns success rate.

    Matches examples/libero/main.py exactly: uses policy.infer() so norm/unnorm
    is handled identically to the serve_policy path.
    """
    bddl_path = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=render_size,
        camera_widths=render_size,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )

    import torch
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        task.problem_folder,
        task.init_states_file,
    )
    init_states = torch.load(init_states_path, weights_only=False)

    prompt = task.language
    logging.info(f"Task {task_id}: '{prompt}'")

    num_success = 0

    for ep in tqdm.trange(n_eval, desc=f"task_{task_id}", leave=False):
        import collections
        ep_seed = 7
        np.random.seed(ep_seed)
        env.reset()
        env.seed(ep_seed)
        obs = env.set_init_state(init_states[ep % len(init_states)])

        for _ in range(settle_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        action_plan = collections.deque()
        frames = []
        done = False
        success = False
        step = 0

        pbar = tqdm.trange(max_steps, desc=f"ep{ep}", leave=False)
        while step < max_steps and not done:
            if video_dir is not None:
                frames.append(obs["agentview_image"][::-1, ::-1].copy())

            if not action_plan:
                element = libero_obs_to_dict(obs, prompt, resize_size=img_size)
                action_chunk = policy.infer(element)["actions"]
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()
            obs, _, done, info = env.step(action.tolist())
            step += 1
            pbar.update(1)

            if done or info.get("success", False):
                success = True
                num_success += 1
                done = True
                break

        pbar.close()

        status = "success" if success else "fail"
        print(f"  ep{ep:02d} [{status}]", flush=True)

        if video_dir is not None and frames:
            vid_path = video_dir / f"task{task_id:02d}_ep{ep:03d}_{status}.mp4"
            try:
                save_video(frames, vid_path, fps=fps)
                print(f"Saved video ({len(frames)} frames) → {vid_path}", flush=True)
            except Exception as e:
                logging.error(f"Failed to save video: {e}")

    env.close()
    success_rate = num_success / n_eval
    logging.info(f"Task {task_id}: {num_success}/{n_eval} = {success_rate:.2%}")
    return success_rate


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    checkpoint_dir: str = tyro.MISSING,
    config_name: str = "pi05_libero",
    suite: str = "libero_10",
    task_ids: list[int] = [],
    n_eval: int = 20,
    max_steps: int = 600,
    replan_steps: int = 5,
    img_size: int = 224,
    render_size: int = 0,
    settle_steps: int = 10,
    rlt_width: int = 0,
    save_videos: bool = True,
    fps: int = 10,
):
    # ── Load model ────────────────────────────────────────────────────────────
    policy, cfg = load_model(checkpoint_dir, config_name, rlt_width)

    # ── Load benchmark ────────────────────────────────────────────────────────
    benchmark = get_benchmark(suite)()
    n_tasks = benchmark.n_tasks
    ids = task_ids if task_ids else list(range(n_tasks))
    logging.info(f"Suite: {suite}, evaluating {len(ids)}/{n_tasks} tasks, {n_eval} eps each")

    # ── Video dir ─────────────────────────────────────────────────────────────
    video_dir = None
    if save_videos:
        video_dir = Path("checkpoints") / f"videos_{suite}"
        video_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Videos → {video_dir}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = {}
    for tid in tqdm.tqdm(ids, desc="Tasks"):
        task = benchmark.get_task(tid)
        sr = evaluate_task(
            policy=policy,
            task=task,
            replan_steps=replan_steps,
            n_eval=n_eval,
            max_steps=max_steps,
            img_size=img_size,
            render_size=render_size or img_size,
            settle_steps=settle_steps,
            task_id=tid,
            video_dir=video_dir,
            fps=fps,
        )
        results[tid] = sr
        tqdm.tqdm.write(f"task_{tid:02d}  SR={sr:.1%}  '{task.language[:60]}'")

    # ── Summary ───────────────────────────────────────────────────────────────
    success_rates = list(results.values())
    print("\n" + "=" * 50)
    print(f"Suite: {suite}  |  Checkpoint: {checkpoint_dir}")
    print("=" * 50)
    for tid, sr in results.items():
        task = benchmark.get_task(tid)
        print(f"  task_{tid:02d}  {sr:.1%}  {task.language[:60]}")
    print("-" * 50)
    print(f"  Mean success: {np.mean(success_rates):.1%}")
    print("=" * 50)

    out = video_dir.parent / f"eval_{suite}.npz" if video_dir else Path(f"eval_{suite}.npz")
    np.savez(str(out), task_ids=list(results.keys()), success_rates=success_rates)
    print(f"Saved to {out}")


if __name__ == "__main__":
    tyro.cli(main)