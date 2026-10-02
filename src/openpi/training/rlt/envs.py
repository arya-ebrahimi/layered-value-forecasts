"""LIBERO / real-robot environment pools and pi0 loading for RLT stage 2.

The LIBERO env runs each env in a spawned subprocess (EGL/OpenGL contexts are not
thread-safe), converts raw obs into the pi0 Observation format, and samples the
canonical per-trial init states. RealRobotEnvWorker exposes the same surface over
the websocket bridge (rlt/robot_bridge.py) so the training loop is backend-agnostic.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import multiprocessing
from multiprocessing import connection as mp_connection
import os
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

import openpi.models.model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.training.config as train_config_lib
from openpi.training.rlt.robot_bridge import _BridgeChannel

logger = logging.getLogger("rlt.envs")


def silence_deprecation_warnings() -> None:
    """Quiet the noisy third-party startup spam so the training log is readable.

    Three distinct emitters, each fired per process (incl. all 16 env subprocesses):
      1. jax/flax DeprecationWarnings (flax scope "shape requires ndarray...", jax
         linear_util "wrap_init missing DebugInfo") — via warnings.warn. JAX rewrites
         the warnings filter list on import so filterwarnings()/-W don't stick;
         overriding warnings.showwarning (the DISPLAY hook) survives that.
      2. gym "unmaintained since 2022" — also a warning; dropped by the same hook.
      3. robosuite "No private macro file..." — emitted via the `robosuite_logs`
         logging logger at INFO; silenced by raising its level.
    Idempotent. Call in the main process and in each spawn subprocess.
    """
    import contextlib
    import io
    import logging as _logging
    import warnings

    # (3) noisy third-party logging loggers (robosuite, PyOpenGL acceleratesupport that
    # logs once per process at INFO, etc.) — raise their level so they stay quiet in
    # BOTH the main process and every spawn subprocess (each imports OpenGL).
    for _noisy in ("robosuite_logs", "OpenGL", "OpenGL.acceleratesupport", "absl", "jax"):
        _logging.getLogger(_noisy).setLevel(_logging.ERROR)

    # (2b) gym prints its "unmaintained since 2022" notice with a raw
    # print(..., file=sys.stderr) at IMPORT time — not a warning, so the showwarning
    # hook can't catch it. Pre-import gym here with stderr swallowed; later imports
    # are cached no-ops, so the notice never reaches the real log.
    if not getattr(warnings, "_openpi_silenced", False):
        with contextlib.suppress(Exception), contextlib.redirect_stderr(io.StringIO()):
            import gym  # noqa: F401

    if getattr(warnings, "_openpi_silenced", False):
        return
    orig = warnings.showwarning
    _drop_msgs = ("Gym has been unmaintained", "unmaintained since 2022")

    def showwarning(message, category, filename, lineno, file=None, line=None):
        # Drop ALL DeprecationWarnings (library-internal: flax scope, jax linear_util,
        # robosuite's deprecated logger.warn(), etc.) plus the specific gym notice.
        if category is DeprecationWarning or any(s in str(message) for s in _drop_msgs):
            return
        orig(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    warnings._openpi_silenced = True


# LIBERO task-id offsets into the flat lerobot dataset (unused for env, kept for parity).
SUITE_TASK_OFFSETS = {"libero_10": 0, "libero_goal": 10, "libero_object": 20, "libero_spatial": 30}
SUITE_N_TASKS = {"libero_10": 10, "libero_goal": 10, "libero_object": 10, "libero_spatial": 10}

# Canonical LIBERO reset protocol (matches eval_libero_sim.py and RLinf's libero_env):
# after env.reset() + env.set_init_state(<trial init state>), step a few no-op actions
# (zero translation/rotation, gripper open = -1) to let the scene settle before the
# policy acts. RLinf uses 15 settle steps; these do NOT count toward the episode length.
LIBERO_DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)
RESET_SETTLE_STEPS = 15


def _reset_env_to(env, init_state):
    """Reset a raw LIBERO env to a specific trial init state, then settle.

    env.reset() -> env.set_init_state(init_state) -> RESET_SETTLE_STEPS no-op steps.
    Returns the settled obs. If init_state is None, falls back to a plain reset (the
    old behaviour) so callers without an init state still work.
    """
    obs = env.reset()
    if init_state is None:
        return obs
    obs = env.set_init_state(init_state)
    for _ in range(RESET_SETTLE_STEPS):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def _ensure_nvidia_egl_vendor() -> None:
    """Force GLVND to load only the NVIDIA EGL driver for headless device rendering.

    When both 10_nvidia.json and 50_mesa.json are present in egl_vendor.d, GLVND
    loads both but Mesa wins the default dispatcher (higher sort order). Mesa's EGL
    configs only support window surfaces (no pbuffer), so mujoco.MjrContext fails
    with 0x8cdd (GL_FRAMEBUFFER_UNSUPPORTED) when it tries to set up an offscreen
    framebuffer. We force GLVND to load ONLY the NVIDIA vendor by pinning
    __EGL_VENDOR_LIBRARY_FILENAMES to the nvidia JSON.
    """
    if os.environ.get("MUJOCO_GL") != "egl" or "__EGL_VENDOR_LIBRARY_FILENAMES" in os.environ:
        return
    # Prefer the system NVIDIA vendor JSON; synthesize one if missing.
    for d in ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"):
        candidates = list(Path(d).glob("*nvidia*.json")) if Path(d).is_dir() else []
        if candidates:
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(candidates[0])
            return
    if not list(Path("/usr/lib/x86_64-linux-gnu").glob("libEGL_nvidia.so*")):
        return  # no NVIDIA EGL driver installed; leave Mesa to handle it (or fail clearly)
    cfg = Path.home() / ".config" / "egl_vendor" / "10_nvidia.json"
    if not cfg.exists():
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(
            '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n'
        )
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(cfg)


def _egl_render_devices(pinned: str | None) -> list[str]:
    """Physical EGL device ids to spread env rendering across.

    If CUDA_VISIBLE_DEVICES names one GPU we render only there (co-located with
    JAX, set by setup_egl). If it names several (or is unset and JAX took all
    GPUs), we spread env framebuffers round-robin over those physical GPUs so 64
    envs become e.g. 32+32 instead of 64 on one card (which exhausts the
    framebuffer pool -> 0x8cdd). Falls back to the single pinned device.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys = [p.strip() for p in cvd.split(",") if p.strip().isdigit()]
    if len(phys) > 1:
        return phys
    if not phys:  # unset -> JAX takes every GPU; spread across all of them
        try:
            import jax

            n = jax.local_device_count()
            if n > 1:
                return [str(i) for i in range(n)]
        except Exception:
            pass
    return [pinned] if pinned is not None else ["0"]


def setup_egl() -> None:
    """Set EGL headless rendering and pick a working EGL device.

    On multi-GPU hosts the first EGL device(s) can fail eglInitialize (driver/libEGL
    mismatch) while another works. robosuite selects the device via
    MUJOCO_EGL_DEVICE_ID; we probe and pick the first index that initializes. Must
    run before any mujoco/libero import (main process AND each subprocess worker).
    Honors a user-set MUJOCO_EGL_DEVICE_ID and skips probing for non-EGL backends.

    EGL device ids are PHYSICAL GPU indices (NVIDIA order), independent of CUDA
    masking. If CUDA_VISIBLE_DEVICES names a single physical GPU, probe THAT index
    first so rendering co-locates on the same GPU JAX uses — otherwise the probe
    would start at physical 0 and render on a different GPU than compute (and with
    many envs that GPU's framebuffer pool can be exhausted -> 0x8cdd
    FRAMEBUFFER_UNSUPPORTED).
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _ensure_nvidia_egl_vendor()
    if os.environ.get("MUJOCO_GL") != "egl" or "MUJOCO_EGL_DEVICE_ID" in os.environ:
        return
    # Probe order: the CUDA-visible physical GPU first (co-locate render with compute),
    # then the rest as fallback.
    order = list(range(8))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys = [p for p in cvd.split(",") if p.strip().isdigit()]
    if len(phys) == 1:
        pref = int(phys[0])
        order = [pref] + [i for i in order if i != pref]
    try:
        from robosuite.renderers.context import egl_context as _egl

        for i in order:
            try:
                _egl.EGL_DISPLAY = None
                disp = _egl.create_initialized_egl_device_display(device_id=i)
                if disp != _egl.EGL.EGL_NO_DISPLAY:
                    _egl.EGL_DISPLAY = None  # reset so the real context re-initializes cleanly
                    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(i)
                    logger.info(f"EGL device {i} selected (MUJOCO_EGL_DEVICE_ID={i})")
                    return
            except Exception:
                continue
        logger.warning("no working EGL device found by probe; leaving MUJOCO_EGL_DEVICE_ID unset")
    except Exception as e:
        logger.warning(f"EGL device probe skipped: {e}")


# ════════════════════════════════════════════════════════════════════════════
# Obs / action conversion
# ════════════════════════════════════════════════════════════════════════════


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q3 = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - q3 * q3)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(q3) / den).astype(np.float32)


def _make_state(o: dict, action_dim: int, state_norm) -> np.ndarray:
    state = np.concatenate(
        [
            o["robot0_eef_pos"].astype(np.float32),
            _quat2axisangle(o["robot0_eef_quat"]),
            o["robot0_gripper_qpos"].astype(np.float32),
        ]
    )
    if len(state) < action_dim:
        state = np.concatenate([state, np.zeros(action_dim - len(state), dtype=np.float32)])
    if state_norm is not None:
        kind, p1, p2 = state_norm
        n = len(p1)
        if kind == "quantile":
            state[:n] = (state[:n] - p1) / (p2 - p1 + 1e-6) * 2.0 - 1.0
        else:
            state[:n] = (state[:n] - p1) / (p2 + 1e-6)
    return state


def obs_multi_to_pi0(obs_list, tok_prompts, tok_masks, action_dim, state_norm) -> _model.Observation:
    """Batch a list of raw LIBERO obs dicts (each with its own prompt) into pi0 Observation."""
    from openpi_client import image_tools

    images, wrists, states = [], [], []
    for o in obs_list:
        img = np.ascontiguousarray(o["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(o["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
        images.append(img)
        wrists.append(wrist)
        states.append(_make_state(o, action_dim, state_norm))
    B = len(obs_list)
    imgs_np = np.stack(images).astype(np.float32) / 255.0 * 2.0 - 1.0
    wrists_np = np.stack(wrists).astype(np.float32) / 255.0 * 2.0 - 1.0
    zeros_np = np.full_like(imgs_np, -1.0)
    return _model.Observation(
        images={
            "base_0_rgb": jnp.array(imgs_np),
            "left_wrist_0_rgb": jnp.array(wrists_np),
            "right_wrist_0_rgb": jnp.array(zeros_np),
        },
        image_masks={
            "base_0_rgb": jnp.ones(B, dtype=bool),
            "left_wrist_0_rgb": jnp.ones(B, dtype=bool),
            "right_wrist_0_rgb": jnp.zeros(B, dtype=bool),
        },
        state=jnp.array(np.stack(states), dtype=jnp.float32),
        tokenized_prompt=jnp.array(np.stack(tok_prompts), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.array(np.stack(tok_masks)),
    )


def unnormalize_action(action_norm, action_normalized: np.ndarray) -> np.ndarray:
    if action_norm is None:
        return action_normalized
    kind, p1, p2 = action_norm
    n = len(p1)
    out = action_normalized.copy()
    if kind == "quantile":
        out[:, :n] = (out[:, :n] + 1.0) / 2.0 * (p2 - p1 + 1e-6) + p1
    else:
        out[:, :n] = out[:, :n] * (p2 + 1e-6) + p1
    return out


@dataclasses.dataclass
class _LiberoEnvFn:
    bddl_path: str
    seed: int
    camera_size: int = 128
    egl_device_id: str | None = None  # parent-selected EGL device, passed to child

    def __call__(self):
        # spawn-mode subprocess: reuse the EGL device the PARENT already found
        # working (avoids every child re-probing devices 0/1, which can corrupt or
        # exhaust the GL driver and crash the child). Falls back to a probe only if
        # the parent didn't pin one.
        if self.egl_device_id is not None:
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            _ensure_nvidia_egl_vendor()
            # robosuite asserts MUJOCO_EGL_DEVICE_ID is a substring of CUDA_VISIBLE_DEVICES
            # when the latter is set. Align them on this child's render GPU so the assert
            # passes and MuJoCo's framebuffer device resolves to the same physical GPU as
            # the EGL display (JAX in this child is pinned to CPU, so this only governs
            # rendering). This keeps the device tie-in consistent across both GPUs.
            os.environ["CUDA_VISIBLE_DEVICES"] = self.egl_device_id
            os.environ["MUJOCO_EGL_DEVICE_ID"] = self.egl_device_id
        else:
            setup_egl()
        from libero.libero.envs import OffScreenRenderEnv

        # Pass the physical GPU id explicitly so robosuite's MjRenderContextOffscreen
        # opens its framebuffer on the correct device instead of defaulting to -1
        # (which pins all 64 envs to GPU 0 regardless of MUJOCO_EGL_DEVICE_ID and
        # exhausts its framebuffer pool → 0x8cdd GL_FRAMEBUFFER_UNSUPPORTED).
        render_gpu_device_id = int(self.egl_device_id) if self.egl_device_id is not None else -1
        env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_path,
            camera_heights=self.camera_size,
            camera_widths=self.camera_size,
            horizon=100_000,
            render_gpu_device_id=render_gpu_device_id,
        )
        env.seed(self.seed)
        return env


def libero_subgoal_names(suite: str, task_id: int) -> tuple[str, ...]:
    """The BDDL subgoal channel names for one (suite, task), with no env and no GL.

    A pure parse of the task's .bddl, so the caller can fix the GVF channel set — and
    therefore the critic's head count — BEFORE any env is built. That ordering is not
    cosmetic: TD3 is initialized before the env pool in train_rlt_libero.main, so a
    name list that only existed after the pool came up could not size the heads.

    `build_subgoal_evaluator` returns this same list in this same order; predicates it
    cannot evaluate in the live scene are reported as permanently 0 rather than
    dropped from the vector, so the columns always line up with these names.
    """
    import importlib.util
    import sys

    libero_root = Path(__file__).resolve().parents[4] / "third_party/libero"
    sys.path.insert(0, str(libero_root))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    from openpi.training.rlt import labeling as _lab

    # bddl_utils is loaded BY FILE PATH, not as `libero.libero.envs.bddl_utils`. That
    # package's __init__ imports robosuite, which resolves and initializes a GL backend
    # from MUJOCO_GL at import time -- an absurd side effect for a function that only
    # parses a text file, and one that runs here at CONFIG-RESOLUTION time, before the
    # env pool (and its EGL device pinning) exists. bddl_utils itself needs only
    # bddl.parsing and numpy. `libero.libero.benchmark` is GL-free and imports normally.
    spec = importlib.util.spec_from_file_location(
        "_libero_bddl_utils", libero_root / "libero/libero/envs/bddl_utils.py"
    )
    bddl_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bddl_utils)

    task = get_benchmark(suite)().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return tuple(pr.name for pr in _lab.derive_subgoal_predicates(bddl_utils.robosuite_parse_problem(bddl)))


def build_subgoal_evaluator(env):
    """(names, evaluate) for the LIBERO env's BDDL subgoal predicates, or ((), None).

    Runs INSIDE the env subprocess -- it is the only side that has `sim.data`, the
    contact arrays and `object_states_dict`. The parent gets the names once in the
    startup handshake and the per-step booleans as `obs["subgoal"]`.

    `names` is exactly `derive_subgoal_predicates`' output, which is a pure function of
    the parsed problem and is therefore also what `libero_subgoal_names` computes
    offline. Every predicate is PROBED ONCE here, and one that raises or has no
    affordance is kept in the vector as a PERMANENT ZERO rather than dropped: the
    column layout has to match a name list the caller already committed to when it
    sized the GVF heads. That is what lets `derive_subgoal_predicates` propose an
    `open` channel for any destination while the ~half of tasks whose destination is a
    plain surface (a plate, a table region) simply never fire it. `unavailable` names
    those, so the caller can warn instead of silently training a constant-0 head.

    A predicate that throws mid-episode is caught per call; it must never take a
    rollout down.
    """
    from openpi.training.rlt import labeling as _lab

    base = getattr(env, "env", env)  # OffScreenRenderEnv -> the problem env
    parsed = getattr(base, "parsed_problem", None)
    if parsed is None:
        return (), None
    states = getattr(base, "object_states_dict", {})

    def _grasp(obj):
        return lambda: bool(base._check_grasp(base.robots[0].gripper, base.get_object(obj)))

    def _contact(obj):
        return lambda: bool(base.check_contact(base.robots[0].gripper, base.get_object(obj)))

    def _open(obj):
        st = states.get(obj)
        if st is None:
            return None
        # Drawer/microwave first, then appliance power: one channel, two affordances
        # (see derive_subgoal_predicates). `is_open` exists on SiteObjectState for any
        # articulated parent, so hasattr alone does not prove it works -- the probe
        # below is what actually decides.
        if hasattr(st, "is_open"):
            return lambda: bool(st.is_open())
        if hasattr(st, "turn_on"):
            return lambda: bool(st.turn_on())
        return None

    def _goal():
        return bool(base._check_success())

    def _collide(objs):
        # Resolved once at build: get_object raises for a name this scene does not
        # have, and the probe below turns that into a permanently-0 channel.
        got = [base.get_object(o) for o in objs]
        return lambda: any(base.check_contact(base.robots[0].gripper, g) for g in got)

    builders = {
        "grasp": lambda pr: _grasp(pr.args[0]),
        "contact": lambda pr: _contact(pr.args[0]),
        "open": lambda pr: _open(pr.args[0]),
        "goal": lambda _pr: _goal,
        "collide": lambda pr: _collide(pr.args),
    }

    names: list[str] = []
    fns: list = []
    unavailable: list[str] = []
    for pr in _lab.derive_subgoal_predicates(parsed):
        try:
            fn = builders[pr.kind](pr)
            if fn is None:
                raise ValueError("no affordance")
            fn()  # probe
        except Exception:  # a missing affordance/object is a zero column, not a crash
            fn, _ = None, unavailable.append(pr.name)
        names.append(pr.name)
        fns.append(fn)

    if not names:
        return (), None

    def evaluate() -> np.ndarray:
        out = np.zeros(len(fns), np.float32)
        for i, fn in enumerate(fns):
            if fn is None:
                continue
            try:
                out[i] = 1.0 if fn() else 0.0
            except Exception:  # never let a predicate kill a rollout
                out[i] = 0.0
        return out

    evaluate.unavailable = tuple(unavailable)
    return tuple(names), evaluate


def _subgoal_payload(names, evaluate) -> dict:
    """Handshake payload: the column names plus the ones this scene cannot evaluate."""
    return {
        "subgoal_names": names,
        "subgoal_unavailable": tuple(getattr(evaluate, "unavailable", ())),
    }


def _attach_subgoal(obs, evaluate):
    """Put the predicate vector into the obs dict the parent receives.

    Riding in `obs` rather than in a new pipe message keeps the child protocol and
    every call site unchanged: `obs` is what `_trim_obs` filters and what ends up in
    `rec.obs_seq`, which is where the labeler reads it.

    Note this does NOT reach `--relabel_only`: the rollout npz stores z_rl / proprio /
    ref / actions, not obs, and relabel rebuilds records with an empty obs_seq. The
    offline round trip goes through the SIDECAR instead -- `_finalize_labels` persists
    the derived events per episode exactly as it does for human labels, and relabel
    reads those back. Which is the right split anyway: a predicate label is already
    ground truth, so the thing worth correcting offline is never the predicate, it is
    a channel definition, and that means re-deriving from a new spec rather than
    re-thresholding an old trace.
    """
    if evaluate is not None and isinstance(obs, dict):
        obs["subgoal"] = evaluate()
    return obs


def _libero_env_worker(parent: mp_connection.Connection, child: mp_connection.Connection, env_fn) -> None:
    parent.close()
    # spawn subprocesses are fresh interpreters — silence the noisy JAX/Flax
    # deprecation warnings here too (showwarning override survives JAX's filter reset).
    silence_deprecation_warnings()
    # Build the env inside a handshake: report success or the full traceback to the
    # parent, so a child that dies during EGL/env init surfaces a readable error
    # instead of a bare "Connection reset by peer" pipe break in the parent.
    try:
        env = env_fn()
        subgoal_names, subgoal_eval = build_subgoal_evaluator(env)
        # The names ride the handshake, not every obs: they are fixed for the life of
        # this env and the parent needs them once, to map channel -> column.
        child.send(("ready", _subgoal_payload(subgoal_names, subgoal_eval)))
    except Exception:
        import traceback

        # Best-effort: if the parent has already gone (broken pipe), don't add a
        # confusing second traceback — just exit. The real error is in env_fn().
        try:
            child.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            traceback.print_exc()
        child.close()
        return
    try:
        while True:
            try:
                cmd, data = child.recv()
            except EOFError:
                break
            if cmd == "step":
                obs, rew, done, info = env.step(data)
                child.send((_attach_subgoal(obs, subgoal_eval), done, info))
            elif cmd == "reset":
                child.send(_attach_subgoal(env.reset(), subgoal_eval))
            elif cmd == "reset_to":
                # reset + set_init_state + settle, done child-side (one round trip).
                child.send(_attach_subgoal(_reset_env_to(env, data), subgoal_eval))
            elif cmd == "reconfigure":
                # swap the env to a new task (bddl). `data` is a fresh env_fn
                # (_LiberoEnvFn) built by the parent for the target task. Mirrors
                # RLinf's ReconfigureSubprocEnv: close the old env, build the new one.
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    env = data()
                    # Rebuilt env => a different task => re-derive. The parent checks
                    # the names still match the configured channels, since a task swap
                    # that changed the predicate set would change the head count.
                    subgoal_names, subgoal_eval = build_subgoal_evaluator(env)
                    child.send(("ready", _subgoal_payload(subgoal_names, subgoal_eval)))
                except Exception:
                    import traceback

                    try:
                        child.send(("error", traceback.format_exc()))
                    except (BrokenPipeError, OSError):
                        traceback.print_exc()
                        break
            elif cmd == "seed":
                env.seed(data)
                child.send(None)
            elif cmd == "close":
                child.send(None)
                break
            else:
                raise NotImplementedError(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        child.close()


class _SubprocEnv:
    """Proxy for a LIBERO env in a spawned subprocess (own GL context)."""

    def __init__(self, env_fn):
        ctx = multiprocessing.get_context("spawn")
        self._parent, child = ctx.Pipe()
        # spawn re-imports the parent script as __mp_main__, re-running its top-level
        # `import jax` chain. The env subprocess only needs MuJoCo, not JAX — and if
        # its JAX tries to grab CUDA it crashes ("no supported devices for CUDA",
        # the GPU is held by the parent). Force the child's JAX to CPU via inherited
        # env. We set it just for the fork, then restore so the parent keeps the GPU.
        prev = os.environ.get("JAX_PLATFORMS")
        os.environ["JAX_PLATFORMS"] = "cpu"
        try:
            self._proc = ctx.Process(target=_libero_env_worker, args=(self._parent, child, env_fn), daemon=True)
            self._proc.start()
        finally:
            if prev is None:
                os.environ.pop("JAX_PLATFORMS", None)
            else:
                os.environ["JAX_PLATFORMS"] = prev
        child.close()
        # wait for the child's startup handshake; raise its traceback on failure.
        status, payload = self._recv()
        if status != "ready":
            raise RuntimeError(f"LIBERO env subprocess failed to start:\n{payload}")
        self.subgoal_names: tuple[str, ...] = tuple((payload or {}).get("subgoal_names", ()))
        self.subgoal_unavailable: tuple[str, ...] = tuple((payload or {}).get("subgoal_unavailable", ()))

    def _recv(self, timeout: float = 300.0):
        """Blocking recv with a liveness watchdog.

        A bare self._parent.recv() blocks FOREVER if the child has died (e.g. EGL
        framebuffer exhaustion 0x8cdd during a reconfigure rebuild, OOM-kill, segfault):
        the child's broken-pipe death produces no data, so the parent hangs silently and
        the whole training job stalls. Instead, poll the pipe and the child's liveness,
        and raise a clear RuntimeError naming the dead slot so the failure surfaces.
        """
        waited = 0.0
        poll_s = 1.0
        while not self._parent.poll(poll_s):
            waited += poll_s
            if not self._proc.is_alive():
                raise RuntimeError(
                    f"LIBERO env subprocess (pid={self._proc.pid}) died (exitcode="
                    f"{self._proc.exitcode}) while the parent awaited a response — likely "
                    f"an EGL framebuffer/OOM failure during env rebuild. See child stderr above."
                )
            if waited >= timeout:
                raise RuntimeError(
                    f"LIBERO env subprocess (pid={self._proc.pid}) unresponsive after "
                    f"{timeout:.0f}s but still alive (hung in MuJoCo/EGL?)."
                )
        return self._parent.recv()

    def seed(self, s):
        self._parent.send(("seed", s))
        self._recv()

    def reset(self):
        self._parent.send(("reset", None))
        return self._recv()

    def reset_to(self, init_state):
        self._parent.send(("reset_to", init_state))
        return self._recv()

    def reconfigure(self, env_fn):
        """Swap to a new task by rebuilding the env in the child (RLinf reconfigure)."""
        self._parent.send(("reconfigure", env_fn))
        status, payload = self._recv()
        if status != "ready":
            raise RuntimeError(f"LIBERO env reconfigure failed:\n{payload}")
        self.subgoal_names = tuple((payload or {}).get("subgoal_names", ()))
        self.subgoal_unavailable = tuple((payload or {}).get("subgoal_unavailable", ()))

    def send_step(self, action):
        self._parent.send(("step", action))

    def recv_step(self):
        obs, done, info = self._recv()
        return obs, done, info

    def close(self):
        try:
            self._parent.send(("close", None))
            self._parent.recv()
        except Exception:
            pass
        self._proc.join(timeout=5)
        self._proc.terminate()


# ════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════


def _normalize_key(k: tuple) -> tuple:
    return tuple(int(x) if isinstance(x, str) and x.isdigit() else x for x in k)


def _filter_params(params: dict, model_keys: set) -> dict:
    from flax.traverse_util import flatten_dict
    from flax.traverse_util import unflatten_dict

    flat = flatten_dict(params)
    norm_to_orig = {_normalize_key(k): k for k in flat}
    norm_model_keys = {_normalize_key(k): k for k in model_keys}
    filtered, matched = {}, set()
    for nk, mk in norm_model_keys.items():
        if nk in norm_to_orig:
            filtered[mk] = flat[norm_to_orig[nk]]
            matched.add(mk)
    missing = model_keys - matched
    if missing:
        logger.warning(f"{len(missing)} model keys not in checkpoint (random init)")
    return unflatten_dict(filtered)


def load_pi0_model(checkpoint_dir: str, config_name: str, dtype=jnp.bfloat16, *, rlt_width: int | None = None):
    """Load a pi0 checkpoint -> (graphdef, state, model_config, train_cfg).

    dtype: restore precision. Defaults to bf16 (the base/frozen VLM dtype). Pass float32
    when loading a full RL checkpoint directly, so the trained action expert (saved in f32)
    keeps its precision — a bf16 restore rounds away a large fraction of the small RL
    fine-tuning delta. The caller casts the frozen VLM back to bf16 after the ae/frozen split.

    rlt_width: per-LOAD override of Pi0Config.rlt_width, for reading a tokenizer trained
    at a width the named config no longer declares. The default (2048, the paper's VLA
    embedding width) was raised after the existing checkpoints were trained at 512, and
    a 512-wide checkpoint restored into a 2048-wide module dies at the first attention
    call ("Incompatible input dimension, got 512 but module expects 2048"). This is a
    LOAD-TIME argument and NOT a config edit on purpose: editing the shared
    TrainConfig would silently re-shape every other run of that config name, including
    stage-1 tokenizer jobs already queued at the current default. None = whatever the
    config says.
    """
    cfg = train_config_lib.get_config(config_name)
    model_config = cfg.model
    if rlt_width is not None and rlt_width != getattr(model_config, "rlt_width", None):
        logger.info(f"rlt_width override: {getattr(model_config, 'rlt_width', None)} -> {rlt_width}")
        model_config = dataclasses.replace(model_config, rlt_width=rlt_width)
        cfg = dataclasses.replace(cfg, model=model_config)
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.is_dir():
        steps = sorted(int(p.name) for p in ckpt_path.iterdir() if p.name.isdigit())
        params_path = (ckpt_path / str(steps[-1]) / "params") if steps else (ckpt_path / "params")
    else:
        params_path = ckpt_path
    logger.info(f"Loading params from {params_path} (dtype={jnp.dtype(dtype).name})")
    params = _model.restore_params(params_path, dtype=dtype)
    model = model_config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    state_dict = state.flat_state()
    filtered = _filter_params(params, set(state_dict.keys()))
    state.replace_by_pure_dict(filtered)
    model = nnx.merge(graphdef, state)
    model.eval()
    graphdef, state = nnx.split(model)
    logger.info("Model loaded successfully")
    return graphdef, state, model_config, cfg


class _EpState:
    """One ongoing episode inside the vectorised env pool."""

    __slots__ = ("chunks", "done", "env", "init_state", "obs", "steps", "success", "task_id", "tok_mask", "tok_prompt")

    def __init__(self, env, tok_prompt, tok_mask, task_id, init_state=None):
        self.env = env
        self.tok_prompt = tok_prompt
        self.tok_mask = tok_mask
        self.task_id = task_id
        # the trial init state this episode starts from. None -> plain reset.
        self.init_state = init_state
        self.reset()

    def reset(self):
        self.obs = self.env.reset_to(self.init_state)
        self.done = False
        self.success = False
        self.steps = 0
        self.chunks: list[dict] = []


# ════════════════════════════════════════════════════════════════════════════
# EnvWorker — builds and resets the LIBERO env pool
# ════════════════════════════════════════════════════════════════════════════


class EnvWorker:
    """Builds the LIBERO env pool (env pool layout follows RLinf's env_worker)."""

    def __init__(self, cfg, tok: PaligemmaTokenizer, action_dim, state_norm, action_norm):
        self.cfg = cfg
        self.tok = tok
        self.action_dim = action_dim
        self.state_norm = state_norm
        self.action_norm = action_norm
        self.num_action_chunks = cfg.action_chunk
        # group_size structures the env pool: consecutive group_size slots share one
        # (task, trial) start. RLT uses group_size=1 (cfg has no such field).
        self.group_size = max(1, getattr(cfg, "group_size", 1))
        self._build_envs()

    def _build_envs(self):
        import sys

        # headless EGL + working-device selection, before any libero/mujoco import.
        setup_egl()
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "third_party/libero"))
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark

        cfg = self.cfg
        n_suite = SUITE_N_TASKS[cfg.suite]
        self.task_ids = cfg.task_ids if cfg.task_ids else list(range(n_suite))
        benchmark = get_benchmark(cfg.suite)()

        def make_bddl(ltid):
            task = benchmark.get_task(ltid)
            return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file), task.language

        def load_init_states(ltid):
            """The canonical per-trial init states for a task (same source as
            eval_libero_sim.py / RLinf get_task_init_states)."""
            import torch

            task = benchmark.get_task(ltid)
            path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
            return np.asarray(torch.load(path, weights_only=False))

        # ── Build the env pool (RLinf-faithful grouping) ───────────────────────
        # num_group groups of group_size envs. A *reset_state_id* is a GLOBAL index
        # over all (task, trial) pairs across all tasks (RLinf total_num_group_envs /
        # cumsum_trial_id_bins). Each rollout make_pool() draws num_group random
        # reset_state_ids, repeats each across its group_size members (so a group shares
        # ONE (task, trial) = identical start, differing only by action noise -> valid
        # group start) and RECONFIGURES each env whose task changed to that task's
        # bddl. Tasks are therefore (re)assigned to groups randomly every rollout and all
        # tasks get covered over rollouts — exactly like RLinf (no fixed task-per-slot),
        # so any total_num_envs/group_size (e.g. 64/8) works.
        group_size = max(1, self.group_size)
        n_groups = max(1, cfg.total_num_envs // group_size)
        n_tasks = len(self.task_ids)
        self.n_groups = n_groups
        self.batch_size = n_groups * group_size
        self._egl_device_id = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        self._egl_render_devices = _egl_render_devices(self._egl_device_id)
        if len(self._egl_render_devices) > 1:
            logger.info(f"spreading env render across EGL devices {self._egl_render_devices}")

        # per-task assets: prompt tokenization, bddl path, trial init states.
        self._task_tok: dict[int, tuple] = {}
        self._task_bddl: dict[int, str] = {}
        self._task_init_states: dict[int, np.ndarray] = {}
        for ltid in self.task_ids:
            bddl_path, lang = make_bddl(ltid)
            tok_prompt, tok_mask = self.tok.tokenize(lang)
            self._task_tok[ltid] = (np.array(tok_prompt), np.array(tok_mask), lang)
            self._task_bddl[ltid] = bddl_path
            self._task_init_states[ltid] = load_init_states(ltid)

        # reset-state space over self.task_ids: bins[i] = #trials of task_ids[i]; cumsum
        # maps a global reset_state_id -> (task index, trial). Mirrors RLinf
        # _compute_total_num_group_envs + _get_task_and_trial_ids_from_reset_state_ids
        # (restricted to task_ids == RLinf's task_id_filter case).
        self._trial_bins = np.array([len(self._task_init_states[t]) for t in self.task_ids])
        self._cumsum_bins = np.cumsum(self._trial_bins)
        self._total_reset_states = int(self._cumsum_bins[-1])

        # persistent RNG (RLinf seeds one np.random.default_rng; reset states are
        # resampled every rollout) + ordered cursor for deterministic eval coverage.
        self._np_rng = np.random.default_rng(cfg.seed)
        self._eval_cursor = 0

        logger.info(
            f"LIBERO suite={cfg.suite} tasks={self.task_ids} | {n_groups} groups x "
            f"group_size {group_size} = batch {self.batch_size} | "
            f"{self._total_reset_states} reset states across {n_tasks} tasks"
        )

        # Build all slots as EGL-isolated subprocess envs (uniform -> reconfiguration is
        # safe; no in-process GL env to rebuild next to JAX's CUDA context). Each slot
        # starts on task_ids[g % n_tasks]; make_pool reconfigures it as needed.
        # Build subprocess envs. Each spawn re-imports the script (jax/openpi) + inits
        # MuJoCo/EGL, so this is a few seconds PER env and runs sequentially — show a
        # progress bar so a slow build (e.g. 50 eval envs) doesn't look hung.
        self._slot_envs: list = []
        self._slot_task: list[int] = []
        for slot_idx in tqdm.tqdm(
            range(self.batch_size), desc="build envs", unit="env", dynamic_ncols=True, leave=False
        ):
            ltid = self.task_ids[(slot_idx // group_size) % n_tasks]
            seed = cfg.seed + ltid * 1000 + slot_idx
            dev = self._egl_render_devices[slot_idx % len(self._egl_render_devices)]
            # camera_size is read off cfg when present and otherwise stays at
            # _LiberoEnvFn's 128 default. NOTE the LIBERO LeRobot dataset the SFT
            # checkpoints were fine-tuned on stores 256x256 frames; both are
            # resize_with_pad'ed to 224, so anything rendered here at 128 reaches the
            # VLA upscaled from a quarter of the pixels.
            env = _SubprocEnv(
                _LiberoEnvFn(
                    bddl_path=self._task_bddl[ltid],
                    seed=seed,
                    camera_size=int(getattr(self.cfg, "camera_size", 128)),
                    egl_device_id=dev,
                )
            )
            self._slot_envs.append(env)
            self._slot_task.append(ltid)
        # BDDL subgoal predicates, discovered per env (see build_subgoal_evaluator).
        # Every slot of a single-task run must agree; if a multi-task pool disagrees,
        # the intersection is all that can be shipped as a fixed-width column vector,
        # and the caller is told rather than silently getting misaligned columns.
        names = {tuple(e.subgoal_names) for e in self._slot_envs}
        self.subgoal_names: tuple[str, ...] = tuple(sorted(names)[0]) if len(names) == 1 else ()
        if len(names) > 1:
            logger.warning(
                f"env slots disagree on their BDDL subgoal predicates ({sorted(names)}) — "
                f"predicate labeling is disabled for this pool. It needs one task per run "
                f"(or a task set whose predicate sets match), since the GVF head count is "
                f"fixed at config time."
            )
        elif self.subgoal_names:
            dead = sorted({n for e in self._slot_envs for n in e.subgoal_unavailable})
            logger.info(f"BDDL subgoal predicates: {list(self.subgoal_names)}")
            if dead:
                logger.warning(
                    f"BDDL predicates {dead} have no affordance in this scene and will be "
                    f"permanently 0 (e.g. an `open` channel on a destination that is a plain "
                    f"surface). Their GVF heads will train on an all-zero cumulant — drop them "
                    f"from --gvf_channels to save the compute."
                )
        logger.info(f"Built {self.batch_size} env slots")

    def _reset_state_to_task_trial(self, rsid: int) -> tuple[int, int]:
        """Map a global reset_state_id -> (task_id, trial_id) via the cumsum bins
        (RLinf _get_task_and_trial_ids_from_reset_state_ids)."""
        ti = int(np.searchsorted(self._cumsum_bins, rsid, side="right"))
        start = int(self._cumsum_bins[ti - 1]) if ti > 0 else 0
        return self.task_ids[ti], rsid - start

    def _reconfigure_slot(self, slot_idx: int, ltid: int) -> None:
        """Swap a slot's env to task `ltid` (rebuild bddl) iff its task changed."""
        if self._slot_task[slot_idx] == ltid:
            return
        seed = self.cfg.seed + ltid * 1000 + slot_idx
        dev = self._egl_render_devices[slot_idx % len(self._egl_render_devices)]
        env_fn = _LiberoEnvFn(bddl_path=self._task_bddl[ltid], seed=seed, egl_device_id=dev)
        self._slot_envs[slot_idx].reconfigure(env_fn)
        self._slot_task[slot_idx] = ltid
        # A task swap re-derives the predicates. Disable labeling rather than ship a
        # vector whose columns now mean something else: the GVF head count is fixed at
        # config time, so a changed predicate set cannot be honoured mid-run.
        got = tuple(self._slot_envs[slot_idx].subgoal_names)
        if self.subgoal_names and got != self.subgoal_names:
            logger.warning(
                f"slot {slot_idx} reconfigured to task {ltid}, whose BDDL subgoal predicates "
                f"{list(got)} differ from the pool's {list(self.subgoal_names)} — predicate "
                f"labeling disabled from here on."
            )
            self.subgoal_names = ()

    def make_pool(
        self,
        *,
        train: bool = True,
        task: int | None = None,
        trial_offset: int = 0,
        n_active: int | None = None,
    ) -> list[_EpState]:
        """Build the episode pool for one rollout/eval (slot order == buffer batch axis).

        Reconfigures envs whose task changed (cheap: reuses the live subprocess, just
        swaps the MuJoCo scene) and resets each to its trial init state (env.reset_to).
        Mirrors RLinf update_reset_state_ids + reset.

        task=<id> (single-task eval): force the slots to that task with DISTINCT trials
            ((trial_offset + slot) % n_trials). Lets one persistent pool be reused across
            tasks — build envs once, then loop tasks reconfiguring in place (no respawn,
            no model reload). Use this for per-task only_eval. n_active<batch_size builds a
            SHORTER pool (only that many slots/episodes) so a task's episodes/task can be
            evaluated in sequential batches with a capped concurrent env count;
            trial_offset advances the trial window across batches (e.g. 0..31 then 32..49).
        train=True  (rollout): one RANDOM reset_state_id per GROUP, repeated across the
            group's group_size members -> identical start within a group (a valid
            baseline). Resampled every call; tasks are (re)assigned to groups randomly,
            so all tasks are covered over rollouts.
        train=False (eval, task=None): DISTINCT reset_state_ids per slot from a persistent
            cursor for even, deterministic coverage across all tasks/trials.

        n_active caps the pool length in EVERY branch, so one env pool can be sized for
        the widest consumer (eval) while rollouts run a narrower slice of it. That is
        how --rollout_num_envs < --total_num_envs works: the extra subprocesses stay
        alive and idle during collection rather than being built and torn down, which
        would cost an EGL/MuJoCo rebuild per iteration. Slots are taken from the FRONT,
        so a capped rollout always uses slots [0, n_active) and `_reconfigure_slot`
        leaves the rest on whatever task they last held.
        """
        group_size = max(1, self.group_size)
        m = self.batch_size if n_active is None else max(1, min(int(n_active), self.batch_size))
        if task is not None:
            n_trials = len(self._task_init_states[task])
            slot_task_trial = [(task, (trial_offset + slot_idx) % n_trials) for slot_idx in range(m)]
        elif train:
            # Sample per GROUP over the full pool, then truncate to m: keeps the
            # group_size structure intact (a truncated pool must not split a group, or
            # a group-relative baseline would see a partial group). With
            # group_size=1 -- the TD3 case -- this is a plain
            # prefix of an independently sampled batch.
            gids = self._np_rng.integers(0, self._total_reset_states, size=self.n_groups)
            rsids = np.repeat(gids, group_size)[:m]  # [m]
            slot_task_trial = [self._reset_state_to_task_trial(int(r)) for r in rsids]
        else:
            rsids = (self._eval_cursor + np.arange(m)) % self._total_reset_states
            self._eval_cursor = int((self._eval_cursor + m) % self._total_reset_states)
            slot_task_trial = [self._reset_state_to_task_trial(int(r)) for r in rsids]

        pool = []
        for slot_idx, (ltid, trial) in enumerate(slot_task_trial):
            self._reconfigure_slot(slot_idx, ltid)
            tok_t, mask_t, _ = self._task_tok[ltid]
            init_state = self._task_init_states[ltid][trial]
            pool.append(_EpState(self._slot_envs[slot_idx], tok_t, mask_t, ltid, init_state=init_state))
        return pool

    def close(self):
        for env in self._slot_envs:
            try:
                env.close()
            except Exception:
                pass


class _BridgeSlotProxy:
    """Matches `_SubprocEnv`'s interface EXACTLY (same method names, same
    per-call granularity: one physical action per send_step/recv_step pair —
    not a whole action_chunk), so `_EpState` and `train_rlt_libero.collect()`
    don't need to know or care which one they're talking to."""

    def __init__(self, channel: _BridgeChannel, action_env_dim: int):
        self._channel = channel
        self._action_env_dim = action_env_dim
        self._telemetry: dict | None = None

    def set_telemetry(self, tel: dict | None) -> None:
        """Attach a payload to the NEXT step command, then clear it.

        Lets the server relay per-chunk diagnostics (critic value, z_rl summary) to
        the robot-side operator, who is the one standing at the arm. Piggybacking on
        the existing step command keeps the request/response lockstep intact — an
        out-of-band message would desynchronize the strict one-command-per-report
        cadence the bridge relies on. Sent once per chunk, not once per physical
        step, so it costs one small dict per ~10 actions.
        """
        self._telemetry = tel

    def reset_to(self, init_state: dict) -> dict:
        self._channel.push_command({"type": "reset", "task_id": init_state["task_id"], "trial": init_state["trial"]})
        # The robot's ready ping (sent right on connect, obs=None -- see
        # robot_bridge.py's WIRE PROTOCOL note) arrives before the
        # settled post-reset obs it reports after go_home(). Skip it instead
        # of handing back obs=None as the episode's starting observation.
        msg = self._channel.pull_obs()
        while msg.get("obs") is None:
            msg = self._channel.pull_obs()
        return msg["obs"]

    def send_step(self, action: np.ndarray) -> None:
        cmd = {"type": "step", "actions": action}
        if self._telemetry is not None:
            cmd["telemetry"] = self._telemetry
            self._telemetry = None
        self._channel.push_command(cmd)

    def recv_step(self) -> tuple[dict, bool, dict]:
        msg = self._channel.pull_obs()
        obs = msg["obs"]
        done = bool(msg.get("done"))
        info = {"reward": msg.get("reward")}
        return obs, done, info

    def close(self) -> None:
        pass  # channel/connection teardown is handled at the RealRobotEnvWorker level


class RealRobotEnvWorker:
    """Real-robot analogue of `EnvWorker`. Identical public surface:
    .state_norm .action_norm .batch_size .make_pool(...) .close()
    """

    def __init__(
        self,
        cfg,
        tok,
        action_dim: int,
        state_norm,
        action_norm,
        channels: list[_BridgeChannel],
        task_prompts: dict[int, str] | None = None,
    ):
        """
        channels: one `_BridgeChannel` per physical robot, each backing a
            `RobotBridgeServer` your robot desktop script(s) connect to.
        task_prompts: {task_id: language instruction}. Replaces LIBERO's
            benchmark.get_task(...).language lookup.

        NOTE on the very first message of a connection: the handler in
        robot_bridge.py always waits for an incoming message before
        it will hand back a queued command — so on connect, before entering
        its normal loop, the robot script must send ONE "ready ping"
        ({"obs": None, "reward": None, "done": None}) so the first
        reset_to() has something to respond to.
        """
        self.cfg = cfg
        self.tok = tok
        self.action_dim = action_dim
        self.state_norm = state_norm
        self.action_norm = action_norm
        self.task_prompts = task_prompts or {0: "perform the task"}
        self.task_ids = cfg.task_ids if getattr(cfg, "task_ids", None) else list(self.task_prompts.keys())

        self._channels = channels
        if not self._channels:
            raise ValueError("RealRobotEnvWorker needs at least one _BridgeChannel")
        self.batch_size = len(self._channels)  # one slot per physical robot; grow via more channels, not repeats
        if cfg.total_num_envs != self.batch_size:
            logger.warning(
                f"cfg.total_num_envs={cfg.total_num_envs} but {self.batch_size} channel(s) were "
                f"provided — using {self.batch_size} (one slot per physical robot)."
            )

        self._task_tok: dict[int, tuple] = {}
        for tid, lang in self.task_prompts.items():
            tok_prompt, tok_mask = self.tok.tokenize(lang)
            self._task_tok[tid] = (np.array(tok_prompt), np.array(tok_mask), lang)

        self._trial_cursor = 0
        logger.info(f"RealRobotEnvWorker: {self.batch_size} physical robot(s) via bridge channels")

    def make_pool(
        self,
        *,
        train: bool = True,
        task: int | None = None,
        trial_offset: int = 0,
        n_active: int | None = None,
    ) -> list[_EpState]:
        """Same semantics/signature as EnvWorker.make_pool. Each slot
        is an independent episode on its own physical robot (GAE: no matched-
        start requirement between slots).

        n_active caps the pool in every branch here too, matching EnvWorker. On this
        path a capped pool means IDLING a physical robot for that phase, which is
        rarely what an operator wants -- but silently ignoring the argument the
        training loop now passes on every rollout would be worse.
        """
        m = self.batch_size if n_active is None else max(1, min(int(n_active), self.batch_size))
        if task is not None:
            slot_task_trial = [(task, trial_offset + i) for i in range(m)]
        elif train:
            rng = np.random.default_rng()
            tids = rng.choice(self.task_ids, size=m)
            slot_task_trial = [(int(t), int(rng.integers(1_000_000))) for t in tids]
        else:
            slot_task_trial = [(self.task_ids[i % len(self.task_ids)], self._trial_cursor + i) for i in range(m)]
            self._trial_cursor += m

        pool = []
        for slot_idx, (tid, trial) in enumerate(slot_task_trial):
            proxy = _BridgeSlotProxy(self._channels[slot_idx], self.cfg.action_env_dim)
            tok_t, mask_t, _ = self._task_tok[tid]
            init_state = {"task_id": tid, "trial": trial}
            pool.append(_EpState(proxy, tok_t, mask_t, tid, init_state=init_state))
        return pool

    def close(self) -> None:
        pass  # channels/servers are owned by whoever constructed them (see train script)
