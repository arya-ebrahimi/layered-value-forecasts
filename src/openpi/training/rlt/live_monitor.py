"""Live critic / z_rl telemetry for real-robot RLT evaluation.

Attaches to `train_rlt_libero.evaluate` through its `critic_apply`/`monitor` hook
and reports, once per action chunk:

  * Q(x, a) for the chunk about to execute — the EXACT value, since the hook fires
    where both the RL state and the post-clip executed action exist. (The offline
    path in an offline analysis has to reconstruct proprio and the
    action from a z_rl-only dump; this one does not.)
  * a compact read on z_rl: its norm, the per-chunk change, and a coarse profile.

Three surfaces, because the two machines see different things:

  server terminal — one line per chunk, with an in-episode sparkline of Q.
  robot terminal  — the same line, relayed over the bridge in the step command
                    (`set_telemetry`), since the operator standing at the arm is
                    the one who needs to see it. Rendered by eval_pi05_real.py.
  files           — a live-updating PNG (`--live_plot`) and, at the end,
                    eval_trace.npz holding z_rl/proprio/action/Q for every chunk.
                    That npz is the complete record the rollout dump is missing.

Everything here is observational: the executed action is computed before the hook
runs and is never modified, so eval numbers are identical with the monitor on.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import queue
import threading
import time

import numpy as np

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> str:
    """Unicode bar chart of a 1-D series — the only 'plot' that survives an ssh session."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return ""
    lo = float(np.min(v)) if lo is None else lo
    hi = float(np.max(v)) if hi is None else hi
    if hi - lo < 1e-9:
        return _SPARK[len(_SPARK) // 2] * v.size
    idx = np.clip(((v - lo) / (hi - lo) * (len(_SPARK) - 1)).round().astype(int), 0, len(_SPARK) - 1)
    return "".join(_SPARK[i] for i in idx)


class LiveMonitor:
    """Per-chunk critic/z_rl telemetry sink. One instance per eval run.

    Args:
        n_envs: pool size (one physical robot per slot).
        out_dir: where eval_trace.npz and the live PNG are written. None disables both.
        live_plot: rewrite <out_dir>/live_monitor.png every `plot_every` chunks.
        plot_every: redraw period in chunks. A redraw costs ~150ms, which is a real
            fraction of a chunk's wall time on a 10-20Hz arm, so it is throttled and
            never on the critical path of an action.
        spark_width: how many recent chunks the terminal sparkline shows.
        q_reduce: "min" matches TD3's own pessimistic estimate; "mean" is smoother.
    """

    def __init__(
        self,
        n_envs: int,
        out_dir: Path | None = None,
        *,
        live_plot: bool = False,
        save_plots: bool = False,
        plot_every: int = 1,
        spark_width: int = 40,
        q_reduce: str = "min",
        print_fn=print,
    ):
        self.n_envs = n_envs
        self.out_dir = Path(out_dir) if out_dir else None
        self.live_plot = live_plot and self.out_dir is not None
        # Keep one PNG PER EPISODE (ep_<n>_<success|failure>.png) instead of only the
        # single live_monitor.png, which every redraw overwrites. Independent of
        # live_plot: --save_plots alone renders once per episode, at episode END, so it
        # costs one ~150ms draw per episode and never lands between a chunk's
        # observation and its action.
        self.save_plots = save_plots and self.out_dir is not None
        self.plot_every = max(1, plot_every)
        self.spark_width = spark_width
        self.q_reduce = q_reduce
        self.print_fn = print_fn
        self.episode = 0
        self._chunks = 0
        self._t0 = time.time()
        # Per-slot in-episode history, reset by on_episode_start.
        self._hist: list[dict] = [self._new_hist() for _ in range(n_envs)]
        # Every chunk of every episode, flushed to eval_trace.npz at the end.
        self._trace: list[dict] = []
        # Rendering runs on its own thread: on_chunk sits between the observation and
        # the action going out, and a ~150ms savefig there would add latency to a
        # 30Hz control loop (a chunk is only ~330ms). The queue holds one frame and
        # drops the rest, so a slow draw falls behind instead of throttling the robot.
        self._frames: queue.Queue | None = None
        if self.live_plot or self.save_plots:
            self.out_dir.mkdir(parents=True, exist_ok=True)  # the worker writes before save_trace does
            self._frames = queue.Queue(maxsize=1)
            threading.Thread(target=self._draw_worker, daemon=True).start()

    @staticmethod
    def _new_hist() -> dict:
        return {"t": [], "q": [], "zn": [], "dz": [], "zrl": [], "prev_z": None}

    # ── hooks called by train_rlt_libero.evaluate ────────────────────────────

    def on_episode_start(self, episode: int, n_active: int) -> None:
        self.episode = episode
        self._hist = [self._new_hist() for _ in range(max(n_active, self.n_envs))]
        self.print_fn(f"[monitor] episode {episode + 1} started ({n_active} slot(s))")

    def on_chunk(self, j: int, step: int, zrl, proprio, action, q) -> dict:
        """Record one chunk for slot j; returns the payload to relay to the robot."""
        zrl = np.asarray(zrl, np.float32)
        q = np.asarray(q, np.float32)
        q_scalar = float(q.min() if self.q_reduce == "min" else q.mean())

        h = self._hist[j]
        dz = 0.0 if h["prev_z"] is None else float(np.abs(zrl - h["prev_z"]).sum())
        h["prev_z"] = zrl
        h["t"].append(step)
        h["q"].append(q_scalar)
        h["zn"].append(float(np.linalg.norm(zrl)))
        h["dz"].append(dz)
        h["zrl"].append(zrl)

        self._trace.append(
            {
                "episode": self.episode,
                "slot": j,
                "step": step,
                "q": q,
                "zrl": zrl,
                "proprio": np.asarray(proprio, np.float32),
                "action": np.asarray(action, np.float32),
            }
        )

        # 16 buckets of 128 dims: enough to see the token's gross activation pattern
        # shift in one line, without printing 2048 numbers.
        prof = zrl.reshape(16, -1).mean(axis=-1) if zrl.size % 16 == 0 else zrl[:16]
        line = (
            f"[ep {self.episode + 1} slot {j} t={step:4d}] "
            f"Q={q_scalar:+.3f} [{' '.join(f'{v:+.2f}' for v in np.atleast_1d(q))}] "
            f"{sparkline(np.array(h['q'][-self.spark_width :]))} "
            f"| |z|={h['zn'][-1]:.1f} d|z|={dz:7.1f} {sparkline(prof)}"
        )
        self.print_fn(line)

        self._chunks += 1
        # Gated on live_plot, NOT on the queue existing: --save_plots also creates the
        # queue, and this per-chunk path would then redraw every chunk and write a live
        # PNG nobody asked for -- and could fill the 1-slot queue that on_episode_end's
        # blocking put depends on.
        if self.live_plot and self._chunks % self.plot_every == 0:
            snap = [
                {"t": list(hh["t"]), "q": list(hh["q"]), "dz": list(hh["dz"]), "zrl": list(hh["zrl"])}
                for hh in self._hist
                if hh["q"]
            ]
            # Renderer still busy -> skip this frame rather than wait on it.
            with contextlib.suppress(queue.Full):
                self._frames.put_nowait((self.episode, snap, None))

        # The payload carries a pre-rendered `line`: the robot desktop has only
        # openpi_client installed, so it must not need this module to display it.
        return {
            "line": line,
            "q": [float(v) for v in np.atleast_1d(q)],
            "q_scalar": q_scalar,
            "z_norm": h["zn"][-1],
            "z_delta": dz,
            "step": int(step),
            "episode": int(self.episode) + 1,
        }

    def on_episode_end(self, episode: int, successes: list[bool]) -> None:
        for j, ok in enumerate(successes):
            h = self._hist[j] if j < len(self._hist) else None
            if not h or not h["q"]:
                continue
            q = np.array(h["q"])
            self.print_fn(
                f"[monitor] ep {episode + 1} slot {j}: {'SUCCESS' if ok else 'FAILURE'} "
                f"| Q first={q[0]:+.3f} last={q[-1]:+.3f} max={q.max():+.3f} mean={q.mean():+.3f} "
                f"| {len(q)} chunks"
            )
        for rec in self._trace:
            if rec["episode"] == episode and rec["slot"] < len(successes):
                rec["success"] = bool(successes[rec["slot"]])

        if self.save_plots and self._frames is not None:
            snap = [
                {"t": list(h["t"]), "q": list(h["q"]), "dz": list(h["dz"]), "zrl": list(h["zrl"])}
                for h in self._hist
                if h["q"]
            ]
            if snap:
                # Read the history BEFORE on_episode_start resets it for the next pass.
                if len(successes) == 1:
                    tag = "success" if successes[0] else "failure"
                else:
                    tag = f"{sum(bool(x) for x in successes)}of{len(successes)}"
                dest = self.out_dir / f"ep_{episode + 1:03d}_{tag}.png"
                # Blocking put, unlike the live path's put_nowait: this frame is the
                # permanent record of the episode and must not be dropped. Safe here --
                # the episode is over, so no arm is waiting on this thread.
                self._frames.put((episode, snap, dest))

    # ── outputs ──────────────────────────────────────────────────────────────

    def _draw_worker(self) -> None:
        """Render queued frames until the process exits (daemon thread).

        matplotlib is imported here, on the thread that uses it, so a run without
        --live_plot never pays for the import.
        """
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        while True:
            episode, snap, dest = self._frames.get()
            try:
                self._draw(plt, episode, snap, dest)
            except Exception as e:  # a broken frame must never take down an eval run
                self.print_fn(f"[monitor] live plot failed (continuing): {e}")
            finally:
                # Pairs with flush()'s join(); must run even on a failed draw or the
                # final flush would block forever on a frame that already errored.
                self._frames.task_done()

    def _draw(self, plt, episode: int, snap: list[dict], dest: Path | None = None) -> None:
        """Render one snapshot of the per-slot history.

        dest=None rewrites the live PNG (atomic swap, so a polling viewer never catches
        a half-written frame); a dest keeps that episode's figure permanently.
        """
        if not snap:
            return
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), facecolor="#fcfcfb", layout="constrained")
        for ax in axes:
            ax.set_facecolor("#fcfcfb")
            ax.grid(True, color="#d8d7d3", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        for j, h in enumerate(snap):
            axes[0].plot(h["t"], h["q"], color="#2a78d6", linewidth=1.8, marker="o", ms=3, label=f"slot {j}")
            axes[1].plot(h["t"], h["dz"], color="#eb6834", linewidth=1.5, label=f"slot {j}")
        axes[0].set_ylabel("Q (executed chunk)")
        axes[0].set_title(
            f"episode {episode + 1} — {'live' if dest is None else dest.stem}", loc="left", fontsize=11
        )
        axes[1].set_ylabel(r"$\sum_i |\Delta z_{rl,i}|$")
        axes[1].set_xlabel("env timestep")

        z = np.stack(snap[0]["zrl"]).T  # [D, chunks]
        im = axes[2].imshow(
            z, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="antialiased"
        )
        axes[2].set_ylabel("z_rl dim (all)")
        axes[2].set_xlabel("chunk index")
        axes[2].grid(False)
        fig.colorbar(im, ax=axes[2], pad=0.02, fraction=0.03)

        if dest is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest, dpi=110, facecolor=fig.get_facecolor())
            plt.close(fig)
            return
        # Keep the .png suffix — matplotlib picks the format from the extension.
        tmp = self.out_dir / ".live_monitor.tmp.png"
        fig.savefig(tmp, dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        # Atomic swap: an image viewer polling the file never catches a half-written
        # frame (a partially flushed PNG renders as garbage or fails to decode).
        tmp.replace(self.out_dir / "live_monitor.png")

    def flush(self, timeout: float = 60.0) -> None:
        """Wait for queued figures to be written.

        The render thread is a daemon, so anything still queued when the process exits is
        silently lost -- which would drop the last episode's plot on every run. Call this
        before save_trace() at the end of an eval.
        """
        if self._frames is None:
            return
        done = threading.Thread(target=self._frames.join, daemon=True)
        done.start()
        done.join(timeout)
        if done.is_alive():
            self.print_fn(f"[monitor] flush timed out after {timeout:.0f}s — some plots may be missing")

    def save_trace(self) -> Path | None:
        """Dump every chunk to eval_trace.npz — z_rl AND proprio AND the executed
        action AND Q, i.e. everything needed to recompute the critic offline."""
        if self.out_dir is None or not self._trace:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "eval_trace.npz"
        np.savez_compressed(
            path,
            zrl=np.stack([r["zrl"] for r in self._trace]),
            proprio=np.stack([r["proprio"] for r in self._trace]),
            action=np.stack([r["action"] for r in self._trace]),
            q=np.stack([r["q"] for r in self._trace]),
            step=np.array([r["step"] for r in self._trace], np.int32),
            episode=np.array([r["episode"] for r in self._trace], np.int32),
            slot=np.array([r["slot"] for r in self._trace], np.int32),
            success=np.array([r.get("success", False) for r in self._trace], bool),
        )
        self.print_fn(f"[monitor] wrote {path} ({len(self._trace)} chunks, {time.time() - self._t0:.0f}s)")
        return path
