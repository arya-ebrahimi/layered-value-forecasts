"""Affine map between the VLA's normalized action space and RL-canonical [-1, 1].

Measured from the warmup reference-chunk distribution by scripts/train_rlt_libero.py.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _from_normalized(v: float, d: int, kind: str, p1, p2) -> float:
    """Inverse of the normalization, for logging a calibration in raw robot units."""
    if kind == "quantile":
        return (v + 1.0) / 2.0 * (p2[d] - p1[d] + 1e-6) + p1[d]
    return v * (p2[d] + 1e-6) + p1[d]


@dataclasses.dataclass
class ActionSpace:
    """Per-dim affine map between the VLA's normalized action space and the RL
    networks' canonical [-1, 1] space, measured from the warmup reference-chunk
    distribution.

    Some checkpoints (e.g. externally-produced few-shot SFTs) ship norm stats whose
    quantile ranges do not contain the policy's own outputs, so normalized reference
    actions land far outside [-1, 1] — silently breaking the tanh actor (cannot
    represent the reference -> BC floor), the +-1 execution clip (corrupts every
    step), and the sigma=0.1 noise scale. Measuring the actual distribution makes the
    RL layer convention-agnostic; for well-behaved checkpoints this map is ~identity.

    Buffer actions/refs live in RL space; from_rl() is applied just before splicing
    executed actions back into the VLA sample for unnormalization.
    """

    center: np.ndarray  # [chunk_action_dim]
    halfwidth: np.ndarray  # [chunk_action_dim]

    def to_rl(self, a: np.ndarray) -> np.ndarray:
        return ((a - self.center) / self.halfwidth).astype(np.float32)

    def from_rl(self, a: np.ndarray) -> np.ndarray:
        return (a * self.halfwidth + self.center).astype(np.float32)

    def save(self, path) -> None:
        np.savez(path, center=self.center, halfwidth=self.halfwidth)

    @classmethod
    def load(cls, path) -> ActionSpace:
        z = np.load(path)
        return cls(center=z["center"].astype(np.float32), halfwidth=z["halfwidth"].astype(np.float32))

    @classmethod
    def from_ref_samples(
        cls,
        refs: np.ndarray,
        action_env_dim: int,
        margin: float = 1.25,
        *,
        action_norm=None,
        trigger_raw_scale: float = 1.0,
        min_halfwidth: float = 0.05,
        discrete_dims: dict[int, float] | None = None,
    ) -> ActionSpace:
        """refs [N, C*ad] reference chunks in VLA-normalized space. Per-ACTION-dim
        quantile bounds (aggregated over chunk positions), tiled back to C*ad.

        action_norm: the (kind, p1, p2) tuple from train_rlt_libero.build_norm. When given,
        the LAST of the action_env_dim columns is calibrated from the norm stats
        instead of from sample quantiles, because it is a discrete open/close
        TRIGGER (franka_raw: -1=close, 0=no-op, +1=open; a 1-2 frame pulse in
        ~0.4% of timesteps -- see
        examples/franka_raw/convert_franka_raw_to_lerobot.py), not a
        continuously-varying signal. A 0.5/99.5 percentile window over a channel
        that is ~0 for ~99% of samples collapses to zero width (observed:
        halfwidth pinned to the 0.05 floor), so to_rl() sends the pulse far
        outside [-1, 1], the +-action_clip clip crushes it, and from_rl() returns
        a value ~80x too small to cross the robot's +-0.5 trigger threshold --
        i.e. the gripper can never close once rollouts stop executing the raw
        reference. Mapping RL +-1 onto normalize(+-trigger_raw_scale) instead
        makes the trigger exactly representable and keeps it symmetric (an
        asymmetric range would let RL 0 decode to a spurious close).
        """
        n, flat = refs.shape
        per_dim = refs.reshape(n * (flat // action_env_dim), action_env_dim)
        lo = np.quantile(per_dim, 0.005, axis=0)
        hi = np.quantile(per_dim, 0.995, axis=0)

        # ── degenerate-window fallback ──────────────────────────────────────
        # A 0.5/99.5 window is the right robust estimator for a continuously-varying
        # channel and the WRONG one for a near-BIMODAL channel -- one that rests at a
        # constant value for >99% of steps and jumps on a rare, discrete command. The
        # window then has ~zero width and `min_halfwidth` catches it, but a floor is
        # not a measurement: it silently becomes a 1/min_halfwidth AMPLIFIER on exactly
        # the rare samples the quantiles excluded.
        #
        # Measured on a real franka insertion run (checkpoints/.../ram_gvf_v1): action
        # dim 5 (yaw) sat at 0.021 with p99 0.034 and spiked to 6.68 on 0.36% of chunk
        # positions -- the rotate step of the insertion. halfwidth pinned to 0.05, so
        # to_rl sent those to |ref| = 133. Downstream: the tanh actor cannot represent
        # them (bc_loss 1072, 98% of it from that one column, force_ratio 8.7), and
        # the +-1 execution clip caps the channel at ~1/130 of the commanded yaw, so
        # the policy could never perform the rotate at all once RL took over. Exactly
        # the failure the gripper branch below was written for -- but that branch only
        # special-cases the LAST column, and this was not it.
        #
        # So: when the quantile window is degenerate, fall back to the observed FULL
        # RANGE, which is guaranteed to contain every sample (with `margin` to spare).
        # A truly constant channel has min == max and still lands on the floor, which
        # is correct -- there is nothing to represent.
        #
        # The trade this makes: one genuinely spurious outlier in an otherwise narrow
        # channel now widens that channel instead of being clipped, coarsening
        # resolution around its resting value. That is the safer direction (a coarse
        # command is recoverable; an uncommandable channel is not), and the warning
        # below makes it visible rather than silent either way.
        narrow = (hi - lo) / 2.0 * margin < min_halfwidth
        if narrow.any():
            lo = np.where(narrow, per_dim.min(axis=0), lo)
            hi = np.where(narrow, per_dim.max(axis=0), hi)

        center7 = (hi + lo) / 2.0
        halfwidth7 = np.maximum((hi - lo) / 2.0 * margin, min_halfwidth)
        # ── discrete/trigger columns: calibrate from NORM STATS, not samples ───
        # A rare two-level command must not be estimated from a finite warmup sample.
        # Measured: refitting the same task twice gave dim 5 observed ranges of
        # [-6.628, +6.681] (both fire directions present) and [-6.305, +0.036] (only
        # the negative one) — the second produces an ASYMMETRIC map centred at -3.13,
        # under which a positive fire needs RL +2.45, clips to +1, and decodes to 12%
        # of the command, while RL 0 stops being neutral (-0.048 rad of standing yaw).
        # Whether a sample contains both levels is luck; the norm stats always know
        # where the levels are, so declare the column instead of estimating it.
        #
        # `discrete_dims` maps column index -> the raw full-scale command, e.g.
        # {5: 0.1, 6: 1.0} for a +-0.1 rad yaw trigger and a +-1 gripper. Passing
        # nothing preserves the historical behaviour exactly: the LAST column is
        # treated as the gripper trigger at `trigger_raw_scale`.
        if action_norm is not None:
            kind, p1, p2 = action_norm
            dims = discrete_dims if discrete_dims is not None else {action_env_dim - 1: trigger_raw_scale}

            def _to_normalized(raw: float, d: int) -> float:
                if kind == "quantile":
                    return (raw - p1[d]) / (p2[d] - p1[d] + 1e-6) * 2.0 - 1.0
                return (raw - p1[d]) / (p2[d] + 1e-6)

            for d, raw_scale in sorted(dims.items()):
                if not 0 <= d < action_env_dim:
                    raise ValueError(f"discrete dim {d} out of range for action_env_dim={action_env_dim}")
                n_pos = _to_normalized(raw_scale, d)
                n_neg = _to_normalized(-raw_scale, d)
                center7[d] = (n_pos + n_neg) / 2.0
                # Same `margin` as the quantile dims: without it a full-scale command
                # lands at exactly RL -+1, which the tanh actor can only reach by
                # saturating. It also keeps this path a no-op for continuously-held
                # gripper channels (LIBERO measures ~1.26 by quantile; full-range x
                # margin gives 1.25), so only the degenerate sparse-pulse case moves.
                halfwidth7[d] = abs(n_pos - n_neg) / 2.0 * margin
                narrow[d] = False  # calibrated, not estimated — nothing to warn about
                logger.info(
                    "ActionSpace: dim %d calibrated from norm stats (%s) as a discrete +-%.4g trigger -> "
                    "center=%.4f halfwidth=%.4f (RL 0 -> raw %+.5g, RL +-1 -> raw %+.4g/%+.4g)",
                    d,
                    kind,
                    raw_scale,
                    center7[d],
                    halfwidth7[d],
                    _from_normalized(center7[d], d, kind, p1, p2),
                    _from_normalized(center7[d] + halfwidth7[d], d, kind, p1, p2),
                    _from_normalized(center7[d] - halfwidth7[d], d, kind, p1, p2),
                )
        reps = flat // action_env_dim
        space = cls(
            center=np.tile(center7, reps).astype(np.float32),
            halfwidth=np.tile(halfwidth7, reps).astype(np.float32),
        )
        logger.info(
            "ActionSpace from %d ref chunks: per-dim lo=%s hi=%s%s",
            n,
            np.round(lo, 3).tolist(),
            np.round(hi, 3).tolist(),
            " (discrete dims recalibrated from norm stats — see above)" if action_norm is not None else "",
        )
        # A degenerate quantile window means the channel is near-bimodal (rare discrete
        # commands) or constant. Never let that pass silently: which of the two it is
        # decides whether the widened range is right, and only the operator knows.
        report = narrow.copy()  # discrete dims already cleared above
        if report.any():
            logger.warning(
                "ActionSpace: dims %s had a degenerate 0.5/99.5 quantile window (<%.3g halfwidth) and were "
                "widened to their observed FULL RANGE instead of pinned to the floor: %s. A floor there is a "
                "1/%.3g amplifier on exactly the rare samples the quantiles excluded — it makes the channel "
                "unrepresentable for the tanh actor and uncommandable through the +-1 execution clip. Check "
                "whether each of these is a rare DISCRETE command (widening is correct) or a spurious outlier "
                "in an otherwise constant channel (widening coarsens the channel; consider calibrating it "
                "explicitly, the way the gripper trigger is).",
                np.flatnonzero(report).tolist(),
                min_halfwidth,
                {
                    int(i): f"halfwidth {halfwidth7[i]:.4g} (range [{lo[i]:.4g}, {hi[i]:.4g}])"
                    for i in np.flatnonzero(report)
                },
                min_halfwidth,
            )
        # Declared discrete dims are calibrated, not measured, so their sample range is
        # not evidence about the checkpoint's normalization and must not trip the warning.
        keep = np.ones(action_env_dim, bool)
        if action_norm is not None:
            keep[list(discrete_dims if discrete_dims is not None else {action_env_dim - 1: 0})] = False
        check_lo, check_hi = lo[keep], hi[keep]
        if (check_lo < -1.05).any() or (check_hi > 1.05).any():
            logger.warning(
                "Reference actions leave [-1, 1] in this checkpoint's normalized space — "
                "the affine RL-space map is load-bearing here (without it, the tanh actor/"
                "clip/noise scale silently break)."
            )
        return space
