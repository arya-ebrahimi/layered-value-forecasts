"""Online per-dim standardization for z_rl (RL Token paper, arXiv 2604.23073).

RLTokenizer._prep_targets RMS-normalizes reconstruction targets during stage-1
training (src/openpi/models/rl_token.py), which makes z_rl carry a large,
near-constant per-dim offset (observed ||z_rl|| ~ 46.5, near-constant across
states) with a small state-varying delta riding on top. Concatenated with
proprio into the RL state x, this offset dominates x's magnitude and starves
the critic's action-conditioning (see RltCritic in networks.py).

Retraining stage 1 with normalize_targets=False would remove the offset at the
source but invalidates the existing frozen tokenizer checkpoint. RunningNorm
fixes it post-hoc, at stage-2 extraction time: standardize z_rl to zero mean /
unit std using statistics accumulated online during the warmup rollouts (raw
VLA reference, no RL updates yet), then freeze so the critic's input
distribution stays stationary once training starts.
"""

from __future__ import annotations

import numpy as np


class RunningNorm:
    """Per-dim running mean/std (parallel Welford) with an explicit freeze."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.count = 0
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.eps = eps
        self.frozen = False

    def update(self, batch: np.ndarray) -> None:
        if self.frozen or batch.shape[0] == 0:
            return
        batch = batch.astype(np.float64)
        b_count = batch.shape[0]
        b_mean = batch.mean(axis=0)
        b_var = batch.var(axis=0)
        delta = b_mean - self.mean
        tot_count = self.count + b_count
        self.mean = self.mean + delta * b_count / tot_count
        self.m2 = self.m2 + b_var * b_count + delta**2 * self.count * b_count / tot_count
        self.count = tot_count

    def freeze(self) -> None:
        self.frozen = True

    @property
    def std(self) -> np.ndarray:
        var = self.m2 / max(self.count - 1, 1)
        return np.sqrt(var + self.eps)

    def normalize(self, batch: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return batch.astype(np.float32)
        return ((batch - self.mean) / self.std).astype(np.float32)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """Update stats (no-op once frozen) then normalize."""
        self.update(batch)
        return self.normalize(batch)

    def save(self, path) -> None:
        """Persist the accumulated stats next to the TD3 checkpoint.

        Load-bearing for standalone eval/resume: the actor is trained on
        STANDARDIZED z_rl, so a fresh RunningNorm (count == 0, normalize() a
        no-op) would feed it raw z_rl carrying the tokenizer's ~46.5 offset --
        a distribution shift large enough to make the loaded policy useless
        while still running without error.
        """
        np.savez(path, count=np.int64(self.count), mean=self.mean, m2=self.m2, frozen=np.bool_(self.frozen))

    @classmethod
    def load(cls, path) -> RunningNorm:
        z = np.load(path)
        obj = cls(dim=len(z["mean"]))
        obj.count = int(z["count"])
        obj.mean = z["mean"].astype(np.float64)
        obj.m2 = z["m2"].astype(np.float64)
        obj.frozen = bool(z["frozen"])
        return obj
