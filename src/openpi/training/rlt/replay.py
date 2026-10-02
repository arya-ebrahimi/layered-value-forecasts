"""Host-side numpy ring replay buffer for RLT chunk-level transitions.

Stores precomputed RL tokens (z_rl) so TD3 updates never touch the frozen VLA.
Next-state fields are stored explicitly: chunk subsampling (stride < C) and
episode boundaries break any "next = current row + 1" assumption.

Per transition:
  zrl / next_zrl        [zrl_dim]           float16 (values O(1); quantization << signal)
  proprio / next_proprio [proprio_dim]      float32
  action / ref / next_ref [chunk_action_dim] float32, normalized model action space
  reward                []                  float32, within-chunk discounted sum sum_i gamma^i r_i
  discount              []                  float32, 0 on success inside the chunk, else gamma^n
  success               []                  float32, EPISODE-level outcome flag (1 if the
                                            episode eventually succeeded) — used only for
                                            stratified sampling, never in the TD backup
  grid_t                []                  int32, the transition's START timestep
  ep_len                []                  int32, its episode's total length L

With subgoal GVF channels configured (see rlt/labeling.py), three more kinds of
column ride alongside `reward`/`discount`, one pair per channel plus one shared:
  c_<name>              []                  float32, the channel's within-chunk cumulant
  z_<name>              []                  float32, its continuation (gamma_k**n, or 0
                                            once the channel has terminated)
  phase                 []                  int32, the episode phase at the chunk's start,
                                            indexing which actor-loss GVF terms are active
  next_phase            []                  int32, the same for the NEXT state (the one
                                            `next_zrl`/`next_proprio` describe)
The cumulant/continuation columns have deliberately NO `next_` variants: the GVF backup
reads `next_ref` / the `next_zrl`+`next_proprio` state that already exist. `phase` does
need one, because gvf_mode="lookahead" evaluates its potential Phi(x', a') under the
NEXT state's channel weights (td3.py). On terminal rows its value is irrelevant
(discount = 0 zeroes the term it feeds), but the column must exist. When no channels
are configured none of these columns exist at all, so the buffer is byte-identical to
the baseline.

grid_t/ep_len are BOOKKEEPING ONLY — no loss, no sampling weight, nothing in the TD
backup reads them. They exist so the per-state trust radius eps can be plotted against
where in a trajectory it was spent (normalized progress t/L, or time-to-end t - L), which
is how the budget is shown migrating between bottlenecks over training. Phase-0 measured
t/L explaining only 1.9% of the variance of g, so the interesting axis is unlikely to be
t/L itself — storing both endpoints keeps every derived axis available offline.
"""

from __future__ import annotations

import numpy as np

from openpi.training.rlt.labeling import GvfChannel

FIELD_DTYPES = {
    "zrl": np.float16,
    "next_zrl": np.float16,
    "proprio": np.float32,
    "next_proprio": np.float32,
    "action": np.float32,
    "ref": np.float32,
    "next_ref": np.float32,
    "reward": np.float32,
    "discount": np.float32,
    "success": np.float32,
    "grid_t": np.int32,
    "ep_len": np.int32,
}

GVF_FIELD_DTYPES = {"phase": np.int32, "next_phase": np.int32}
"""Per-channel columns get np.float32; `phase`/`next_phase` are the shared int32 ones."""


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        zrl_dim: int,
        proprio_dim: int,
        chunk_action_dim: int,
        channels: tuple[GvfChannel, ...] = (),
    ):
        self.channels = tuple(channels)
        shapes = {
            "zrl": (zrl_dim,),
            "next_zrl": (zrl_dim,),
            "proprio": (proprio_dim,),
            "next_proprio": (proprio_dim,),
            "action": (chunk_action_dim,),
            "ref": (chunk_action_dim,),
            "next_ref": (chunk_action_dim,),
            "reward": (),
            "discount": (),
            "success": (),
            "grid_t": (),
            "ep_len": (),
        }
        dtypes = dict(FIELD_DTYPES)
        if self.channels:
            for k, dt in GVF_FIELD_DTYPES.items():
                shapes[k] = ()
                dtypes[k] = dt
            for ch in self.channels:
                shapes[ch.cumulant_key] = ()
                shapes[ch.continuation_key] = ()
                dtypes[ch.cumulant_key] = np.float32
                dtypes[ch.continuation_key] = np.float32
        self.data = {k: np.zeros((capacity, *shapes[k]), dtype=dtypes[k]) for k in shapes}
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, transitions: dict[str, np.ndarray]) -> None:
        keys = set(self.data)
        if set(transitions) != keys:
            raise ValueError(f"expected fields {sorted(keys)}, got {sorted(transitions)}")
        n = len(transitions["reward"])
        if n == 0:
            return
        idx = (self.ptr + np.arange(n)) % self.capacity
        for k, buf in self.data.items():
            v = np.asarray(transitions[k])
            if v.shape != (n, *buf.shape[1:]):
                raise ValueError(f"field {k}: expected shape {(n, *buf.shape[1:])}, got {v.shape}")
            buf[idx] = v.astype(buf.dtype)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch_size: int, rng: np.random.Generator, success_frac: float = 0.0) -> dict[str, np.ndarray]:
        """Uniform sample; with success_frac > 0, that fraction of the batch is drawn
        from success-EPISODE rows (stratified). Motivation: at ~20% task success with
        truncation at 480 steps, failure episodes contribute ~5x more rows each, so
        ~95% of a uniform batch has target ~0 and the critic's fit (and its dQ/da)
        collapses toward an action-independent constant; oversampling the success
        manifold keeps the value-carrying region represented. Targets are unchanged —
        only the sampling weights shift. Falls back to uniform when no success rows.
        """
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, self.size, size=batch_size)
        if success_frac > 0.0:
            succ_idx = np.flatnonzero(self.data["success"][: self.size] > 0.5)
            if len(succ_idx):
                n_s = round(batch_size * success_frac)
                idx[:n_s] = rng.choice(succ_idx, size=n_s, replace=True)
        return {k: buf[idx] for k, buf in self.data.items()}
