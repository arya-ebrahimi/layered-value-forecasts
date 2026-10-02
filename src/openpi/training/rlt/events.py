"""Per-episode event diagnostics from the BDDL subgoal predicates.

`obs["subgoal"][k]` (envs.build_subgoal_evaluator) is a per-step BOOLEAN OF THE
CURRENT STATE: `_check_grasp` is true on every step the object is held and false the
instant it is dropped, so it toggles freely and fires again on a regrasp. These stats
count its rising edges per episode, so the RLT and LVF arms log identical, directly
comparable event-rate columns.

INITIAL CONDITIONS ARE NOT ACHIEVEMENTS. A predicate already true at step 0 (a drawer
that starts open, a resting contact) describes the scene, not something the policy did,
so it is never counted -- the same convention labeling.auto_label_libero_predicates
uses for the GVF cumulants.

ATTRIBUTION. `rec.obs_seq` holds L+1 observations for L executed actions, so index t+1
is the state AFTER action t. An event is attributed to step t iff it is a rising edge
at index t+1.
"""

from __future__ import annotations

import numpy as np

EXCLUDED_EVENTS: frozenset[str] = frozenset({"goal"})
"""Predicates that are the task reward rather than a subgoal: `goal` is literally
env._check_success(), so its rate is just the success rate."""


def indicator_trace(obs_seq, subgoal_names: tuple[str, ...], names: tuple[str, ...]) -> np.ndarray:
    """Raw per-step state indicators for `names`, shape [L+1, K].

    Row t is the predicate vector at `obs_seq[t]`, so row 0 is the INITIAL state and row
    t+1 is the state after action t. Missing/short `subgoal` vectors give zeros (the
    real-robot path ships none).
    """
    n_obs, k = len(obs_seq), len(names)
    out = np.zeros((n_obs, k), np.float32)
    if not k or not n_obs:
        return out
    col = {n: i for i, n in enumerate(subgoal_names)}
    idx = [col.get(n) for n in names]
    for t, o in enumerate(obs_seq):
        v = o.get("subgoal") if isinstance(o, dict) else None
        if v is None:
            continue
        v = np.asarray(v, np.float32)
        for j, c in enumerate(idx):
            if c is not None and c < v.size:
                out[t, j] = 1.0 if v[c] > 0.5 else 0.0
    return out


def rising_edges(ind: np.ndarray) -> np.ndarray:
    """Per-executed-step rising edges, shape [L, K], from an [L+1, K] indicator trace."""
    if ind.shape[0] <= 1 or ind.shape[1] == 0:
        return np.zeros((max(ind.shape[0] - 1, 0), ind.shape[1]), np.float32)
    return ((ind[1:] > 0.5) & (ind[:-1] <= 0.5)).astype(np.float32)


def episode_stats(
    obs_seq,
    subgoal_names: tuple[str, ...],
    names: tuple[str, ...],
    *,
    ep_len: int,
    success: bool,
) -> dict[str, float]:
    """Per-episode event diagnostics.

    Keys, per event i:
        fire_<i>    1.0 if the event fired at all this episode
        count_<i>   number of raw rising edges (a regrasp counts again)
    plus:
        grasp_no_success   1.0 if `grasp` fired and the task did NOT succeed
        ret_terminal       undiscounted terminal return (1.0 on success)
    """
    L = int(ep_len)
    out: dict[str, float] = {}
    if names and L > 0:
        ind = indicator_trace(obs_seq, subgoal_names, names)
        if ind.shape[0] < L + 1:
            ind = np.concatenate([ind, np.repeat(ind[-1:], L + 1 - ind.shape[0], axis=0)], axis=0)
        raw = rising_edges(ind[: L + 1])
        for j, name in enumerate(names):
            c = float(raw[:, j].sum())
            out[f"fire_{name}"] = 1.0 if c > 0 else 0.0
            out[f"count_{name}"] = c
        if "grasp" in names:
            out["grasp_no_success"] = float(out["fire_grasp"] > 0 and not success)
    out["ret_terminal"] = 1.0 if success else 0.0
    return out


def aggregate_stats(per_episode: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every key over episodes (fractions for the 0/1 keys, means for counts)."""
    if not per_episode:
        return {}
    keys = sorted({k for d in per_episode for k in d})
    return {k: float(np.mean([d[k] for d in per_episode if k in d])) for k in keys}
