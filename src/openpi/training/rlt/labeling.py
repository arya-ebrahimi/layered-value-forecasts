"""Subgoal-event channels: the spec, the human keyboard labeler, the sidecar
format, the LIBERO auto-labeler, and the chunk-level cumulant math.

WHY THIS EXISTS. The RLT task reward is sparse (~2% of buffer rows) and its credit
chain is ~40 chunks long, so dQ_task/da is near-noise at exactly the two states that
decide the episode. A subgoal GVF is labeled on EVERY episode (not just the
successful ones) and its credit chain is ~5-10 chunks, so it is both far better
estimated and far more action-sensitive. `td3.py` reads the resulting heads'
action-gradients directly in the actor loss.

EVENT SEMANTICS. A channel carries one of two labels per episode, and BOTH of them
terminate the channel -- they differ only in the terminal value:

    +1  the event OCCURRED here            cumulant 1.0, continuation 0.0
    -1  the event can NO LONGER occur      cumulant 0.0, continuation 0.0

So `grasp`: +1 = object grasped, -1 = dropped / the attempt is unrecoverable.
`coll`: +1 = collided with the holder, -1 = cleared it, the risk window is closed.
An episode that ends with neither label terminates the channel at value 0 on its
final chunk. Every GVF is therefore a DISCOUNTED PROBABILITY of its event, living in
[0, 1], which is what makes one `lam` scale meaningful across channels and what lets
the heads use a raw linear output (a sigmoid would saturate and kill dQ/da, the whole
reason the heads are here).

`sign` is applied at the ACTOR LOSS ONLY and is never folded into the cumulant, so a
logged head value is always directly readable as "probability of this event".

TERMINAL OWNERSHIP. This module owns stdin while it runs (raw cbreak mode) and
forwards every key outside its own keymap to `on_other_key`. Two listeners on one
terminal drop keypresses nondeterministically, which is a miserable bug to chase
mid-session on hardware; the labeler is the one that owns input because it needs
precise timestamps and must sit closest to the raw event.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import json
import logging
from pathlib import Path
import select
import sys
import threading

import numpy as np

logger = logging.getLogger("rlt.labeling")


# ════════════════════════════════════════════════════════════════════════════
# Part A — channel spec
# ════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class GvfChannel:
    """One auxiliary event channel. Frozen (and therefore hashable) because a tuple
    of these rides in TD3Config, which is a static argument to jitted code."""

    name: str  # "grasp", "coll"
    key: str  # keyboard key that labels this channel (lowercase => +1, shifted => -1)
    gamma: float = 0.95  # channel-specific discount (short chains -> lower gamma)
    lam: float = 0.3  # actor-loss coefficient
    sign: float = +1.0  # +1 => actor MAXIMIZES this GVF, -1 => MINIMIZES
    phases: tuple[int, ...] = ()  # episode phases where the actor term is active

    @property
    def neg_key(self) -> str:
        """Shift + the channel key, i.e. the `-1` (event can no longer occur) label."""
        return self.key.upper()

    @property
    def cumulant_key(self) -> str:
        return f"c_{self.name}"

    @property
    def continuation_key(self) -> str:
        return f"z_{self.name}"


DEFAULT_CHANNELS: tuple[GvfChannel, ...] = (
    GvfChannel("grasp", key="g", gamma=0.95, lam=0.3, sign=+1.0, phases=(0,)),
    GvfChannel("coll", key="c", gamma=0.95, lam=0.3, sign=-1.0, phases=(1,)),
)

PAUSE_KEY = "p"
UNDO_KEY = "u"
PHASE_CHANNEL = "grasp"
"""Back-compat alias for the single-channel case; PHASE_CHANNELS is the real setting."""

PHASE_CHANNELS: tuple[str, ...] = ("grasp",)
"""ORDERED channels whose first label each advance the episode one phase.

An episode starts in phase 0 and moves to phase i+1 at the first label of
PHASE_CHANNELS[i], so N phase channels give phases 0..N. The default ("grasp",)
is the two-phase book task: 0 before the grasp, 1 after.

A multi-stage task overrides it in order, e.g. ("grasp", "place") for hammer:
    phase 0  approaching        -> the `grasp` GVF steers
    phase 1  hammer in hand     -> the `place` GVF steers
    phase 2  hammer in the box  -> the `push` GVF steers
Advancement is MONOTONE: each channel's label counts only at or after the previous
boundary, so an out-of-order press cannot rewind the phase."""


# ════════════════════════════════════════════════════════════════════════════
# Part B — bridge keymap and the collision guard
# ════════════════════════════════════════════════════════════════════════════

BRIDGE_KEYMAP: dict[str, str] = {
    "\n": "episode end — SUCCESS (reward 1.0 + done)",
    "\r": "episode end — SUCCESS (carriage-return form of Enter)",
    "\x7f": "episode end — manual FAILURE (backspace)",
    " ": "pause/resume the robot-side control loop (space)",
}
"""The real-robot bridge's own bindings, transcribed from
`KeyboardInputManager._keyboard_listener` in train_pi05_real.py -- the operator-side
process that owns the arm and reports `reward`/`done` back over the websocket that
`_BridgeSlotProxy.recv_step` (rlt/envs.py) reads.

`Enter` is RESERVED and must not be rebound. It is existing operator muscle memory
and a mis-press is expensive in both directions: a pause-instead-of-success wastes a
hardware episode, and a success-instead-of-pause writes a false reward=1.0 positive
into the buffer -- exactly the failure the `done`/`reward` comment at collect()'s
inner step loop warns about. Labeling pauses on `p` instead.

Note the two keymaps normally live in DIFFERENT processes on different machines (the
bridge's in the robot desktop's terminal, this module's in the trainer's), so there is
no literal stdin contention today. The collision assert below runs anyway: the moment
someone runs both in one terminal, or moves the labeler robot-side, a collision would
be silent and would cost hardware episodes to discover."""


class KeyCollisionError(RuntimeError):
    pass


def assert_no_key_collision(channels: tuple[GvfChannel, ...]) -> None:
    """Abort at startup if any labeling key collides with a bridge-reserved key.

    Also catches labeling keys colliding with each other. If the bridge's failure key
    ever moves next to `g`/`c` on the keyboard, move the CHANNEL keys -- the failure
    key is muscle memory and the channel keys are not.
    """
    # Type-check before touching .key. This is the entry point every caller reaches
    # first, and it has already been handed the wrong thing once: main() binds
    # `channels` to a list of _BridgeChannel websocket handles on the real-robot path,
    # which shadowed the GVF tuple and surfaced here as a bare AttributeError on
    # `.key`. Say what arrived instead.
    wrong = [c for c in channels if not isinstance(c, GvfChannel)]
    if wrong:
        raise TypeError(
            f"expected a tuple of GvfChannel, got {type(wrong[0]).__name__}. "
            f"(train_rlt_libero.main() binds `channels` to _BridgeChannel handles on the "
            f"real-robot path — the GVF tuple is `cfg.parsed_gvf_channels`.)"
        )
    # A LIST, not a dict: two channels claiming the same key must collide, and a dict
    # would silently let the second overwrite the first.
    claims: list[tuple[str, str]] = []
    for ch in channels:
        if len(ch.key) != 1 or not ch.key.isalpha() or not ch.key.islower():
            raise KeyCollisionError(f"channel {ch.name!r}: key must be a single lowercase letter, got {ch.key!r}")
        claims.append((ch.key, f"channel {ch.name} +1"))
        claims.append((ch.neg_key, f"channel {ch.name} -1"))
    claims.append((PAUSE_KEY, "pause/resume"))
    claims.append((UNDO_KEY, "undo last label"))

    claimed: dict[str, str] = {}
    for key, what in claims:
        if key in claimed:
            raise KeyCollisionError(f"labeling key {key!r} is claimed twice: {claimed[key]} and {what}")
        claimed[key] = what

    bridge_desc = "\n".join(f"    {k!r:>8} -> {v}" for k, v in BRIDGE_KEYMAP.items())
    clashes = [
        f"{k!r} ({claimed[k]}) collides with the bridge's {BRIDGE_KEYMAP[k]}" for k in claimed if k in BRIDGE_KEYMAP
    ]
    if clashes:
        raise KeyCollisionError(
            "labeling keymap collides with keys the real-robot bridge has reserved.\n"
            + "\n".join(f"  - {c}" for c in clashes)
            + "\n  bridge-reserved bindings (train_pi05_real.py KeyboardInputManager):\n"
            + bridge_desc
            + "\n  Move the CHANNEL keys (GvfChannel.key), not the bridge's — Enter/backspace/space are\n"
            "  operator muscle memory and a mis-press costs a hardware episode."
        )


def keymap_help(channels: tuple[GvfChannel, ...], lookback_steps: int) -> str:
    lines = [f"  {'Enter':>7}  episode end — success        [bridge, do not rebind]"]
    lines.append(f"  {'Bksp':>7}  episode end — manual fail    [bridge]")
    for ch in channels:
        lines.append(f"  {ch.key:>7}  {ch.name} +1 (event occurred)")
        lines.append(f"  {ch.neg_key:>7}  {ch.name} -1 (can no longer occur)")
    lines.append(f"  {PAUSE_KEY:>7}  pause / resume (rollout halts BEFORE the next action)")
    lines.append(f"  {UNDO_KEY:>7}  undo last label")
    return (
        f"labeling keymap (quick mode records at step t-{lookback_steps}; pause mode records at the shown step):\n"
        + "\n".join(lines)
    )


# ════════════════════════════════════════════════════════════════════════════
# Part B — the listener
# ════════════════════════════════════════════════════════════════════════════


class KeyboardLabeler:
    """Non-blocking raw-mode stdin listener; the sole owner of the terminal.

    Quick mode (default): a channel key records +1 at `current_step - lookback_steps`
    for the focused episode, shifted records -1. No pause, lowest latency -- for
    events the operator can anticipate.

    Pause mode: `p` halts the rollout. The rollout loop must consult `.paused` BEFORE
    it dispatches the next action, so the pause leaves NO trace in the episode: no
    send_action, no recv_step, no transition appended, `steps` does not advance. While
    paused the operator types `g +1` / `c -1` / `u` / `p` (Enter submits); those land
    at the exact displayed step with NO lookback offset, because the operator had time
    to look. Keys are not forwarded while paused -- that is the point of a pause.

    The labeler is a no-op (never touches termios) when stdin is not a TTY, so batch/
    Slurm runs and the LIBERO auto-label path cost nothing and print nothing.
    """

    def __init__(
        self,
        channels: tuple[GvfChannel, ...],
        *,
        lookback_steps: int = 2,
        on_other_key: Callable[[str], None] | None = None,
        stream=None,
    ):
        assert_no_key_collision(channels)
        self.channels = channels
        self.by_key = {ch.key: (ch, +1) for ch in channels} | {ch.neg_key: (ch, -1) for ch in channels}
        self.lookback_steps = int(lookback_steps)
        self._on_other_key = on_other_key
        self._out = stream if stream is not None else sys.stdout

        self.paused = False
        self._pending: list[tuple[str, int, int]] = []  # (channel name, +-1, resolved step)
        self._undo = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._line = ""  # pause-mode command accumulator
        self._focus_step = 0  # last step the rollout reported, for the pause prompt

    # ---- lifecycle -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        try:
            return bool(self.channels) and sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, AttributeError):  # detached/closed stdin
            return False

    def start(self) -> KeyboardLabeler:
        if not self.enabled or self._running:
            if self.channels and not self.enabled:
                logger.info("stdin is not a TTY — keyboard labeling disabled (use --auto_label or sidecars)")
            return self
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True, name="rlt-labeler")
        self._thread.start()
        print(keymap_help(self.channels, self.lookback_steps), file=self._out, flush=True)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> KeyboardLabeler:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- rollout-side API ------------------------------------------------
    def note_step(self, step: int) -> None:
        """Tell the labeler where the focused episode is, for the pause prompt."""
        self._focus_step = int(step)

    def drain(self) -> list[tuple[str, int, int]]:
        """Pop the (channel, value, step) labels pressed since the last call.

        The step is resolved AT PRESS TIME, not here: a quick-mode press carries the
        latency correction (`step - lookback_steps`, floored at 0) against the step
        the rollout had last reported via `note_step`, while a pause-mode entry lands
        at the displayed step exactly. Resolving here instead would misdate every
        label by however far the rollout advanced before the drain.
        """
        with self._lock:
            pending, self._pending = self._pending, []
        return pending

    def pop_undo(self) -> bool:
        """True once per `u` press."""
        if self._undo.is_set():
            self._undo.clear()
            return True
        return False

    def resolve_step(self, step: int) -> int:
        return max(0, int(step) - self.lookback_steps)

    # ---- listener internals ---------------------------------------------
    def _emit(self, name: str, value: int, step: int) -> None:
        with self._lock:
            self._pending.append((name, value, max(0, int(step))))

    def _listen(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                key = sys.stdin.read(1)
                if not key:
                    continue
                if self.paused:
                    self._handle_paused_key(key)
                else:
                    self._handle_quick_key(key)
        except Exception:  # a dead listener must not kill the rollout
            logger.exception("keyboard labeler died; labeling is off for the rest of this run")
        finally:
            with contextlib.suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _handle_quick_key(self, key: str) -> None:
        if key == PAUSE_KEY:
            self.paused = True
            self._line = ""
            self._prompt()
            return
        if key == UNDO_KEY:
            self._undo.set()
            print("[label] undo requested", file=self._out, flush=True)
            return
        if key in self.by_key:
            ch, value = self.by_key[key]
            self._emit(ch.name, value, self.resolve_step(self._focus_step))
            print(
                f"[label] {ch.name} {value:+d} @ step ~{self.resolve_step(self._focus_step)} "
                f"(pressed at {self._focus_step}, lookback {self.lookback_steps})",
                file=self._out,
                flush=True,
            )
            return
        if self._on_other_key is not None:
            self._on_other_key(key)  # forwarded unmodified to the bridge's handler

    def _prompt(self) -> None:
        names = ", ".join(f"{ch.key}={ch.name}" for ch in self.channels)
        print(
            f"\n[label] PAUSED at step {self._focus_step} (rollout halted BEFORE the next action).\n"
            f"[label]   channels: {names} | type e.g. 'g +1' or 'c -1', "
            f"'{UNDO_KEY}' to undo, '{PAUSE_KEY}' to resume. Enter submits.\n"
            f"[label]   pause-mode labels land at step {self._focus_step} exactly (no lookback).\n"
            "[label] > ",
            end="",
            file=self._out,
            flush=True,
        )

    def _handle_paused_key(self, key: str) -> None:
        if key in ("\n", "\r"):
            line, self._line = self._line.strip(), ""
            print(file=self._out, flush=True)
            if line:
                self._run_pause_command(line)
            if self.paused:
                print("[label] > ", end="", file=self._out, flush=True)
            return
        if key == "\x7f":  # backspace edits the line while paused
            self._line = self._line[:-1]
            return
        self._line += key
        print(key, end="", file=self._out, flush=True)

    def _run_pause_command(self, line: str) -> None:
        parts = line.split()
        head = parts[0]
        if head in (PAUSE_KEY, "resume"):
            self.paused = False
            print(f"[label] resumed at step {self._focus_step}", file=self._out, flush=True)
            return
        if head == UNDO_KEY:
            self._undo.set()
            print("[label] undo requested", file=self._out, flush=True)
            return
        by_name = {ch.key: ch for ch in self.channels} | {ch.name: ch for ch in self.channels}
        if head not in by_name:
            print(f"[label] unknown command {line!r}", file=self._out, flush=True)
            return
        ch = by_name[head]
        value = +1
        if len(parts) > 1:
            tok = parts[1].lstrip("+")
            if tok in ("-1", "-"):
                value = -1
            elif tok not in ("1", ""):
                print(f"[label] unknown value {parts[1]!r} (want +1 or -1)", file=self._out, flush=True)
                return
        # Pause-mode labels are exact: the operator looked before committing.
        self._emit(ch.name, value, self._focus_step)
        print(
            f"[label] {ch.name} {value:+d} @ step {self._focus_step} (exact, no lookback)", file=self._out, flush=True
        )


# ════════════════════════════════════════════════════════════════════════════
# Part B — sidecar persistence
# ════════════════════════════════════════════════════════════════════════════


def sidecar_path(root: Path | str, run_id: str, ep_id: int) -> Path:
    return Path(root) / "labels" / str(run_id) / f"{int(ep_id)}.json"


def save_episode_labels(
    path: Path | str,
    *,
    ep_id: int,
    ep_len: int,
    label_lookback_steps: int,
    events: dict[str, list[tuple[int, int]]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ep_id": int(ep_id),
        "ep_len": int(ep_len),
        "label_lookback_steps": int(label_lookback_steps),
        "events": {name: [[int(s), int(v)] for s, v in evs] for name, evs in events.items()},
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def next_ep_id(root: Path | str, run_id: str) -> int:
    """One past the highest ep_id already written under labels/<run_id>/.

    Episode ids are RUN-GLOBAL and name the sidecars, so a resumed run must continue
    the sequence rather than restart at 0. Restarting would make the resumed run's
    first episode load episode 0's old sidecar and adopt its labels wholesale --
    timestamps from a different episode entirely, silently, with no error.
    """
    d = Path(root) / "labels" / str(run_id)
    if not d.is_dir():
        return 0
    ids = [int(f.stem) for f in d.glob("*.json") if f.stem.lstrip("-").isdigit()]
    return max(ids) + 1 if ids else 0


def load_episode_labels(path: Path | str) -> dict | None:
    """Returns the sidecar payload with `events` normalized to sorted (step, value)
    tuple lists, or None when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    payload["events"] = {
        name: sorted((int(s), int(v)) for s, v in evs) for name, evs in payload.get("events", {}).items()
    }
    return payload


# ════════════════════════════════════════════════════════════════════════════
# Part C — phases
# ════════════════════════════════════════════════════════════════════════════


def derive_phases(
    events: dict[str, list[tuple[int, int]]],
    ep_len: int,
    *,
    phase_channels: tuple[str, ...] = PHASE_CHANNELS,
) -> np.ndarray:
    """Per-step phase array of length `ep_len`, advancing one phase at each of
    `phase_channels`' first label, in order (see PHASE_CHANNELS).

    An unlabeled channel STOPS the progression: the episode stays at whatever phase it
    had reached. That is deliberate -- if the operator never marked the place, the
    hammer was never placed, so the push GVF has no business steering afterwards.
    """
    phase = np.zeros(max(int(ep_len), 0), dtype=np.int32)
    if phase.size == 0:
        return phase
    cut = 0
    for i, name in enumerate(phase_channels):
        # Only labels at or after the previous boundary count, so an out-of-order
        # press cannot move the phase backwards or open a gap.
        later = [int(step) for step, _ in (events.get(name) or []) if int(step) >= cut]
        if not later:
            break
        first = min(later)
        phase[min(max(first, 0), phase.size) :] = i + 1
        cut = first
    return phase


def check_success_consistency(
    events: dict[str, list[tuple[int, int]]],
    *,
    success: bool,
    ep_id,
    phase_channels: tuple[str, ...] = PHASE_CHANNELS,
    source: str = "human",
) -> str | None:
    """Warn (never auto-insert) when a SUCCESSFUL episode carries no +1 on a channel
    the phase progression depends on: the timestep would be unknown, and a fabricated
    timestamp is worse than a missing label. Returns the warning text, or None.

    `phase_channels` defaults to the module constant but MUST be passed the run's
    configured `--phase_channels`. Hardcoding "grasp" made this fire on every success
    of a task that has no grasp in it at all -- libero_goal task 5 is
    push_the_plate_to_the_front_of_the_stove, where a two-fingerpad grasp is not part
    of any solution -- burying real warnings under one per successful episode.

    `source` only shapes the message. Under predicate labeling there is no operator
    and no threshold, so "the operator missed it" sends the reader hunting for a
    labeling bug that does not exist; the honest reading is that the policy reached
    the goal without that event, which is usually a different STRATEGY (a push rather
    than a pick) and is worth knowing as a result, not as a defect.
    """
    if not success:
        return None
    missing = [n for n in phase_channels if not any(v > 0 for _, v in events.get(n) or [])]
    if not missing:
        return None
    if source == "predicate":
        return (
            f"episode {ep_id}: success=True but the BDDL predicate(s) {missing} never fired — "
            f"the policy reached the goal without that event (e.g. pushing an object rather than "
            f"grasping it). Not auto-inserting. NOTE the phase never advances on such an episode, "
            f"so every later-phase channel stays gated off for all of it."
        )
    return (
        f"episode {ep_id}: success=True but {missing} has no +1 label — "
        f"the operator missed it (or the auto-labeler's threshold is off). Not auto-inserting: "
        f"a fabricated timestep is worse than a missing one."
    )


# ════════════════════════════════════════════════════════════════════════════
# Part D — chunk-level cumulants
# ════════════════════════════════════════════════════════════════════════════


def chunk_cumulant(
    channel_events: list[tuple[int, int]],
    *,
    t: int,
    n: int,
    ep_len: int,
    gamma: float,
) -> tuple[float, float]:
    """(cumulant c_k, continuation z_k) for the chunk [t, t+n) of an episode of
    length `ep_len`.

        first labeled step j in [t, t+n), value +1 -> c = gamma**(j - t), z = 0
        first labeled step j in [t, t+n), value -1 -> c = 0,               z = 0
        no label and the chunk reaches the end     -> c = 0,               z = 0
        otherwise                                  -> c = 0,               z = gamma**n

    Both label values TERMINATE the channel; they differ only in the terminal value.
    An episode ending with neither label terminates the channel at 0 on its final
    chunk. Channel discounting is per-channel `gamma` (short chains want a lower
    gamma than the task critic's).
    """
    for step, value in sorted(channel_events):
        if t <= step < t + n:
            return (float(gamma ** (step - t)), 0.0) if value > 0 else (0.0, 0.0)
    if t + n >= ep_len:
        return 0.0, 0.0
    return 0.0, float(gamma**n)


def chunk_cumulants(
    channels: tuple[GvfChannel, ...],
    events: dict[str, list[tuple[int, int]]],
    *,
    t: int,
    n: int,
    ep_len: int,
) -> dict[str, float]:
    """`chunk_cumulant` for every channel, keyed as the batch keys `c_<name>` /
    `z_<name>` that replay.py carries and td3.py reads."""
    out: dict[str, float] = {}
    for ch in channels:
        c, z = chunk_cumulant(events.get(ch.name) or [], t=t, n=n, ep_len=ep_len, gamma=ch.gamma)
        out[ch.cumulant_key] = c
        out[ch.continuation_key] = z
    return out


# ════════════════════════════════════════════════════════════════════════════
# Part B2 — LIBERO subgoal predicates (exact, from the BDDL task definition)
# ════════════════════════════════════════════════════════════════════════════
#
# `auto_label_libero` below infers events from PROPRIO ALONE, because only the
# trimmed obs keys cross the env-subprocess boundary. That forced two proxies -- a
# grasp read as "gripper closed AND the eef rose 2cm", a collision read as "commanded
# but not moving" -- and they are visibly weak: on libero_goal task 3 the proxy grasp
# put only 3-6% of buffer rows past the phase boundary, so a phase-1-gated channel was
# switched off almost everywhere.
#
# LIBERO already knows the answer exactly. Every task is a BDDL problem whose
# `goal_state` is first-order predicates over MuJoCo state, evaluated in-process by
# `env._eval_predicate` against `object_states_dict` (contacts, containment, joint
# angles). Evaluating those per step inside the subprocess and shipping the booleans
# out as one more obs key gives GROUND-TRUTH subgoal labels with no human, no
# thresholds, and no per-suite tuning.
#
# WHAT IS DERIVABLE, AND WHAT IS NOT. The goal conjunction is the SUCCESS condition,
# so a channel on it alone re-encodes the sparse task reward and buys little. The
# subgoals that actually carry new information come from the task's *structure*:
#
#   * `obj_of_interest[0]`, when it is a movable object rather than a fixture or a
#     region site, is the thing being manipulated -- so a contact-based grasp of it is
#     a genuine intermediate event on EVERY episode, successful or not.
#   * the remaining obj_of_interest entries are the destination. When one of them is a
#     site on an ARTICULATED fixture it carries an `is_open` affordance, which is how
#     "open the drawer" becomes a labeled subgoal on
#     open_the_top_drawer_and_put_the_bowl_inside -- a step the goal conjunct
#     `(In bowl top_region)` never mentions.
#   * fixtures that are NOT part of the goal are the things worth not hitting, which
#     is the honest version of the blocked-motion `coll` proxy.
#
# The `open` candidate is emitted for every non-movable object of interest and DROPPED
# AT RUNTIME for the ones with no such affordance (`plate_1` has none), rather than
# being guessed from the name here: derivation stays a pure function of the parsed
# problem and therefore offline-testable, while the affordance question is answered by
# the env that owns it. See rlt/envs.build_subgoal_evaluator.


@dataclasses.dataclass(frozen=True)
class SubgoalPredicate:
    """One per-step boolean the env subprocess evaluates and ships out in `obs`.

    `kind` selects the evaluator (envs.build_subgoal_evaluator); `args` are BDDL
    object names. `name` is also the GVF CHANNEL name, so it must stay stable across
    tasks even though which predicates exist does not -- a run configures a fixed
    channel set and therefore a fixed number of heads.
    """

    name: str
    kind: str  # "grasp" | "contact" | "open" | "goal" | "collide"
    args: tuple[str, ...] = ()


SUBGOAL_SIGNS: dict[str, float] = {"collide": -1.0}
"""Channels the actor MINIMIZES. Everything else is maximized (+1)."""

SUBGOAL_PHASES: dict[str, tuple[int, ...]] = {"grasp": (0,), "contact": (0,), "collide": (1,)}
"""Default phase gating, mirroring DEFAULT_CHANNELS: the manipulation channels steer
before the grasp, the collision channel after it. `open`/`goal` are left ungated
because their useful window is task-dependent."""

SUBGOAL_KEYS: dict[str, str] = {"grasp": "g", "contact": "t", "open": "o", "goal": "l", "collide": "c"}
"""Keyboard keys, so a predicate-labeled channel can still be hand-corrected through
the same sidecar path. Must not collide with BRIDGE_KEYMAP (assert_no_key_collision)."""


def derive_subgoal_predicates(parsed_problem: dict) -> tuple[SubgoalPredicate, ...]:
    """The subgoal predicates implied by one parsed BDDL problem.

    Pure function of `robosuite_parse_problem` output -- no env, no MuJoCo -- so the
    whole mapping is testable offline against the shipped .bddl files.

    Ordering is fixed (contact, grasp, open, goal, collide) so a channel's index in
    the shipped `obs["subgoal"]` vector is stable for a given task.
    """
    # `objects`/`fixtures` map a CATEGORY to its instance names
    # ({"cream_cheese": ["cream_cheese_1"]}), while obj_of_interest and goal_state
    # name INSTANCES -- flatten before comparing or every membership test fails and
    # the grasp channels silently vanish.
    objects = {inst for insts in (parsed_problem.get("objects") or {}).values() for inst in insts}
    fixture_insts = {inst: cat for cat, insts in (parsed_problem.get("fixtures") or {}).items() for inst in insts}
    ooi = list(parsed_problem.get("obj_of_interest") or [])
    goal_state = [list(g) for g in (parsed_problem.get("goal_state") or [])]

    # A movable object of interest is one declared under (:objects ...). Region sites
    # ("wooden_cabinet_1_top_region") and fixtures are not manipulable, so a grasp
    # predicate on them would be permanently false rather than merely rare.
    #
    # Among those, prefer one the GOAL asks to be MOVED: in `(on x y)` / `(in x y)`
    # the first argument ends up somewhere and the second is where. obj_of_interest is
    # NOT ordered manipulated-object-first -- libero_10's two LIVING_ROOM two-mug tasks
    # (benchmark 4 and 6) declare the destination plates under (:objects ...) AND list
    # them first, so taking ooi[0] put contact AND grasp on a plate the gripper never
    # touches. Both heads then trained on an all-zero cumulant and the shaping
    # collapsed to the `goal` channel, which is just a discounted success predictor --
    # redundant with the reward rather than a subgoal (measured: gvf_bonus_frac 0.46
    # where every other task gives ~0.65, and the shaped arm lost to plain RLT).
    # Falls back to ooi order when the goal has no on/in conjunct (e.g. `turnon`), so
    # the other 38 tasks' channel sets are unchanged.
    moved = {g[1] for g in goal_state if len(g) >= 3 and g[0].lower() in ("on", "in")}
    movable = [o for o in ooi if o in objects and o in moved] or [o for o in ooi if o in objects]
    targets = [o for o in ooi if o not in objects]

    out: list[SubgoalPredicate] = []
    if movable:
        out.append(SubgoalPredicate("contact", "contact", (movable[0],)))
        out.append(SubgoalPredicate("grasp", "grasp", (movable[0],)))
    if targets:
        # The destination's ENABLING state: drawer/microwave open, stove on. One
        # channel covers both affordances because a task has at most one such step and
        # the head count is fixed at config time -- the evaluator probes `is_open`
        # first, then `turn_on`. Emitted as a CANDIDATE for the first non-movable
        # target and dropped at runtime when that object has neither (a plate has
        # neither, a cabinet drawer region has is_open) -- see the module note above on
        # why the affordance question is not answered here.
        out.append(SubgoalPredicate("open", "open", (targets[0],)))
    if goal_state:
        out.append(SubgoalPredicate("goal", "goal", ()))
    # Fixture INSTANCES the task never asks the arm to touch. Tables are excluded: the
    # objects rest on one, so gripper-table contact is routine rather than a fault.
    # Matched on the substring, not equality -- the work surface is `table` in
    # libero_goal but `kitchen_table` / `living_room_table` / `study_table` in
    # libero_10, and an equality test made every libero_10 task's collide channel fire
    # on the bench the arm works over.
    goal_names = {a for g in goal_state for a in g[1:]} | set(ooi)
    hazards = tuple(
        inst
        for inst, cat in sorted(fixture_insts.items())
        if "table" not in cat and not any(n.startswith(inst) for n in goal_names)
    )
    if hazards:
        out.append(SubgoalPredicate("collide", "collide", hazards))
    return tuple(out)


def subgoal_channels(
    names: tuple[str, ...],
    *,
    gamma: float = 0.95,
    lam: float = 0.3,
) -> tuple[GvfChannel, ...]:
    """GvfChannels for a set of discovered predicate names, in the given order.

    The sign/phase/key conventions live in the SUBGOAL_* tables above so the CLI's
    `--gvf_channels auto` and a hand-written spec cannot disagree about what `collide`
    means.
    """
    return tuple(
        GvfChannel(
            n,
            key=SUBGOAL_KEYS.get(n, n[0]),
            gamma=gamma,
            lam=lam,
            sign=SUBGOAL_SIGNS.get(n, +1.0),
            phases=SUBGOAL_PHASES.get(n, ()),
        )
        for n in names
    )


def auto_label_libero_predicates(
    rec,
    channels: tuple[GvfChannel, ...],
    subgoal_names: tuple[str, ...],
) -> dict[str, list[tuple[int, int]]]:
    """Events from the per-step predicate trace the env shipped in `obs["subgoal"]`.

    Produces exactly the structure the keyboard path produces, so phases, cumulants,
    sidecars and `--relabel_only` are identical on both paths.

    THE LABEL IS THE FIRST RISING EDGE, not "is true at step t". A GVF channel's event
    semantics are "the event OCCURRED here", and `chunk_cumulant` terminates the
    channel at the first label -- so a predicate that is true for 50 consecutive steps
    is ONE event at the first of them. Predicates true at t=0 (before the policy did
    anything -- an already-open drawer, a resting contact) are deliberately NOT
    labeled: they describe the initial condition, not an achievement, and labeling
    them would hand every episode a free event at step 0 and make the channel
    constant.

    `obs_seq` holds L+1 observations for L executed steps, so index t+1 is the state
    AFTER action t -- the step the event is attributed to.
    """
    events: dict[str, list[tuple[int, int]]] = {ch.name: [] for ch in channels}
    obs_seq = rec.obs_seq
    ep_len = len(rec.act_seq)
    if ep_len == 0 or not obs_seq:
        return events
    idx = {n: i for i, n in enumerate(subgoal_names)}

    def _vec(i: int):
        o = obs_seq[i] if i < len(obs_seq) else obs_seq[-1]
        v = o.get("subgoal") if isinstance(o, dict) else None
        return None if v is None else np.asarray(v, np.float32)

    v0 = _vec(0)
    for ch in channels:
        k = idx.get(ch.name)
        if k is None or v0 is None or k >= v0.size:
            continue
        # Latched at the initial state => an initial condition, not an achievement.
        was = bool(v0[k] > 0.5)
        for t in range(ep_len):
            v = _vec(t + 1)
            if v is None or k >= v.size:
                break
            now = bool(v[k] > 0.5)
            if now and not was:
                events[ch.name] = [(t, +1)]
                break
            was = now
    return events


def batch_field_names(channels: tuple[GvfChannel, ...]) -> list[str]:
    """The extra transition columns the channels imply (see replay.FIELD_DTYPES)."""
    if not channels:
        return []
    # `next_phase` rides alongside `phase` because gvf_mode="lookahead" weights its
    # next-state potential Phi(x', a') by the phase of the state x' is, not of x.
    return ["phase", "next_phase", *(k for ch in channels for k in (ch.cumulant_key, ch.continuation_key))]


# ════════════════════════════════════════════════════════════════════════════
# Part B — LIBERO auto-labeling
# ════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class AutoLabelParams:
    """Thresholds for `auto_label_libero`. Defaults are LIBERO/franka scale."""

    gripper_closed: float = 0.05
    """Finger gap (sum of |robot0_gripper_qpos|) below which the gripper counts as
    closed on something."""
    lift_z: float = 0.02
    """Height above the episode's initial eef z that counts as "lifted"."""
    hold_steps: int = 3
    """Consecutive steps the closed+lifted condition must hold before it fires, so a
    single noisy frame cannot manufacture a grasp."""
    blocked_cmd: float = 0.15
    """Commanded translation magnitude (normalized action units) above which the arm
    is genuinely being driven."""
    blocked_move: float = 0.002
    """Realized eef displacement (metres/step) below which that drive produced no
    motion, i.e. the arm is pressing into something."""
    blocked_steps: int = 4
    """Consecutive blocked steps before `coll` fires."""


AUTO_LABEL_DEFAULTS = AutoLabelParams()


def auto_label_libero(
    rec,
    channels: tuple[GvfChannel, ...] = DEFAULT_CHANNELS,
    params: AutoLabelParams = AUTO_LABEL_DEFAULTS,
) -> dict[str, list[tuple[int, int]]]:
    """Derive the human event structure from sim state, for testing the whole
    pipeline without a human in the loop.

    Produces exactly the structure the keyboard path produces -- `{name: [(step,
    +-1), ...]}` with at most one entry per channel -- so everything downstream
    (phases, cumulants, sidecars, `--relabel_only`) is identical on both paths.

    `rec` is a `_EpisodeRecord` (train_rlt_libero.py): `.obs_seq` (L+1 trimmed obs),
    `.act_seq` (L executed normalized actions), `.success`.

    TWO SUBSTITUTIONS, both forced by what actually crosses the process boundary. The
    LIBERO envs run in subprocesses and hand back only the trimmed obs keys
    (`_OBS_KEYS`): no object pose and no `sim.data` contact array. So:
      * `grasp` uses "gripper closed AND the eef has risen `lift_z` above its own
        episode start", held `hold_steps` frames -- the object-z test with the eef as
        its proxy, which is exact whenever the gripper is closed on the object.
      * `coll` uses a BLOCKED-MOTION detector -- commanded translation large while
        realized eef displacement is ~0, held `blocked_steps` frames. That is what a
        geom contact with an immovable holder looks like from proprio alone.
    Both are proxies for the sim signals a MuJoCo-side labeler would read directly;
    they are here to exercise the pipeline, not to be ground truth.
    """
    events: dict[str, list[tuple[int, int]]] = {ch.name: [] for ch in channels}
    obs_seq, act_seq = rec.obs_seq, rec.act_seq
    ep_len = len(act_seq)
    if ep_len == 0 or not obs_seq:
        return events

    def _eef(i: int) -> np.ndarray | None:
        o = obs_seq[i] if i < len(obs_seq) else None
        if o is None or "robot0_eef_pos" not in o:
            return None
        return np.asarray(o["robot0_eef_pos"], np.float64)

    names = {ch.name for ch in channels}

    if "grasp" in names:
        z0 = _eef(0)
        run = 0
        for t in range(ep_len):
            o = obs_seq[t + 1] if t + 1 < len(obs_seq) else obs_seq[-1]
            qpos = np.asarray(o.get("robot0_gripper_qpos", np.zeros(2)), np.float64)
            eef = _eef(t + 1)
            closed = float(np.sum(np.abs(qpos))) < params.gripper_closed
            lifted = z0 is not None and eef is not None and (eef[2] - z0[2]) > params.lift_z
            run = run + 1 if (closed and lifted) else 0
            if run >= params.hold_steps:
                events["grasp"] = [(t - params.hold_steps + 1, +1)]
                break

    if "coll" in names:
        run = 0
        for t in range(ep_len):
            a = np.asarray(act_seq[t], np.float64)
            p0, p1 = _eef(t), _eef(t + 1)
            if p0 is None or p1 is None:
                break
            driving = float(np.linalg.norm(a[:3])) > params.blocked_cmd
            stuck = float(np.linalg.norm(p1 - p0)) < params.blocked_move
            run = run + 1 if (driving and stuck) else 0
            if run >= params.blocked_steps:
                events["coll"] = [(t - params.blocked_steps + 1, +1)]
                break

    return events
