"""Thin relay between the robot desktop's WebsocketClientPolicy (transport
direction unchanged: client sends obs, server responds) and the RLT trainer's
synchronous env interface (proactively calls env.send_step(action) then
env.recv_step(), exactly like EnvWorker / _SubprocEnv in envs.py).

WHY A RELAY AND NOT AN INFERENCE SERVER: train_rlt_libero.collect() already runs
the VLA + actor locally, in-process, with the current weights — same as it does
in sim — and it needs the z_rl / reference / action it produced for the replay
buffer. Running inference inside the websocket handler instead would duplicate
that call site and need its own weight-swap locking. So this file does nothing
but relay raw obs/action traffic, which is what lets RealRobotEnvWorker be a
genuine drop-in for EnvWorker.

THREADING: the websocket server runs its own asyncio loop in a background
thread (`RobotBridgeServer.start()`); the trainer calls into
`_BridgeChannel` synchronously from the main (JAX) trainer thread. A pair of
maxsize=1 queues is the handoff; `asyncio.to_thread` lets the async handler
wait on the sync queue without blocking other connections' event-loop work.

WIRE PROTOCOL (robot desktop's WebsocketClientPolicy, training=True):
  client -> server: {"obs": {...}, "reward": float|None, "done": bool|None}
      reward/done describe the OUTCOME of the previous command; both None
      right after the robot connects (send one such "ready ping" immediately
      on connect, before any real obs exists — see note in RealRobotEnvWorker)
      and both None on the first report after a reset.
  server -> client: {"type": "step", "actions": <single action, action_env_dim>}
      -> robot executes ONE physical action (matches env.send_step's
      granularity exactly — same cadence as LIBERO's per-substep stepping)
      and reports the outcome in its next message
                  or {"type": "reset", "task_id": int, "trial": int}
      -> robot runs its own reset routine, then reports back
      {"obs": <settled obs>, "reward": None, "done": None}
"""

from __future__ import annotations

import asyncio
import http
import logging
import queue
import threading
import traceback

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger("rlt.robot_bridge")


class _BridgeChannel:
    """One physical-robot connection's sync <-> async handoff.

    Only one episode is ever in flight per channel (matches
    RealRobotEnvWorker's group_size=1 assumption from --adv_type gae) — a
    push_command() issued before the previous recv_step() has been consumed
    will simply block on the maxsize=1 queue, which is correct: there is only
    one physical robot behind this channel.
    """

    def __init__(self):
        self._to_robot: queue.Queue = queue.Queue(maxsize=1)
        self._from_robot: queue.Queue = queue.Queue(maxsize=1)
        self.connected = threading.Event()

    # ---- called from the TRAINER thread (synchronous) ----
    def push_command(self, cmd: dict, timeout: float | None = None) -> None:
        self._to_robot.put(cmd, timeout=timeout)

    def pull_obs(self, timeout: float | None = None) -> dict:
        return self._from_robot.get(timeout=timeout)

    # ---- called from the asyncio handler (background thread) ----
    async def pull_command_async(self) -> dict:
        return await asyncio.to_thread(self._to_robot.get)

    async def push_obs_async(self, payload: dict) -> None:
        await asyncio.to_thread(self._from_robot.put, payload)


class RobotBridgeServer:
    """Same transport shape as serve_policy.py's WebsocketPolicyServer, but
    relays instead of running inference — see module docstring."""

    def __init__(
        self, channel: _BridgeChannel, host: str = "0.0.0.0", port: int | None = None, metadata: dict | None = None
    ):
        self._channel = channel
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Run the server in a background thread so the main thread stays
        free for the synchronous JAX training loop (`train_rlt_libero.main`)."""
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
        self._thread.start()

    async def _run(self):
        async with _server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None, process_request=_health_check
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"robot connected from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        self._channel.connected.set()
        try:
            while True:
                try:
                    msg = msgpack_numpy.unpackb(await websocket.recv())
                    await self._channel.push_obs_async(msg)
                    cmd = await self._channel.pull_command_async()
                    await websocket.send(packer.pack(cmd))
                except websockets.ConnectionClosed:
                    logger.info(f"robot disconnected ({websocket.remote_address})")
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            self._channel.connected.clear()


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
