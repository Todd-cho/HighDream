from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
import math
import multiprocessing as mp
import queue
import signal
import socket
import struct
import sys
import threading
import time
from typing import Any, Protocol

from dogfight.unreal.protocol import (
    AIType,
    CMD,
    ClientJoinInfo,
    GameControl,
    Init,
    MessageType,
    PlaneInfo,
    SetPlaneID,
    SimulationState,
    pack_client_join_info,
    pack_cmd,
    pack_simulation_state,
    unpack_game_control,
    unpack_init,
    unpack_message_type,
    unpack_plane_info,
    unpack_set_plane_id,
)


class CommandPolicy(Protocol):
    def reset(self, context: "RemoteClientContext") -> None: ...
    def compute_command(self, context: "RemoteClientContext") -> CMD: ...


@dataclass
class PlaneSnapshot:
    is_valid: bool = False
    plane_id: int = -1
    frame_index: int = 0
    plane_info: PlaneInfo | None = None

    def update(self, plane_info: PlaneInfo) -> None:
        self.is_valid = True
        self.plane_id = plane_info.plane_id
        self.frame_index = plane_info.index
        self.plane_info = plane_info


@dataclass
class RemoteClientContext:
    plane_id: int = -1
    frame_index: int = 0
    initial_state: Init | None = None
    own_plane: PlaneSnapshot = field(default_factory=PlaneSnapshot)
    enemy_plane: PlaneSnapshot = field(default_factory=PlaneSnapshot)
    game_control: GameControl | None = None


@dataclass
class PacketTrace:
    direction: str = ""
    message_type: str = ""
    timestamp: float = 0.0
    size_bytes: int = 0
    endpoint: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


class _IPCCommandPolicy:
    """Cheap policy used inside the network process.

    It publishes the newest coherent state to the controller process and
    returns the last completed four-axis command from shared memory.
    """

    def __init__(self, state_queue, reset_queue, shared_action):
        self.state_queue = state_queue
        self.reset_queue = reset_queue
        self.shared_action = shared_action
        self.action_source_updates = 0
        self.action_hold_frames = 0
        self.max_action_age_frames = 0
        self._last_source_frame = -2

    @staticmethod
    def _replace_latest(q, value) -> None:
        try:
            q.put_nowait(value)
            return
        except queue.Full:
            pass
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(value)
        except queue.Full:
            pass

    def reset(self, context: RemoteClientContext) -> None:
        self._replace_latest(self.reset_queue, context)
        with self.shared_action.get_lock():
            self.shared_action[:] = (0.0, 0.0, 0.0, 1.0, -1.0)
        self._last_source_frame = -2

    def compute_command(self, context: RemoteClientContext) -> CMD:
        self._replace_latest(self.state_queue, context)
        with self.shared_action.get_lock():
            action = tuple(self.shared_action[:4])
            source_frame = int(self.shared_action[4])
        if source_frame != self._last_source_frame:
            self.action_source_updates += 1
            self._last_source_frame = source_frame
        else:
            self.action_hold_frames += 1
        if source_frame >= 0:
            self.max_action_age_frames = max(
                self.max_action_age_frames,
                max(0, int(context.frame_index) - source_frame),
            )
        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=action[0],
            pitch_cmd=action[1],
            yaw_cmd=action[2],
            throttle_cmd=action[3],
        )


def _multiprocess_network_entry(
    client_kwargs: dict,
    state_queue,
    reset_queue,
    shared_action,
    stop_event,
) -> None:
    """Windows-spawn-safe network process entry point."""
    # Windows delivers console Ctrl+C to every process sharing the console.
    # Only the parent should translate it into an orderly stop_event; otherwise
    # the child can receive KeyboardInterrupt while joining its own threads and
    # lose both CONTROL_SUMMARY and a clean exit code.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    proxy = _IPCCommandPolicy(state_queue, reset_queue, shared_action)
    client = UnrealAIPilotUDPClient(command_policy=proxy, **client_kwargs)

    def watch_stop() -> None:
        stop_event.wait()
        client._running = False

    watcher = threading.Thread(target=watch_stop, daemon=True)
    watcher.start()
    client.run()


class MultiprocessUnrealAIPilotUDPClient:
    """Keep UDP frame service isolated from CPU-heavy controller work.

    The child process owns the socket and answers every server frame with the
    newest completed action.  The parent process owns the real policy, so a
    slow W97 search cannot hold the child's receive loop through Python's GIL.
    """

    def __init__(self, command_policy: CommandPolicy, **client_kwargs):
        self.command_policy = command_policy
        self.client_kwargs = client_kwargs
        self._ctx = mp.get_context("spawn")
        self._state_queue = self._ctx.Queue(maxsize=1)
        self._reset_queue = self._ctx.Queue(maxsize=1)
        self._shared_action = self._ctx.Array(
            # roll, pitch, yaw, throttle, source_frame. The fifth value is
            # diagnostic only and lets the network process report how stale
            # the held command became when an RL policy ran below 60 Hz.
            "d", (0.0, 0.0, 0.0, 1.0, -1.0), lock=True
        )
        self._stop_event = self._ctx.Event()
        self._process = None
        self._policy_computes = 0
        self._stop_lock = threading.Lock()
        self._stopped = False

    @staticmethod
    def _drain_latest(q):
        latest = None
        while True:
            try:
                latest = q.get_nowait()
            except queue.Empty:
                return latest

    def _reset_policy(self, context: RemoteClientContext) -> None:
        # States queued before Init belong to the previous episode.
        self._drain_latest(self._state_queue)
        self.command_policy.reset(context)
        with self._shared_action.get_lock():
            self._shared_action[:] = (0.0, 0.0, 0.0, 1.0, -1.0)

    def run(self) -> None:
        self._stopped = False
        self._stop_event.clear()
        self._process = self._ctx.Process(
            target=_multiprocess_network_entry,
            args=(
                self.client_kwargs,
                self._state_queue,
                self._reset_queue,
                self._shared_action,
                self._stop_event,
            ),
            name="unreal-udp-network",
        )
        self._process.start()
        print(
            f"[MP_TRANSPORT] network_pid={self._process.pid} "
            f"controller_pid={mp.current_process().pid}"
        )

        try:
            while self._process.is_alive():
                reset_context = self._drain_latest(self._reset_queue)
                if reset_context is not None:
                    self._reset_policy(reset_context)

                try:
                    context = self._state_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                newer_context = self._drain_latest(self._state_queue)
                if newer_context is not None:
                    context = newer_context

                # Do not publish a computation that crossed an episode reset.
                reset_context = self._drain_latest(self._reset_queue)
                if reset_context is not None:
                    self._reset_policy(reset_context)
                    continue

                try:
                    command = self.command_policy.compute_command(context)
                except Exception as exc:
                    print(f"[MP_POLICY_ERROR] {type(exc).__name__}: {exc}")
                    continue

                with self._shared_action.get_lock():
                    self._shared_action[:] = (
                        float(command.roll_cmd),
                        float(command.pitch_cmd),
                        float(command.yaw_cmd),
                        float(command.throttle_cmd),
                        float(context.frame_index),
                    )
                self._policy_computes += 1
        finally:
            self.stop()

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_event.set()
            process = self._process
            # The child performs bounded joins for its control, heartbeat and
            # monitor threads, so allow those to finish before fallback kill.
            if process is not None and process.is_alive():
                process.join(timeout=5.0)
            if process is not None and process.is_alive():
                print("[MP_TRANSPORT] network process did not stop; terminating")
                process.terminate()
                process.join(timeout=1.0)
            exitcode = None if process is None else process.exitcode
            print(
                f"[MP_CONTROL_SUMMARY] policy_computes={self._policy_computes} "
                f"network_exitcode={exitcode}"
            )


class UnrealAIPilotUDPClient:
    def __init__(
        self,
        command_policy: CommandPolicy,
        server_ip: str = "221.151.77.208",
        server_port: int = 9999,
        team_name: str = "ASDF",
        ai_type: AIType = AIType.ReinforcementLearning,
        simulation_state: int = 1,
        heartbeat_interval_sec: float = 1.0,
        command_delay_sec: float = 0.06,
        recv_timeout_sec: float = 0.2,
        enable_terminal_monitor: bool = False,
        terminal_monitor_interval_sec: float = 0.2,
        damage_log_path: str | None = None,
    ):
        self.command_policy = command_policy
        self.server_ip = server_ip
        self.server_port = server_port
        self.team_name = team_name
        self.ai_type = ai_type
        self.simulation_state = simulation_state
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.command_delay_sec = command_delay_sec
        self.recv_timeout_sec = recv_timeout_sec
        self.enable_terminal_monitor = enable_terminal_monitor
        self.terminal_monitor_interval_sec = terminal_monitor_interval_sec

        self._socket: socket.socket | None = None
        self._server_addr: tuple[str, int] | None = None
        self._local_endpoint = ""
        self._udp_mode = "connected"
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._terminal_monitor_thread: threading.Thread | None = None
        # PlaneInfo reception must never wait for an expensive policy.  W97's
        # 3-D min-max search can take tens of milliseconds, while the server
        # expects a CMD on every 60 Hz frame.  The receive thread therefore
        # returns the most recently completed command immediately and a
        # separate worker refreshes that command from the newest state.
        self._control_thread: threading.Thread | None = None
        self._control_event = threading.Event()
        self._policy_lock = threading.Lock()
        self._pending_control_context: RemoteClientContext | None = None
        self._control_generation = 0
        self._cached_control = (0.0, 0.0, 0.0, 1.0)
        self._last_command_frame = -1
        self._last_control_pair_frame = -1
        self._control_compute_count = 0
        self._command_frame_count = 0
        self._duplicate_frame_drop_count = 0
        self._mismatched_pair_wait_count = 0
        self._malformed_packet_count = 0
        self._lock = threading.Lock()
        self.context = RemoteClientContext()
        self._own_info_received = False
        self._enemy_info_received = False
        self._start_time = time.time()
        self._last_received_packet = PacketTrace()
        self._last_sent_packet = PacketTrace()
        self._last_plane_info_by_id: dict[int, PacketTrace] = {}
        self._rx_counts = {message_type.name: 0 for message_type in MessageType}
        self._tx_counts = {message_type.name: 0 for message_type in MessageType}
        # Raw-byte capture for MT_Damage (2026-08-21): no struct/unpack
        # exists for this message type yet (protocol.py only defines the
        # enum value, see MessageType.MT_Damage) -- these packets were
        # silently dropped by _process_packet before (no matching elif
        # branch). Dumps hex bytes + a wall-clock-relative timestamp so a
        # live run can be correlated against visually observed health-bar
        # changes to reverse-engineer the field layout, same approach
        # already used empirically for several PlaneInfo fields.
        self._damage_log_path = damage_log_path
        self._damage_log_file = None
        self._damage_log_writer = None
        if damage_log_path:
            import csv as _csv
            self._damage_log_file = open(damage_log_path, "w", newline="", encoding="utf-8")
            self._damage_log_writer = _csv.writer(self._damage_log_file)
            self._damage_log_writer.writerow(["t_sec", "length", "hex"])
            self._damage_log_file.flush()

    def connect(self) -> None:
        if self._socket is not None:
            return
        server_host = self._normalize_server_host(self.server_ip)
        self._server_addr = (server_host, self.server_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self._is_loopback_host(self.server_ip):
            self._socket.bind(("127.0.0.1", 0))
            self._udp_mode = "local-unconnected"
        else:
            self._socket.connect(self._server_addr)
            self._udp_mode = "connected"
        self._local_endpoint = self._format_endpoint(self._socket.getsockname())
        self._socket.settimeout(self.recv_timeout_sec)

    def run(self) -> None:
        self.connect()
        self._running = True
        self._start_control_worker()
        self._start_terminal_monitor()
        self._start_heartbeat()
        try:
            self._receive_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._control_event.set()
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        if self._terminal_monitor_thread and self._terminal_monitor_thread.is_alive():
            self._terminal_monitor_thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._damage_log_file is not None:
            self._damage_log_file.close()
            self._damage_log_file = None
        print(
            "[CONTROL_SUMMARY] "
            f"cmd_frames={self._command_frame_count} "
            f"policy_computes={self._control_compute_count} "
            f"last_cmd_frame={self._last_command_frame} "
            f"duplicate_drops={self._duplicate_frame_drop_count} "
            # The first PlaneInfo of a normal own/enemy pair necessarily has
            # a different latest frame from the other aircraft. This counter
            # therefore measures partial-pair arrivals, not command delay.
            f"pair_partial_arrivals={self._mismatched_pair_wait_count} "
            f"malformed_packets={self._malformed_packet_count}"
            + (
                " "
                f"action_source_updates={self.command_policy.action_source_updates} "
                f"action_hold_frames={self.command_policy.action_hold_frames} "
                f"max_action_age_frames={self.command_policy.max_action_age_frames}"
                if isinstance(self.command_policy, _IPCCommandPolicy)
                else ""
            )
        )

    def enable_packet_monitor(self, refresh_interval_sec: float | None = None) -> None:
        with self._lock:
            self.enable_terminal_monitor = True
            if refresh_interval_sec is not None:
                self.terminal_monitor_interval_sec = refresh_interval_sec
        if self._running and not (self._terminal_monitor_thread and self._terminal_monitor_thread.is_alive()):
            self._start_terminal_monitor()

    def render_terminal_packet_monitor(self) -> str:
        with self._lock:
            context_copy = RemoteClientContext(
                plane_id=self.context.plane_id,
                frame_index=self.context.frame_index,
                initial_state=copy.deepcopy(self.context.initial_state),
                own_plane=copy.deepcopy(self.context.own_plane),
                enemy_plane=copy.deepcopy(self.context.enemy_plane),
                game_control=copy.deepcopy(self.context.game_control),
            )
            last_rx = copy.deepcopy(self._last_received_packet)
            last_tx = copy.deepcopy(self._last_sent_packet)
            plane_info_by_id = copy.deepcopy(self._last_plane_info_by_id)
            rx_counts = dict(self._rx_counts)
            tx_counts = dict(self._tx_counts)
            control_compute_count = self._control_compute_count
            command_frame_count = self._command_frame_count
            duplicate_frame_drop_count = self._duplicate_frame_drop_count
            mismatched_pair_wait_count = self._mismatched_pair_wait_count
            last_command_frame = self._last_command_frame

        lines = [
            "=== Unreal UDP Packet Monitor ===",
            f"server={self.server_ip}:{self.server_port} team={self.team_name} ai_type={self.ai_type.name}",
            f"udp_mode={self._udp_mode} local={self._local_endpoint}",
            f"running={self._running} uptime_sec={time.time() - self._start_time:.1f} plane_id={context_copy.plane_id} frame_index={context_copy.frame_index}",
            "",
            "[Context]",
            f"own_plane.valid={context_copy.own_plane.is_valid} own_plane.id={context_copy.own_plane.plane_id} own_plane.frame={context_copy.own_plane.frame_index}",
            f"enemy_plane.valid={context_copy.enemy_plane.is_valid} enemy_plane.id={context_copy.enemy_plane.plane_id} enemy_plane.frame={context_copy.enemy_plane.frame_index}",
            f"game_control.command={getattr(context_copy.game_control, 'command', 'n/a')}",
            "",
            "[Counters]",
            "RX: " + self._format_counts(rx_counts),
            "TX: " + self._format_counts(tx_counts),
            (
                "Control: "
                f"cmd_frames={command_frame_count} computes={control_compute_count} "
                f"last_cmd_frame={last_command_frame} "
                f"duplicate_drops={duplicate_frame_drop_count} "
                f"pair_waits={mismatched_pair_wait_count}"
            ),
            "",
            "[PlaneInfo Latest]",
        ]
        lines.extend(self._format_plane_info_summary(plane_info_by_id))
        lines.extend([
            "",
            "[Last RX]",
        ])
        lines.extend(self._format_packet_trace(last_rx))
        lines.extend(["", "[Last TX]"])
        lines.extend(self._format_packet_trace(last_tx))
        return "\n".join(lines)

    def refresh_terminal_packet_monitor(self) -> None:
        snapshot = self.render_terminal_packet_monitor()
        sys.stdout.write("\r\x1b[2J\x1b[H")
        sys.stdout.write(snapshot)
        if not snapshot.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _start_control_worker(self) -> None:
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="unreal-policy-worker",
            daemon=True,
        )
        self._control_thread.start()

    def _control_loop(self) -> None:
        """Compute only the newest available state without blocking UDP RX."""
        while self._running:
            self._control_event.wait(timeout=0.2)
            if not self._running:
                break
            with self._lock:
                context = self._pending_control_context
                generation = self._control_generation
                self._pending_control_context = None
                self._control_event.clear()
            if context is None:
                continue
            try:
                with self._policy_lock:
                    cmd = self.command_policy.compute_command(context)
            except Exception as exc:
                # Preserve the last safe command and keep receiving; the
                # exception remains visible instead of killing UDP service.
                print(f"[CONTROL_WORKER_ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            with self._lock:
                if generation != self._control_generation:
                    continue
                self._cached_control = (
                    float(cmd.roll_cmd),
                    float(cmd.pitch_cmd),
                    float(cmd.yaw_cmd),
                    float(cmd.throttle_cmd),
                )
                self._control_compute_count += 1

    def _start_terminal_monitor(self) -> None:
        if not self.enable_terminal_monitor:
            return
        self._terminal_monitor_thread = threading.Thread(target=self._terminal_monitor_loop, daemon=True)
        self._terminal_monitor_thread.start()

    def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                self.send_simulation_state()
                self.send_client_join_info()
            except (ConnectionResetError, OSError):
                # Same transient ICMP-port-unreachable quirk as
                # _receive_loop's ConnectionResetError handling: don't let
                # one failed send (e.g. server hasn't opened its listener
                # yet) permanently kill the heartbeat thread -- keep
                # retrying every heartbeat_interval_sec until the peer is
                # actually up.
                pass
            time.sleep(self.heartbeat_interval_sec)

    def _terminal_monitor_loop(self) -> None:
        while self._running:
            self.refresh_terminal_packet_monitor()
            time.sleep(self.terminal_monitor_interval_sec)

    def _receive_loop(self) -> None:
        assert self._socket is not None
        while self._running:
            try:
                if self._udp_mode == "local-unconnected":
                    buffer, remote_addr = self._socket.recvfrom(1024)
                    remote_endpoint = self._format_endpoint(remote_addr)
                else:
                    buffer = self._socket.recv(1024)
                    remote_endpoint = self._format_endpoint(self._server_addr)
            except socket.timeout:
                continue
            except ConnectionResetError:
                # Windows UDP quirk: a prior sendto() to a port nobody is
                # listening on yet triggers an ICMP "port unreachable"
                # reply, which Windows then delivers to the NEXT recv()
                # call on this socket as ConnectionResetError -- even
                # though UDP has no real "connection" to reset. This is
                # transient (the peer may not have opened its listener
                # yet, e.g. before the server-side "Start" button is
                # pressed), not fatal, so retry like a timeout instead of
                # silently exiting the whole client.
                time.sleep(self.recv_timeout_sec)
                continue
            except OSError:
                break
            try:
                self._process_packet(buffer, remote_endpoint)
            except (struct.error, ValueError) as exc:
                # A truncated/unknown V1.2 datagram must not kill UDP service
                # for the rest of the round. Count it and continue; genuine
                # policy/code exceptions are intentionally not swallowed.
                self._malformed_packet_count += 1
                print(
                    f"[MALFORMED_PACKET] bytes={len(buffer)} "
                    f"error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    def send_simulation_state(self) -> None:
        assert self._socket is not None
        simulation_state = SimulationState(state=self.simulation_state)
        packet = pack_simulation_state(simulation_state)
        self._send_packet(packet)
        self._record_packet(
            "TX",
            MessageType.MT_SimState,
            simulation_state,
            len(packet),
            self._format_endpoint(self._server_addr),
        )

    def send_client_join_info(self) -> None:
        assert self._socket is not None
        with self._lock:
            plane_id = self.context.plane_id
        join_info = ClientJoinInfo(team_name=self.team_name, ai_type=self.ai_type, plane_id=plane_id)
        packet = pack_client_join_info(join_info)
        self._send_packet(packet)
        self._record_packet(
            "TX",
            MessageType.MT_ClientInfo,
            join_info,
            len(packet),
            self._format_endpoint(self._server_addr),
        )

    def send_command(self, cmd: CMD) -> None:
        assert self._socket is not None
        packet = pack_cmd(cmd)
        self._send_packet(packet)
        self._record_packet(
            "TX",
            MessageType.MT_CMD,
            cmd,
            len(packet),
            self._format_endpoint(self._server_addr),
        )

    def _send_packet(self, packet: bytes) -> None:
        assert self._socket is not None
        assert self._server_addr is not None
        if self._udp_mode == "local-unconnected":
            self._socket.sendto(packet, self._server_addr)
        else:
            self._socket.send(packet)

    def _process_packet(self, buffer: bytes, remote_endpoint: str = "") -> None:
        if len(buffer) < 4:
            return

        try:
            message_type = unpack_message_type(buffer)
        except ValueError:
            # Server sent a message type our MessageType enum doesn't know
            # about (observed at round-end, e.g. a result/game-over packet
            # with no corresponding MT_* value here) -- capture its raw
            # bytes (tagged type=-1) instead of just dropping it silently,
            # since this is a real candidate for where a win/loss/round-over
            # signal could be riding.
            self._log_raw_damage_packet(-1, buffer)
            return
        if message_type == MessageType.MT_Damage:
            self._log_raw_damage_packet(int(message_type), buffer)
            return
        if message_type == MessageType.MT_SetPlaneID:
            packet = unpack_set_plane_id(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_set_plane_id(packet)
        elif message_type == MessageType.MT_Init:
            packet = unpack_init(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_init(packet)
        elif message_type == MessageType.MT_GameControl:
            packet = unpack_game_control(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_game_control(packet)
        elif message_type == MessageType.MT_PlaneInfo:
            packet = unpack_plane_info(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_plane_info(packet)

    def _handle_set_plane_id(self, packet: SetPlaneID) -> None:
        with self._lock:
            self.context.plane_id = packet.plane_id
            self._own_info_received = False
            self._enemy_info_received = False
            reset_context = copy.deepcopy(self.context)
            self._reset_control_cache_locked()
        with self._policy_lock:
            self.command_policy.reset(reset_context)

    def _handle_init(self, packet: Init) -> None:
        with self._lock:
            self.context.initial_state = packet
            self.context.own_plane = PlaneSnapshot()
            self.context.enemy_plane = PlaneSnapshot()
            self._own_info_received = False
            self._enemy_info_received = False
            reset_context = copy.deepcopy(self.context)
            self._reset_control_cache_locked()
        with self._policy_lock:
            self.command_policy.reset(reset_context)

    def _handle_game_control(self, packet: GameControl) -> None:
        with self._lock:
            self.context.game_control = packet

    def _handle_plane_info(self, packet: PlaneInfo) -> None:
        send_cmd: CMD | None = None
        with self._lock:
            if self.context.plane_id != -1 and packet.plane_id == self.context.plane_id:
                if self.context.own_plane.is_valid and packet.index <= self.context.own_plane.frame_index:
                    if packet.index == self.context.own_plane.frame_index:
                        self._duplicate_frame_drop_count += 1
                    return
                self.context.own_plane.update(packet)
                self._own_info_received = True
            else:
                if self.context.enemy_plane.is_valid and packet.index <= self.context.enemy_plane.frame_index:
                    if packet.index == self.context.enemy_plane.frame_index:
                        self._duplicate_frame_drop_count += 1
                    return
                self.context.enemy_plane.update(packet)
                self._enemy_info_received = True

            # A CMD does not need to wait for both PlaneInfo packets. Reply to
            # the first packet observed for each new server frame using the
            # last completed control. Waiting for an exact own/enemy pair made
            # a single delayed UDP packet count as a missed command in V1.2.
            if packet.index > self._last_command_frame:
                cached = self._cached_control
                self._last_command_frame = packet.index
                self._command_frame_count += 1
                send_cmd = CMD(
                    plane_id=self.context.plane_id,
                    index=packet.index,
                    roll_cmd=cached[0],
                    pitch_cmd=cached[1],
                    yaw_cmd=cached[2],
                    throttle_cmd=cached[3],
                )

            # Policy calculation still requires a coherent pair. It may run
            # slower than 60 Hz, but it never controls command delivery.
            if not (self.context.own_plane.is_valid and self.context.enemy_plane.is_valid):
                context_copy = None
            else:
                own_frame = self.context.own_plane.frame_index
                enemy_frame = self.context.enemy_plane.frame_index
                if own_frame != enemy_frame:
                    self._mismatched_pair_wait_count += 1
                    context_copy = None
                elif own_frame <= self._last_control_pair_frame:
                    context_copy = None
                else:
                    self._last_control_pair_frame = own_frame
                    self.context.frame_index = own_frame
                    context_copy = RemoteClientContext(
                        plane_id=self.context.plane_id,
                        frame_index=own_frame,
                        initial_state=copy.deepcopy(self.context.initial_state),
                        own_plane=copy.deepcopy(self.context.own_plane),
                        enemy_plane=copy.deepcopy(self.context.enemy_plane),
                        game_control=copy.deepcopy(self.context.game_control),
                    )

            self._own_info_received = False
            self._enemy_info_received = False
            if context_copy is not None:
                # Replace, don't queue: the controller should always solve the
                # newest geometry and never build latency by processing history.
                self._pending_control_context = context_copy
                self._control_event.set()

        if send_cmd is not None:
            if self.command_delay_sec > 0:
                time.sleep(self.command_delay_sec)
            self.send_command(send_cmd)

    def _reset_control_cache_locked(self) -> None:
        """Reset round-local async state. Caller must hold ``self._lock``."""
        self._control_generation += 1
        self._pending_control_context = None
        self._control_event.clear()
        self._cached_control = (0.0, 0.0, 0.0, 1.0)
        self._last_command_frame = -1
        self._last_control_pair_frame = -1
        self._control_compute_count = 0
        self._command_frame_count = 0
        self._duplicate_frame_drop_count = 0
        self._mismatched_pair_wait_count = 0
        self._malformed_packet_count = 0

    def _log_raw_damage_packet(self, message_type_value: int, buffer: bytes) -> None:
        if self._damage_log_writer is None:
            return
        self._damage_log_writer.writerow([
            f"{time.time() - self._start_time:.3f}",
            len(buffer),
            buffer.hex(),
        ])
        self._damage_log_file.flush()

    def _record_packet(
        self,
        direction: str,
        message_type: MessageType,
        packet: Any,
        size_bytes: int,
        endpoint: str = "",
    ) -> None:
        trace = PacketTrace(
            direction=direction,
            message_type=message_type.name,
            timestamp=time.time(),
            size_bytes=size_bytes,
            endpoint=endpoint,
            fields=self._packet_to_fields(packet),
        )
        with self._lock:
            if direction == "RX":
                self._last_received_packet = trace
                self._rx_counts[message_type.name] += 1
                if message_type == MessageType.MT_PlaneInfo and isinstance(packet, PlaneInfo):
                    self._last_plane_info_by_id[int(packet.plane_id)] = trace
            else:
                self._last_sent_packet = trace
                self._tx_counts[message_type.name] += 1

    def _packet_to_fields(self, packet: Any) -> dict[str, Any]:
        if is_dataclass(packet):
            return self._normalize_value(asdict(packet))
        return {"value": self._normalize_value(packet)}

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._normalize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        if hasattr(value, "name") and hasattr(value, "value"):
            return value.name
        return value

    def _format_counts(self, counts: dict[str, int]) -> str:
        visible_counts = [f"{name}={count}" for name, count in counts.items() if count]
        if not visible_counts:
            return "no packets yet"
        return ", ".join(visible_counts)

    def _format_packet_trace(self, trace: PacketTrace) -> list[str]:
        if not trace.message_type:
            return ["no packet yet"]
        lines = [
            f"type={trace.message_type} size_bytes={trace.size_bytes} age_sec={time.time() - trace.timestamp:.2f}",
        ]
        if trace.endpoint:
            label = "from" if trace.direction == "RX" else "to"
            lines.append(f"{label}={trace.endpoint}")
        lines.extend(self._flatten_mapping(trace.fields))
        return lines

    def _format_plane_info_summary(self, traces: dict[int, PacketTrace]) -> list[str]:
        if not traces:
            return ["plane_id=0 no packet yet", "plane_id=1 no packet yet"]
        plane_ids = [0, 1]
        plane_ids.extend(sorted(plane_id for plane_id in traces if plane_id not in {0, 1}))
        return [
            self._format_plane_info_row(plane_id, traces.get(plane_id))
            for plane_id in plane_ids
        ]

    def _format_plane_info_row(self, plane_id: int, trace: PacketTrace | None) -> str:
        if trace is None or not trace.fields:
            return f"plane_id={plane_id} no packet yet"
        fields = trace.fields
        position = fields.get("position", {})
        rotation = fields.get("rotation", {})
        velocity = fields.get("velocity", {})
        vx = self._as_float(velocity.get("x"))
        vy = self._as_float(velocity.get("y"))
        vz = self._as_float(velocity.get("z"))
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        return (
            f"plane_id={plane_id} age={time.time() - trace.timestamp:.2f}s "
            f"frame={fields.get('index', 'n/a')} "
            f"pos=({self._as_float(position.get('x')):.2f},"
            f"{self._as_float(position.get('y')):.2f},"
            f"{self._as_float(position.get('z')):.2f}) "
            f"rot=({self._as_float(rotation.get('roll')):.1f},"
            f"{self._as_float(rotation.get('pitch')):.1f},"
            f"{self._as_float(rotation.get('yaw')):.1f}) "
            f"vel=({vx:.2f},{vy:.2f},{vz:.2f}) speed={speed:.2f}"
        )

    def _flatten_mapping(self, value: Any, prefix: str = "") -> list[str]:
        if isinstance(value, dict):
            lines: list[str] = []
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                lines.extend(self._flatten_mapping(item, next_prefix))
            return lines
        if isinstance(value, list):
            lines: list[str] = []
            for index, item in enumerate(value):
                next_prefix = f"{prefix}[{index}]"
                lines.extend(self._flatten_mapping(item, next_prefix))
            return lines
        return [f"{prefix}={value}"]

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        value = host.strip().lower()
        return value in {"localhost", "127.0.0.1", "::1"}

    @staticmethod
    def _normalize_server_host(host: str) -> str:
        if UnrealAIPilotUDPClient._is_loopback_host(host):
            return "127.0.0.1"
        return host

    @staticmethod
    def _format_endpoint(endpoint: Any) -> str:
        if not endpoint:
            return ""
        try:
            host, port = endpoint[:2]
        except Exception:
            return str(endpoint)
        return f"{host}:{port}"
