"""Deterministic check for the non-blocking 60 Hz Unreal command path."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dogfight.unreal.client import UnrealAIPilotUDPClient
from dogfight.unreal.protocol import CMD, PlaneInfo, Rotation3D, Vector3D


class SlowPolicy:
    def __init__(self):
        self.calls = 0

    def reset(self, context):
        self.calls = 0

    def compute_command(self, context):
        time.sleep(0.05)  # 20 Hz controller under a 60 Hz packet stream.
        self.calls += 1
        return CMD(context.plane_id, context.frame_index, 0.2, -0.3, 0.0, 0.8)


def plane(frame: int, plane_id: int) -> PlaneInfo:
    return PlaneInfo(
        frame,
        plane_id,
        Vector3D(float(frame), 0.0, 3000.0),
        Rotation3D(0.0, 0.0, 0.0),
        Vector3D(250.0, 0.0, 0.0),
    )


def main() -> None:
    policy = SlowPolicy()
    client = UnrealAIPilotUDPClient(policy, command_delay_sec=0.0)
    client.context.plane_id = 1
    sent: list[CMD] = []
    client.send_command = sent.append  # type: ignore[method-assign]
    client._running = True
    client._start_control_worker()

    start = time.perf_counter()
    for frame in range(1, 61):
        client._handle_plane_info(plane(frame, 1))
        client._handle_plane_info(plane(frame, 2))
        # Duplicate packets must not create duplicate commands.
        if frame % 10 == 0:
            client._handle_plane_info(plane(frame, 1))
            client._handle_plane_info(plane(frame, 2))
    receive_elapsed = time.perf_counter() - start
    time.sleep(0.18)
    client._running = False
    client._control_event.set()
    client._control_thread.join(timeout=1.0)

    assert len(sent) == 60, len(sent)
    assert [cmd.index for cmd in sent] == list(range(1, 61))
    assert receive_elapsed < 0.25, receive_elapsed
    assert policy.calls < 20, policy.calls
    assert client._duplicate_frame_drop_count == 12
    print(
        "PASS",
        f"cmd_frames={len(sent)}",
        f"policy_computes={policy.calls}",
        f"receive_elapsed={receive_elapsed:.4f}s",
        f"duplicate_drops={client._duplicate_frame_drop_count}",
    )


if __name__ == "__main__":
    main()
