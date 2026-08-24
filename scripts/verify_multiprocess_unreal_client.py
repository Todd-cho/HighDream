"""Smoke-test Windows spawn and clean shutdown of isolated UDP transport."""

from __future__ import annotations

from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path[:0] = [str(ROOT), str(SRC)]

from dogfight.unreal.client import MultiprocessUnrealAIPilotUDPClient
from dogfight.unreal.protocol import CMD


class DummyPolicy:
    def reset(self, context) -> None:
        pass

    def compute_command(self, context) -> CMD:
        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=0.0,
            pitch_cmd=0.0,
            yaw_cmd=0.0,
            throttle_cmd=1.0,
        )


def main() -> None:
    client = MultiprocessUnrealAIPilotUDPClient(
        command_policy=DummyPolicy(),
        server_ip="127.0.0.1",
        server_port=59999,
        team_name="mp-smoke",
        recv_timeout_sec=0.05,
    )
    # Windows spawn imports the project in the child before entering the UDP
    # loop; allow that startup to complete before testing cooperative stop.
    threading.Timer(3.0, client.stop).start()
    client.run()
    if client._process is None or client._process.exitcode != 0:
        raise SystemExit(f"FAIL network_exitcode={client._process.exitcode}")
    print("PASS multiprocess spawn/shutdown")


if __name__ == "__main__":
    main()
