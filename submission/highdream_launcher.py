from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import sys

DEFAULT_CONFIG = {
    "server_ip": "127.0.0.1", "server_port": 9999, "team_name": "HighDream",
    "multiprocess_transport": True, "safety_override": True, "action_repeat": 1,
    "log_csv": "logs/HighDream_live.csv",
    "damage_log_csv": "logs/HighDream_damage_raw.csv",
}

def _runtime_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))

def _external_root() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

def _load_config() -> dict:
    config_path = _external_root() / "HighDream_config.json"
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8-sig") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("HighDream_config.json must contain a JSON object")
        config.update(loaded)
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[HighDream] default config created: {config_path}")
    return config

def _external_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _external_root() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def main() -> None:
    config = _load_config()
    bundle = _runtime_root() / "artifacts" / "models" / "highdream" / "altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10"
    from dogfight.ai.pure_torch_sac_provider import PureTorchSACActionProvider
    from dogfight.unreal.client import MultiprocessUnrealAIPilotUDPClient, UnrealAIPilotUDPClient
    from dogfight.unreal.policies import ProviderCommandPolicy, SafetyOverrideCommandPolicy, SafetyOverrideConfig
    from dogfight.unreal.protocol import AIType

    provider = PureTorchSACActionProvider(bundle)
    policy = ProviderCommandPolicy(
        action_provider=provider,
        observation_mode="tactical19",
        action_repeat=int(config.get("action_repeat", 1)),
        log_csv_path=str(_external_path(str(config["log_csv"]))),
    )
    if bool(config.get("safety_override", True)):
        policy = SafetyOverrideCommandPolicy(policy, SafetyOverrideConfig(enabled=True))
    client_cls = MultiprocessUnrealAIPilotUDPClient if bool(config.get("multiprocess_transport", True)) else UnrealAIPilotUDPClient
    print(f"[HighDream] pure v8 | server={config['server_ip']}:{config['server_port']} | transport={'multiprocess' if client_cls is MultiprocessUnrealAIPilotUDPClient else 'threaded'}")
    client = client_cls(
        command_policy=policy,
        server_ip=str(config["server_ip"]), server_port=int(config["server_port"]),
        team_name=str(config["team_name"]), ai_type=AIType.ReinforcementLearning,
        simulation_state=1, heartbeat_interval_sec=1.0, command_delay_sec=0.0,
        recv_timeout_sec=0.2, enable_terminal_monitor=True,
        terminal_monitor_interval_sec=1.0,
        damage_log_path=str(_external_path(str(config["damage_log_csv"]))),
    )
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[HighDream] stopped by Ctrl+C")
    finally:
        provider.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
