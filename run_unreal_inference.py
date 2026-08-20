from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent   # Release/ 루트
SRC = ROOT / "src"
RELEASE_ROOT = ROOT
DEFAULT_BT_DLL = RELEASE_ROOT / "AIP_BASE.dll"
DEFAULT_BT_RULE_XML = RELEASE_ROOT / "Rule_forTraining.xml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.hybrid_action_provider import HybridActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient
from dogfight.unreal.policies import SafetyOverrideCommandPolicy, SafetyOverrideConfig

# python run_unreal_inference.py --mode rl --bundle-dir artifacts\models\team01\v1 --team-name team01
# python run_unreal_inference.py --mode bt --team-name team01

def parse_args():
    parser = argparse.ArgumentParser(description="Run RL/BT/Hybrid inference and communicate with the Unreal AI server over UDP.")
    parser.add_argument("--mode", choices=["rl", "bt", "hybrid"], required=True, help="Inference backend to use.")
    parser.add_argument("--server-ip", default="192.168.10.115", help="Unreal server IP address.")
    parser.add_argument("--server-port", type=int, default=9999, help="Unreal server UDP port.")
    parser.add_argument("--team-name", default="FDSA", help="Client team name sent to the Unreal server.")
    parser.add_argument("--simulation-state", type=int, default=1, help="Heartbeat simulation state value.")
    parser.add_argument("--heartbeat-sec", type=float, default=1.0, help="Heartbeat interval in seconds.")
    parser.add_argument("--command-delay-sec", type=float, default=0.0, help="Delay before replying with CMD after both PlaneInfo packets are ready.")
    parser.add_argument("--recv-timeout-sec", type=float, default=0.2, help="UDP socket receive timeout.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=6,
        help=(
            "Number of completed own/enemy PlaneInfo pairs to hold each action. "
            "Use 6 to match Release training step_ratio=6; use 1 for per-packet policy calls."
        ),
    )
    parser.add_argument(
        "--debug-action-repeat",
        action="store_true",
        help="Print action-repeat counter, frame indices, update/hold state, and action values.",
    )
    parser.add_argument(
        "--debug-raw-state-frames",
        type=int,
        default=0,
        help=(
            "Print raw PlaneInfo position/rotation/velocity for own+enemy "
            "for the first N compute_command calls, unmodified by any "
            "coordinate/sign conversion. Use to empirically verify Unreal's "
            "position.x/y/z axis convention (N/E/D? units m or cm?) against "
            "a known start geometry (e.g. ownship north, enemy south)."
        ),
    )
    parser.add_argument(
        "--damage-log-csv",
        default=None,
        help=(
            "Write raw hex bytes for every MT_Damage packet (and any "
            "unrecognized message type, tagged type=-1) to this CSV path. "
            "No struct/unpack exists for MT_Damage yet -- this is for "
            "reverse-engineering the field layout by correlating against "
            "visually observed health-bar changes during a live run."
        ),
    )
    parser.add_argument(
        "--tactical-wrapper",
        action="store_true",
        help=(
            "Wrap the RL policy (--mode rl only) with TacticalWrapperActionProvider "
            "(src/dogfight/ai/tactical_wrapper.py) -- a non-learned "
            "Neutral/Offensive/Overshoot state machine that blends the "
            "policy with a rule-based lead-pursuit controller at high ATA. "
            "Off by default -- the model runs raw otherwise."
        ),
    )
    parser.add_argument(
        "--action-rate-limit",
        type=float,
        default=None,
        help=(
            "Max |delta| per policy decision on the ownship's own raw "
            "roll/pitch/yaw command (throttle unrestricted). Must match "
            "whatever env_config.action_rate_limit the loaded checkpoint was "
            "trained with -- see single_agent_env.py's "
            "_apply_action_rate_limit. None (default) disables."
        ),
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help=(
            "Write one CSV row per policy decision (position/rotation/speed "
            "for both aircraft, distance, ATA, AA, and the raw action "
            "roll/pitch/yaw/throttle commands) to this path. Added "
            "2026-08-20 to diagnose live-behavior issues (this env can't "
            "watch a screen recording) -- e.g. python run_unreal_inference.py "
            "... --log-csv artifacts/logs/live_run1.csv"
        ),
    )
    parser.add_argument("--packet-monitor", action="store_true", help="Render live RX/TX packet values in the terminal.")
    parser.add_argument("--packet-monitor-interval-sec", type=float, default=0.2, help="Refresh interval for the live packet monitor.")
    parser.add_argument("--observation-mode", default="tactical16", choices=["classic12", "relative14", "tactical16", "tactical19", "custom"], help="Observation mode for RL inference.")
    parser.add_argument("--observation-module", default="", help="Optional custom observation module.")
    parser.add_argument("--ownship-force-side", type=int, default=1, help="Force side to use for the ownship in BT inference.")
    parser.add_argument("--target-force-side", type=int, default=2, help="Force side to use for the enemy in BT inference.")
    parser.add_argument(
        "--bt-dll",
        default=str(DEFAULT_BT_DLL),
        help="Behavior tree DLL path for BT inference.",
    )
    parser.add_argument(
        "--bt-rule-xml",
        default=str(DEFAULT_BT_RULE_XML),
        help=(
            "Rule XML source to activate while the client runs. "
            "Use Rule_forTraining.xml by default, or pass a team file such as "
            "Rule_team01.xml."
        ),
    )
    parser.add_argument("--bundle-dir", help="Lightweight RL bundle directory created by train_rllib.py.")
    parser.add_argument("--policy-id", default="default_policy", help="RLlib policy id to load from the lightweight bundle.")
    parser.add_argument("--explore", action="store_true", help="Enable stochastic action sampling for RL inference.")
    parser.add_argument("--hybrid-mode", choices=["residual", "blend", "switch"], default="residual", help="Hybrid action composition strategy.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend weight for hybrid blend mode.")
    parser.add_argument("--residual-scale", type=float, default=0.35, help="Residual scaling factor for hybrid residual mode.")
    parser.add_argument(
        "--ai-type",
        choices=["rule", "rl", "sl", "fusion", "etc"],
        default="rl",
        help="AI type announced to the Unreal server in ClientJoinInfo.",
    )
    parser.add_argument(
        "--safety-override",
        action="store_true",
        help=(
            "Wrap the command policy with the hard, non-learned dive-recovery "
            "override verified in training/frozen-eval (single_agent_env.py's "
            "_apply_safety_override, ported to the live Unreal UDP path "
            "2026-08-19). Off by default -- the model runs raw otherwise, "
            "same as every prior live-Unreal test this session."
        ),
    )
    parser.add_argument("--safety-override-altitude-m", type=float, default=500.0)
    parser.add_argument("--safety-override-pitch-deg", type=float, default=-5.0)
    parser.add_argument("--safety-override-roll-deg", type=float, default=45.0)
    parser.add_argument("--safety-override-time-horizon-s", type=float, default=15.0)
    parser.add_argument("--safety-override-hard-floor-m", type=float, default=400.0)
    parser.add_argument(
        "--safety-override-min-altitude-m",
        type=float,
        default=304.8,
        help="Actual crash floor used for the predictive time-to-impact estimate (competition rule: 1000ft).",
    )
    return parser.parse_args()


def build_action_provider(args):
    if args.mode == "bt":
        return BTActionProvider(dll_name=args.bt_dll)

    if args.bundle_dir is None:
        raise ValueError("--bundle-dir is required for rl and hybrid modes")

    rl_provider = RLActionProvider(
        bundle_dir=args.bundle_dir,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=args.explore,
    )

    if args.mode == "rl":
        if args.tactical_wrapper:
            from dogfight.ai.tactical_wrapper import TacticalWrapperActionProvider, TacticalWrapperConfig
            return TacticalWrapperActionProvider(rl_provider, TacticalWrapperConfig())
        return rl_provider

    bt_provider = BTActionProvider(dll_name=args.bt_dll)
    return HybridActionProvider(
        primary_provider=rl_provider,
        secondary_provider=bt_provider,
        mode=args.hybrid_mode,
        alpha=args.alpha,
        residual_scale=args.residual_scale,
    )


def parse_ai_type(value: str) -> AIType:
    mapping = {
        "rule": AIType.RuleBased,
        "rl": AIType.ReinforcementLearning,
        "sl": AIType.SupervisedLearning,
        "fusion": AIType.Fusion,
        "etc": AIType.etc,
    }
    return mapping[value]


def main():
    args = parse_args()
    observation_hook = load_observation_hook(args.observation_module) if args.observation_module else None
    with activate_rule_xml(args.bt_rule_xml, ROOT):
        action_provider = build_action_provider(args)
        command_policy = ProviderCommandPolicy(
            action_provider=action_provider,
            observation_mode=observation_hook["mode"] if observation_hook else args.observation_mode,
            observation_fn=observation_hook["build_observation"] if observation_hook else None,
            ownship_force_side=args.ownship_force_side,
            target_force_side=args.target_force_side,
            action_repeat=args.action_repeat,
            debug_action_repeat=args.debug_action_repeat,
            debug_raw_state_frames=args.debug_raw_state_frames,
            log_csv_path=args.log_csv,
            action_rate_limit=args.action_rate_limit,
        )
        if args.safety_override:
            command_policy = SafetyOverrideCommandPolicy(
                inner=command_policy,
                config=SafetyOverrideConfig(
                    enabled=True,
                    altitude_m=args.safety_override_altitude_m,
                    pitch_deg=args.safety_override_pitch_deg,
                    roll_deg=args.safety_override_roll_deg,
                    time_horizon_s=args.safety_override_time_horizon_s,
                    hard_floor_m=args.safety_override_hard_floor_m,
                    min_altitude_m=args.safety_override_min_altitude_m,
                ),
            )
            print(
                f"[safety-override] enabled: altitude_m={args.safety_override_altitude_m} "
                f"pitch_deg={args.safety_override_pitch_deg} roll_deg={args.safety_override_roll_deg} "
                f"time_horizon_s={args.safety_override_time_horizon_s} "
                f"hard_floor_m={args.safety_override_hard_floor_m} "
                f"min_altitude_m={args.safety_override_min_altitude_m}"
            )
        client = UnrealAIPilotUDPClient(
            command_policy=command_policy,
            server_ip=args.server_ip,
            server_port=args.server_port,
            team_name=args.team_name,
            ai_type=parse_ai_type(args.ai_type),
            simulation_state=args.simulation_state,
            heartbeat_interval_sec=args.heartbeat_sec,
            command_delay_sec=args.command_delay_sec,
            recv_timeout_sec=args.recv_timeout_sec,
            enable_terminal_monitor=args.packet_monitor,
            terminal_monitor_interval_sec=args.packet_monitor_interval_sec,
            damage_log_path=args.damage_log_csv,
        )

        try:
            client.run()
        finally:
            action_provider.close()


if __name__ == "__main__":
    main()
