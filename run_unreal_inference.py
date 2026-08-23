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
    parser.add_argument("--mode", choices=["rl", "bt", "hybrid", "pulse", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27", "w28", "w29", "w30", "w31", "w32", "w33", "w34", "w35", "w36", "w37", "w38", "w39", "w40", "w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56", "w56rl", "w57", "w58", "w59", "w60", "w61", "w62"], required=True, help="Inference backend to use.")
    parser.add_argument(
        "--pulse-sequence",
        choices=["roll", "pitch", "pitch_trim", "roll_inertia", "yaw"],
        default="roll",
        help=(
            "--mode pulse only: which fixed-command step sequence to send "
            "(open-loop, no RL/geometry) for live system identification -- "
            "see src/dogfight/ai/pulse_test_provider.py. The resulting "
            "--log-csv own_roll_deg/own_pitch_deg trace IS the measured "
            "step response."
        ),
    )
    parser.add_argument(
        "--w1-bank-step-test",
        action="store_true",
        help=(
            "--mode w1/w2/w3 only: isolate the roll rate-cascade from pursuit "
            "geometry entirely -- target_bank follows a scripted "
            "+30/0/-30/0deg step sequence (5s each), pitch just holds "
            "flat (gamma_error=0), no target/enemy involved. Validate this "
            "settles at each target without overshoot BEFORE trusting W1 "
            "against a live 91deg pursuit -- see Desktop/전술.txt."
        ),
    )
    parser.add_argument(
        "--pulse-loop",
        action="store_true",
        help="--mode pulse only: repeat the sequence forever instead of holding the last step.",
    )
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
        "--pursuit-controller",
        action="store_true",
        help=(
            "Use the stable bank-to-turn pursuit controller for RL mode. "
            "Mutually exclusive with --tactical-wrapper; intended for live A/B testing."
        ),
    )
    parser.add_argument(
        "--pc-world-frame-pitch-gate",
        action="store_true",
        help=(
            "PursuitControllerConfig.world_frame_pitch_gate=True (P4, "
            "2026-08-21): blocks further nose-up pull unless the target is "
            "actually above ownship in world-frame altitude, regardless of "
            "bank -- fixes a sustained shallow climb that bled distance for "
            "the whole engagement in JSBSim ablation testing. Only applies "
            "with --pursuit-controller. Superseded by --pc-turn-pull-decomposition "
            "(P5) -- kept for A/B reference."
        ),
    )
    parser.add_argument(
        "--pc-turn-pull-decomposition",
        action="store_true",
        help=(
            "PursuitControllerConfig.turn_pull_decomposition=True (P5, "
            "2026-08-21, user diagnosis of P4's live failure at 91deg): "
            "replaces P4's hard climb-pull clamp with an additive "
            "coordinated-turn pull (scales with bank, present regardless of "
            "target position) + a small world-frame altitude correction + "
            "vertical-rate damping. P4's hard clamp zeroed the "
            "coordinated-turn back-pressure a banked turn needs just to "
            "hold altitude and actually turn, not just the excess climb "
            "pull. Only applies with --pursuit-controller."
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
    parser.add_argument("--w56rl-pitch-scale", type=float, default=0.20, help="Pitch residual authority for --mode w56rl.")
    parser.add_argument("--w56rl-zero-residual", action="store_true", help="Force residual output to zero for W56 parity testing.")
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
    # W57-W62 are narrow derivatives of W56.  Resolve them through the W56
    # configuration path so future W56-family settings cannot accidentally
    # diverge, then override only the measured terminal vertical-rate gain.
    requested_mode = args.mode

    if requested_mode == "w56rl":
        if args.bundle_dir is None:
            raise ValueError("--bundle-dir is required for w56rl mode")
        from dogfight.ai.w56_residual_action_provider import W56ResidualActionProvider
        return W56ResidualActionProvider(
            bundle_dir=args.bundle_dir,
            algorithm_factory=build_algorithm_from_bundle,
            policy_id=args.policy_id,
            pitch_scale=args.w56rl_pitch_scale,
            force_zero_residual=args.w56rl_zero_residual,
        )

    if requested_mode in ("w57", "w58", "w59", "w60", "w61", "w62"):
        args.mode = "w56"

    if args.mode == "bt":
        return BTActionProvider(dll_name=args.bt_dll)

    if args.mode in ("w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27", "w28", "w29", "w30", "w31", "w32", "w33", "w34", "w35", "w36", "w37", "w38", "w39", "w40", "w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56"):
        from dogfight.ai.integrated_bfm_controller import (
            IntegratedBFMConfig,
            IntegratedBFMController,
        )
        if args.mode in ("w40", "w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56"):
            # Corrected W39 gun-track: enter only from a stable, nearly
            # aligned solution and retain pursuit-side authority until the
            # horizontal error is essentially zero.
            controller = IntegratedBFMController(IntegratedBFMConfig(
                controller_name=args.mode,
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                adaptive_turn_prediction=True,
                turn_prediction_max_arc_deg=30.0,
                turn_prediction_unstable_horizon_s=0.8,
                turn_prediction_min_stable_s=0.7,
                turn_prediction_yaw_accel_limit_degps2=18.0,
                lag_pursuit_ata_deg=50.0,
                lag_pursuit_range_m=2800.0,
                lag_pursuit_closure_mps=170.0,
                lag_pursuit_offset_min_m=250.0,
                lag_pursuit_offset_max_m=750.0,
                lag_pursuit_offset_gain_s=2.0,
                vertical_alignment_elevation_deg=(
                    20.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                vertical_alignment_range_m=(
                    3200.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                vertical_alignment_closure_mps=(
                    60.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 1.0e9
                ),
                vertical_alignment_lag_offset_m=(
                    1400.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                vertical_alignment_lag_gain_m_per_deg=(
                    40.0 if args.mode in ("w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                vertical_alignment_lag_max_m=(
                    3000.0 if args.mode in ("w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                vertical_alignment_target_closure_mps=(
                    0.0 if args.mode in ("w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56")
                    else 20.0 if args.mode in ("w41", "w42")
                    else 0.0
                ),
                turn_match_enter_ata_deg=18.0,
                turn_match_exit_ata_deg=32.0,
                turn_match_enter_range_m=3000.0,
                turn_match_exit_range_m=3400.0,
                turn_match_min_threat_ata_deg=40.0,
                turn_match_enter_course_error_deg=12.0,
                turn_match_exit_course_error_deg=24.0,
                turn_match_enter_max_closure_mps=220.0,
                turn_match_entry_hold_s=0.5,
                turn_match_require_stable_prediction=True,
                turn_match_yaw_rate_gain=0.65,
                turn_match_course_gain=0.55,
                turn_match_los_rate_gain=0.20,
                turn_match_rate_limit_degps=12.0,
                turn_match_sign_guard_error_deg=4.0,
                turn_match_sign_guard_rate_degps=3.0,
                terminal_track_enter_ata_deg=(
                    20.0 if args.mode in ("w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 8.0 if args.mode in ("w44", "w45") else 0.0
                ),
                terminal_track_exit_ata_deg=(
                    30.0 if args.mode in ("w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 12.0 if args.mode in ("w44", "w45") else 0.0
                ),
                terminal_track_enter_range_m=(
                    1800.0 if args.mode in ("w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 1600.0 if args.mode in ("w44", "w45") else 0.0
                ),
                terminal_track_exit_range_m=(
                    4500.0 if args.mode in ("w54", "w55", "w56") else 2100.0 if args.mode in ("w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53") else 1900.0 if args.mode in ("w44", "w45") else 0.0
                ),
                terminal_track_prelock_ata_deg=(15.0 if args.mode in ("w55", "w56") else 6.0 if args.mode == "w54" else 0.0),
                terminal_track_prelock_range_m=(3000.0 if args.mode in ("w55", "w56") else 4000.0 if args.mode == "w54" else 0.0),
                terminal_track_min_threat_ata_deg=(
                    0.0 if args.mode in ("w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 20.0 if args.mode in ("w44", "w45", "w46", "w47") else 0.0
                ),
                terminal_track_yaw_rate_gain=(0.50 if args.mode in ("w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.70),
                terminal_track_course_gain=(0.80 if args.mode in ("w47", "w48") else 2.00 if args.mode in ("w46", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 1.20),
                terminal_track_los_rate_gain=(1.00 if args.mode in ("w47", "w48") else 0.40),
                terminal_track_rate_limit_degps=(16.0 if args.mode in ("w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 14.0),
                terminal_pitch_attitude_kp=(2.0 if args.mode in ("w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0),
                terminal_pitch_los_rate_gain=(1.0 if args.mode in ("w51", "w52", "w53", "w54", "w55", "w56") else 0.0),
                terminal_vertical_unload_el_deg=(0.0 if args.mode == "w56" else 6.0 if args.mode in ("w53", "w54", "w55") else 8.0 if args.mode == "w52" else 0.0),
                terminal_vertical_unload_ratio=(0.4 if args.mode in ("w53", "w54", "w55") else 0.7),
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=180.0,
                direct_course_bank_full_error_deg=60.0,
                direct_course_bank_exponent=0.50,
                direct_course_rear_ambiguity_deg=170.0,
                direct_course_roll_bias=0.07,
                direct_course_min_bank_deg=(78.0 if args.mode in ("w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0),
                direct_course_min_bank_ata_deg=(35.0 if args.mode in ("w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0),
                bank_reversal_error_deg=130.0,
                bank_reversal_full_cmd=0.75,
                bank_reversal_release_bank_deg=30.0,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=0.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_bank_deg=0.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                bank_kp=1.5,
                gamma_limit_deg=30.0,
                gamma_kp=(1.2 if args.mode in ("w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 1.0),
                pitch_rate_limit_degps=(
                    16.0 if args.mode in ("w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 10.0
                ),
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                fine_vertical_los_blend=(
                    0.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.25
                ),
                vertical_prediction_horizon_s=(
                    0.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 1.0
                ),
                use_altitude_rate_vertical_prediction=(
                    args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56")
                ),
                vertical_prediction_stable_horizon_s=1.8,
                vertical_prediction_unstable_horizon_s=0.35,
                vertical_prediction_min_stable_s=0.7,
                vertical_prediction_accel_limit_mps2=35.0,
                vertical_maneuver_gamma_limit_deg=(
                    68.0 if args.mode in ("w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56")
                    else 55.0 if args.mode == "w41"
                    else 0.0
                ),
                vertical_maneuver_ata_gate_deg=(
                    75.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                vertical_bank_relief_elevation_deg=(
                    18.0 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                vertical_bank_relief_min_scale=(
                    0.35 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 1.0
                ),
                vertical_speed_damping_deadband_mps=20.0,
                vertical_speed_damping_gain=(
                    0.006 if args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.010
                ),
                vertical_speed_damping_track_desired_gamma=(
                    args.mode in ("w41", "w42", "w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56")
                ),
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.0,
                throttle_max=1.00,
                throttle_merge=0.95,
                throttle_slew_per_s=0.80,
                closure_throttle_ata_deg=55.0,
                closure_throttle_range_m=4000.0,
                closure_throttle_far_range_m=3000.0,
                closure_throttle_near_range_m=1500.0,
                closure_target_far_mps=100.0,
                closure_target_mid_mps=45.0,
                closure_target_near_mps=10.0,
                closure_throttle_base=0.48,
                closure_throttle_gain=0.005,
                minimum_energy_speed_mps=(
                    165.0 if args.mode in ("w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
                minimum_energy_throttle=(
                    0.90 if args.mode in ("w43", "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53", "w54", "w55", "w56") else 0.0
                ),
            ))
            if requested_mode in ("w57", "w58", "w59", "w60", "w61", "w62"):
                controller.cfg.controller_name = requested_mode
                controller.cfg.terminal_pitch_los_rate_gain = 1.5
            if requested_mode == "w58":
                # Do not let fine tracking override the defensive manager in
                # a mutual head-on.  Terminal tracking is reserved for a
                # sufficiently asymmetric side/rear-quarter opportunity.
                controller.cfg.terminal_track_min_threat_ata_deg = 40.0
            if requested_mode in ("w59", "w60", "w61", "w62"):
                # The controller releases its terminal latch below half of
                # this threshold.  80deg therefore means enter at >=80deg
                # and release as soon as threat ATA falls below 40deg.
                controller.cfg.terminal_track_min_threat_ata_deg = 80.0
            if requested_mode == "w60":
                # W59 correctly avoided a mutual head-on but its defensive
                # state had no actual escape manoeuvre in the W40+ family.
                # Add a bounded out-of-plane break only under an immediate,
                # close nose-on threat; normal acquisition stays unchanged.
                controller.cfg.defensive_vertical_escape = True
                controller.cfg.defensive_escape_threat_ata_deg = 20.0
                controller.cfg.defensive_escape_own_ata_deg = 60.0
                controller.cfg.defensive_escape_range_m = 1800.0
                controller.cfg.defensive_escape_gamma_deg = 25.0
                controller.cfg.defensive_escape_deadband_m = 120.0
                controller.cfg.defensive_escape_switch_s = 2.3
            if requested_mode in ("w61", "w62"):
                # Keep W59's stable defensive baseline, then convert a
                # defeated nose-on threat into a side/rear pursuit setup.
                controller.cfg.post_defense_conversion_duration_s = 4.0
                controller.cfg.post_defense_conversion_arm_window_s = (
                    3.0 if requested_mode == "w62" else 0.0
                )
                controller.cfg.post_defense_conversion_min_threat_ata_deg = 40.0
                controller.cfg.post_defense_conversion_max_range_m = 3000.0
                controller.cfg.post_defense_conversion_rear_offset_m = 1500.0
                controller.cfg.post_defense_conversion_lateral_offset_m = 800.0
                controller.cfg.post_defense_conversion_min_target_speed_mps = 40.0
            return controller
        if args.mode == "w39":
            # W33 acquisition unchanged; only after a valid rear-quarter
            # setup, match target turn rate and manage closure for a sustained
            # gun solution instead of cutting through the target's circle.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w39",
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                adaptive_turn_prediction=True,
                turn_prediction_max_arc_deg=30.0,
                turn_prediction_unstable_horizon_s=0.8,
                turn_prediction_min_stable_s=0.7,
                turn_prediction_yaw_accel_limit_degps2=18.0,
                lag_pursuit_ata_deg=50.0,
                lag_pursuit_range_m=2800.0,
                lag_pursuit_closure_mps=160.0,
                lag_pursuit_offset_min_m=300.0,
                lag_pursuit_offset_max_m=900.0,
                lag_pursuit_offset_gain_s=2.5,
                turn_match_enter_ata_deg=25.0,
                turn_match_exit_ata_deg=42.0,
                turn_match_enter_range_m=2800.0,
                turn_match_exit_range_m=3600.0,
                turn_match_min_threat_ata_deg=40.0,
                turn_match_yaw_rate_gain=1.0,
                turn_match_course_gain=0.12,
                turn_match_rate_limit_degps=14.0,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=180.0,
                direct_course_bank_full_error_deg=60.0,
                direct_course_bank_exponent=0.50,
                direct_course_rear_ambiguity_deg=170.0,
                direct_course_roll_bias=0.07,
                bank_reversal_error_deg=125.0,
                bank_reversal_full_cmd=0.80,
                bank_reversal_release_bank_deg=25.0,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=0.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_bank_deg=0.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                bank_kp=1.5,
                gamma_limit_deg=30.0,
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                vertical_prediction_horizon_s=1.0,
                vertical_speed_damping_deadband_mps=20.0,
                vertical_speed_damping_gain=0.010,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.20,
                throttle_max=1.00,
                throttle_merge=0.95,
                throttle_slew_per_s=0.80,
                closure_throttle_ata_deg=55.0,
                closure_throttle_range_m=4000.0,
                closure_throttle_far_range_m=3000.0,
                closure_throttle_near_range_m=1500.0,
                closure_target_far_mps=100.0,
                closure_target_mid_mps=45.0,
                closure_target_near_mps=10.0,
                closure_throttle_base=0.48,
                closure_throttle_gain=0.005,
            ))
        if args.mode in ("w34", "w35", "w36", "w37", "w38"):
            # W33 horizontal maneuver prediction plus world-frame vertical
            # pursuit and a real defensive escape plane.  W36-W38 are
            # pre-registered fallback profiles selected by live symptoms.
            profiles = {
                "w34": dict(lag_ata=35.0, lag_range=2200.0, lag_closure=100.0,
                            lag_min=250.0, lag_max=800.0,
                            rev_error=0.0, rev_cmd=0.0, rev_release=0.0,
                            def_threat=20.0, def_own=60.0, def_range=1800.0,
                            def_gamma=32.0, def_switch=2.3),
                "w35": dict(lag_ata=60.0, lag_range=2800.0, lag_closure=180.0,
                            lag_min=400.0, lag_max=1200.0,
                            rev_error=120.0, rev_cmd=0.80, rev_release=25.0,
                            def_threat=20.0, def_own=25.0, def_range=2500.0,
                            def_gamma=38.0, def_switch=2.3),
                # Attack fallback: moderate early lag and fast reversal, but
                # conservative defence so it does not abandon a gun setup.
                "w36": dict(lag_ata=55.0, lag_range=2600.0, lag_closure=220.0,
                            lag_min=300.0, lag_max=900.0,
                            rev_error=120.0, rev_cmd=0.80, rev_release=25.0,
                            def_threat=15.0, def_own=45.0, def_range=1800.0,
                            def_gamma=32.0, def_switch=2.3),
                # Defensive fallback: deny a stable enemy gun solution early.
                "w37": dict(lag_ata=60.0, lag_range=2800.0, lag_closure=180.0,
                            lag_min=400.0, lag_max=1200.0,
                            rev_error=120.0, rev_cmd=0.80, rev_release=25.0,
                            def_threat=30.0, def_own=20.0, def_range=3000.0,
                            def_gamma=42.0, def_switch=1.7),
                # Stability fallback: smaller offsets and a softer reversal
                # if W35 chatters or repeatedly swaps turn direction.
                "w38": dict(lag_ata=50.0, lag_range=2400.0, lag_closure=220.0,
                            lag_min=300.0, lag_max=900.0,
                            rev_error=140.0, rev_cmd=0.65, rev_release=35.0,
                            def_threat=20.0, def_own=35.0, def_range=2200.0,
                            def_gamma=34.0, def_switch=2.5),
            }
            profile = profiles[args.mode]
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name=args.mode,
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                adaptive_turn_prediction=True,
                turn_prediction_max_arc_deg=30.0,
                turn_prediction_unstable_horizon_s=0.8,
                turn_prediction_min_stable_s=0.7,
                turn_prediction_yaw_accel_limit_degps2=18.0,
                lag_pursuit_ata_deg=profile["lag_ata"],
                lag_pursuit_range_m=profile["lag_range"],
                lag_pursuit_closure_mps=profile["lag_closure"],
                lag_pursuit_offset_min_m=profile["lag_min"],
                lag_pursuit_offset_max_m=profile["lag_max"],
                lag_pursuit_offset_gain_s=2.5,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=180.0,
                direct_course_bank_full_error_deg=60.0,
                direct_course_bank_exponent=0.50,
                direct_course_rear_ambiguity_deg=170.0,
                direct_course_roll_bias=0.07,
                bank_reversal_error_deg=profile["rev_error"],
                bank_reversal_full_cmd=profile["rev_cmd"],
                bank_reversal_release_bank_deg=profile["rev_release"],
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=0.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_bank_deg=0.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                bank_kp=1.5,
                gamma_limit_deg=30.0,
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                fine_vertical_los_blend=0.0,
                use_altitude_rate_vertical_prediction=True,
                vertical_prediction_stable_horizon_s=1.8,
                vertical_prediction_unstable_horizon_s=0.35,
                vertical_prediction_min_stable_s=0.7,
                vertical_prediction_accel_limit_mps2=35.0,
                vertical_maneuver_gamma_limit_deg=55.0,
                vertical_maneuver_ata_gate_deg=75.0,
                vertical_bank_relief_elevation_deg=18.0,
                vertical_bank_relief_min_scale=0.35,
                vertical_speed_damping_deadband_mps=20.0,
                vertical_speed_damping_gain=0.010,
                defensive_vertical_escape=True,
                defensive_escape_threat_ata_deg=profile["def_threat"],
                defensive_escape_own_ata_deg=profile["def_own"],
                defensive_escape_range_m=profile["def_range"],
                defensive_escape_gamma_deg=profile["def_gamma"],
                defensive_escape_deadband_m=120.0,
                defensive_escape_switch_s=profile["def_switch"],
                max_test_duration_s=0.0,
                max_test_pitch_cmd=0.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.0,
                throttle_max=1.00,
                throttle_merge=0.95,
                throttle_slew_per_s=0.80,
                closure_throttle_ata_deg=50.0,
                closure_throttle_range_m=4000.0,
                closure_throttle_far_range_m=3000.0,
                closure_throttle_near_range_m=1500.0,
                closure_target_far_mps=120.0,
                closure_target_mid_mps=70.0,
                closure_target_near_mps=20.0,
                closure_throttle_base=0.50,
                closure_throttle_gain=0.004,
            ))
        if args.mode == "w33":
            # Maneuver-aware W32: constrain constant-turn prediction to a
            # believable arc, collapse the horizon while the target changes
            # turn, and use lag pursuit only after alignment at high closure.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w33",
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                adaptive_turn_prediction=True,
                turn_prediction_max_arc_deg=30.0,
                turn_prediction_unstable_horizon_s=0.8,
                turn_prediction_min_stable_s=0.7,
                turn_prediction_yaw_accel_limit_degps2=18.0,
                lag_pursuit_ata_deg=35.0,
                lag_pursuit_range_m=2200.0,
                lag_pursuit_closure_mps=100.0,
                lag_pursuit_offset_min_m=250.0,
                lag_pursuit_offset_max_m=800.0,
                lag_pursuit_offset_gain_s=2.5,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=180.0,
                direct_course_bank_full_error_deg=60.0,
                direct_course_bank_exponent=0.50,
                direct_course_rear_ambiguity_deg=170.0,
                direct_course_roll_bias=0.07,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=0.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_bank_deg=0.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                bank_kp=1.5,
                gamma_limit_deg=30.0,
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                vertical_prediction_horizon_s=1.0,
                vertical_speed_damping_deadband_mps=20.0,
                vertical_speed_damping_gain=0.010,
                max_test_duration_s=0.0,
                max_test_pitch_cmd=0.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.0,
                throttle_max=1.00,
                throttle_merge=0.95,
                throttle_slew_per_s=0.80,
                closure_throttle_ata_deg=50.0,
                closure_throttle_range_m=4000.0,
                closure_throttle_far_range_m=3000.0,
                closure_throttle_near_range_m=1500.0,
                closure_target_far_mps=120.0,
                closure_target_mid_mps=70.0,
                closure_target_near_mps=20.0,
                closure_throttle_base=0.50,
                closure_throttle_gain=0.004,
            ))
        if args.mode == "w32":
            # W31 reached ATA 9.8deg but only while still 3.2km away; on the
            # next merge closure was 300+m/s.  Start energy management before
            # the WEZ using radial closure, not scalar speed difference.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w32",
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=180.0,
                direct_course_bank_full_error_deg=60.0,
                direct_course_bank_exponent=0.50,
                direct_course_rear_ambiguity_deg=170.0,
                direct_course_roll_bias=0.07,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=0.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_bank_deg=0.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                bank_kp=1.5,
                gamma_limit_deg=30.0,
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                vertical_prediction_horizon_s=1.0,
                vertical_speed_damping_deadband_mps=20.0,
                vertical_speed_damping_gain=0.010,
                max_test_duration_s=0.0,
                max_test_pitch_cmd=0.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.0,
                throttle_max=1.00,
                throttle_merge=0.95,
                throttle_slew_per_s=0.80,
                closure_throttle_ata_deg=50.0,
                closure_throttle_range_m=4000.0,
                closure_throttle_far_range_m=3000.0,
                closure_throttle_near_range_m=1500.0,
                closure_target_far_mps=120.0,
                closure_target_mid_mps=70.0,
                closure_target_near_mps=20.0,
                closure_throttle_base=0.50,
                closure_throttle_gain=0.004,
            ))
        if args.mode == "w31":
            # W30 solved horizontal capture (ATA 6deg) but its -0.8 banked
            # turn feed-forward exceeded the gamma loop's +0.4 maximum
            # correction, allowing +129m/s climb.  Preserve W30 guidance and
            # add final-stage vertical-rate braking outside a 20m/s deadband.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w31",
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=180.0,
                direct_course_bank_full_error_deg=60.0,
                direct_course_bank_exponent=0.50,
                direct_course_rear_ambiguity_deg=170.0,
                direct_course_roll_bias=0.07,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=0.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_bank_deg=0.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                bank_kp=1.5,
                gamma_limit_deg=30.0,
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                vertical_prediction_horizon_s=1.0,
                vertical_speed_damping_deadband_mps=20.0,
                vertical_speed_damping_gain=0.010,
                max_test_duration_s=0.0,
                max_test_pitch_cmd=0.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.40,
                throttle_max=1.00,
                throttle_merge=0.95,
            ))
        if args.mode == "w30":
            # No coarse latch, no fixed minimum turn request, no timed max-
            # authority override.  Every 10Hz synchronized telemetry pair
            # maps the latest predicted horizontal course error continuously
            # into bank, while target-Z/gamma owns pitch.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w30",
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=180.0,
                direct_course_bank_full_error_deg=60.0,
                direct_course_bank_exponent=0.50,
                direct_course_rear_ambiguity_deg=170.0,
                direct_course_roll_bias=0.07,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=0.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_bank_deg=0.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                bank_kp=1.5,
                gamma_limit_deg=30.0,
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                vertical_prediction_horizon_s=1.0,
                max_test_duration_s=0.0,
                max_test_pitch_cmd=0.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.40,
                throttle_max=1.00,
                throttle_merge=0.95,
            ))
        if args.mode == "w29":
            # W28 proved horizontal guidance was corrected, but its fixed
            # -0.9 acquisition pull overrode an explicit dive request while
            # the target was hundreds of metres below.  Let the gamma loop
            # own final pitch; a stronger banked-turn feed-forward supplies
            # load only when vertical tracking permits it.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w29",
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=120.0,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=10.5,
                acquisition_min_turn_rate_ata_deg=35.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=70.0,
                coarse_turn_reentry_ata_deg=100.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                gamma_limit_deg=30.0,
                turn_pull_at_max_bank=-0.80,
                pitch_cmd_limit=1.00,
                vertical_prediction_horizon_s=1.0,
                max_test_duration_s=60.0,
                max_test_requires_coarse_turn=False,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                # Disable fixed-pitch override: gamma/target-Z loop owns it.
                max_test_pitch_cmd=0.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.40,
                throttle_max=1.00,
                throttle_merge=0.95,
            ))
        if args.mode == "w28":
            # Structural guidance correction: bank commands use roll-free
            # world horizontal course error, target motion follows a measured
            # constant-turn arc, and defensive guidance keeps short lead.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w28",
                use_horizontal_course_guidance=True,
                use_constant_turn_prediction=True,
                target_velocity_alpha=0.80,
                horizon_defensive_s=0.8,
                guidance_rear_commit_deg=120.0,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=10.5,
                acquisition_min_turn_rate_ata_deg=35.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=70.0,
                coarse_turn_reentry_ata_deg=100.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=60.0,
                max_test_requires_coarse_turn=False,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=60.0,
                max_test_pitch_ata_gate_deg=35.0,
                max_test_pitch_cmd=-0.90,
                max_test_pitch_vertical_gain=0.0,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.40,
                throttle_max=1.00,
                throttle_merge=0.95,
            ))
        if args.mode == "w27":
            # Pursuit-priority experiment: W26's climb suppression relaxed
            # pull to -0.5 and turn rate fell to 7.5deg/s before capture.
            # Accept altitude gain, hold at least -0.9 pull while ATA is large,
            # and retain only speed/altitude hard gates.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w27",
                guidance_rear_commit_deg=120.0,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=10.5,
                acquisition_min_turn_rate_ata_deg=35.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=70.0,
                coarse_turn_reentry_ata_deg=100.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=60.0,
                max_test_requires_coarse_turn=False,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=60.0,
                max_test_pitch_ata_gate_deg=35.0,
                max_test_pitch_cmd=-0.90,
                max_test_pitch_relaxed_cmd=0.0,
                max_test_pitch_vertical_gain=0.0,
                max_test_pitch_bank_relax_gain=0.0,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.85,
                corner_speed_gain=0.010,
                throttle_min=0.40,
                throttle_max=1.00,
                throttle_merge=0.95,
            ))
        if args.mode == "w26":
            # W25 dropped to a 66deg bank / 6deg/s request at transition
            # while retaining strong pull, converting lift into a +59m/s
            # climb.  Guarantee useful acquisition rate and relax pull as
            # bank falls below the 80deg max-rate reference.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w26",
                guidance_rear_commit_deg=120.0,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                acquisition_min_turn_rate_degps=10.5,
                acquisition_min_turn_rate_ata_deg=35.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=70.0,
                coarse_turn_reentry_ata_deg=100.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=60.0,
                max_test_requires_coarse_turn=False,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=60.0,
                max_test_pitch_ata_gate_deg=35.0,
                max_test_pitch_cmd=-1.00,
                max_test_pitch_relaxed_cmd=-0.35,
                max_test_pitch_vertical_gain=0.010,
                max_test_target_vertical_speed_mps=0.0,
                max_test_pitch_bank_relax_gain=0.020,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.80,
                corner_speed_gain=0.008,
                throttle_min=0.35,
                throttle_max=0.98,
                throttle_merge=0.90,
            ))
        if args.mode == "w25":
            # W24's intercept controller was capped at 5.5deg/s, collapsing
            # target bank to 64deg while the opponent still held 12deg/s.
            # Raise the intercept authority and add coarse-turn hysteresis so
            # ATA noise around 70deg cannot toggle both roll and pull modes.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w25",
                guidance_rear_commit_deg=120.0,
                turn_rate_break_degps=12.0,
                turn_rate_reacquire_degps=11.0,
                turn_rate_track_degps=8.0,
                turn_rate_defensive_degps=12.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=70.0,
                coarse_turn_reentry_ata_deg=100.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=60.0,
                max_test_requires_coarse_turn=False,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=60.0,
                max_test_pitch_ata_gate_deg=35.0,
                max_test_pitch_cmd=-1.00,
                max_test_pitch_relaxed_cmd=-0.35,
                max_test_pitch_vertical_gain=0.010,
                max_test_target_vertical_speed_mps=0.0,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.80,
                corner_speed_gain=0.008,
                throttle_min=0.35,
                throttle_max=0.98,
                throttle_merge=0.90,
            ))
        if args.mode == "w24":
            # W23 matched the opponent's max-rate turn but stayed in a fixed
            # 80deg circle through the merge.  Exit coarse turn at ATA 70deg
            # so the existing predicted-intercept guidance can shape the
            # approach; keep adaptive pull alive down to ATA 35deg.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w24",
                guidance_rear_commit_deg=120.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=70.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=45.0,
                max_test_requires_coarse_turn=False,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=68.0,
                max_test_pitch_ata_gate_deg=35.0,
                max_test_pitch_cmd=-1.00,
                max_test_pitch_relaxed_cmd=-0.70,
                max_test_pitch_vertical_gain=0.006,
                max_test_target_vertical_speed_mps=0.0,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.80,
                corner_speed_gain=0.008,
                throttle_min=0.35,
                throttle_max=0.98,
                throttle_merge=0.90,
            ))
        if args.mode == "w23":
            # W22 stabilized bank/vertical speed but relaxed to -0.82 pull
            # and bled to 194m/s, leaving turn rate at 9.2deg/s.  Keep more
            # pull and add power to hold the opponent's measured 210-215m/s.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w23",
                guidance_rear_commit_deg=120.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=45.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=30.0,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=68.0,
                max_test_pitch_ata_gate_deg=60.0,
                max_test_pitch_cmd=-1.00,
                max_test_pitch_relaxed_cmd=-0.70,
                max_test_pitch_vertical_gain=0.006,
                max_test_target_vertical_speed_mps=0.0,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.80,
                corner_speed_gain=0.008,
                throttle_min=0.35,
                throttle_max=0.98,
                throttle_merge=0.90,
            ))
        if args.mode == "w22":
            # W21 found the useful full-pull point (12-13deg/s near level),
            # but fixed -1.0 then climbed and bled speed.  Keep the fast roll
            # brake, add the measured +0.07 holding bias, and relax pull in
            # proportion to positive vertical speed.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w22",
                guidance_rear_commit_deg=120.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=45.0,
                coarse_turn_bank_deg=80.0,
                coarse_turn_roll_bias=0.07,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=15.0,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=68.0,
                max_test_pitch_ata_gate_deg=60.0,
                max_test_pitch_cmd=-1.00,
                max_test_pitch_relaxed_cmd=-0.45,
                max_test_pitch_vertical_gain=0.012,
                max_test_target_vertical_speed_mps=0.0,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.50,
                corner_speed_gain=0.006,
                throttle_min=0.20,
                throttle_merge=0.65,
            ))
        if args.mode == "w21":
            # Top-down follow-up to W20: W20 proved that the live aircraft can
            # reach 11-15deg/s, but holding +0.8 roll to 70deg overshot 91deg.
            # Release at 45deg so the measured roll-rate cascade brakes the
            # inertia, while testing full -1.0 pull and slightly more power.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w21",
                guidance_rear_commit_deg=120.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=45.0,
                coarse_turn_bank_deg=80.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.50,
                pitch_cmd_limit=1.00,
                max_test_duration_s=15.0,
                max_test_roll_full_until_deg=45.0,
                max_test_roll_full_cmd=0.80,
                # Equal thresholds deliberately disable W20's +0.30 taper
                # band; after 45deg the closed-loop rate brake takes over.
                max_test_roll_taper_until_deg=45.0,
                max_test_roll_taper_cmd=0.0,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.50,
                max_test_pitch_bank_gate_deg=70.0,
                max_test_pitch_ata_gate_deg=60.0,
                max_test_pitch_cmd=-1.00,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.50,
                corner_speed_gain=0.006,
                throttle_min=0.20,
                throttle_merge=0.65,
            ))
        if args.mode == "w20":
            # Short upper-bound system-identification run.  Use large control
            # authority with bank-aware taper/braking, then stop after 15s.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w20",
                guidance_rear_commit_deg=120.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=45.0,
                coarse_turn_bank_deg=80.0,
                roll_rate_limit_degps=90.0,
                roll_cmd_limit=0.80,
                turn_pull_at_max_bank=-0.40,
                pitch_cmd_limit=0.80,
                max_test_duration_s=15.0,
                max_test_roll_full_until_deg=70.0,
                max_test_roll_full_cmd=0.80,
                max_test_roll_taper_until_deg=78.0,
                max_test_roll_taper_cmd=0.30,
                max_test_roll_brake_above_deg=82.0,
                max_test_roll_brake_cmd=0.35,
                max_test_pitch_bank_gate_deg=70.0,
                max_test_pitch_ata_gate_deg=60.0,
                max_test_pitch_cmd=-0.80,
                max_test_min_speed_mps=180.0,
                max_test_min_altitude_m=1200.0,
                corner_speed_mps=215.0,
                corner_throttle_base=0.35,
                corner_speed_gain=0.006,
                throttle_min=0.15,
                throttle_merge=0.55,
            ))
        if args.mode == "w19":
            # Match the live opponent's measured max-rate state: about 80deg
            # bank, 214-215m/s, nearly level flight and 12deg/s turn rate.
            # Do not clamp pitch as W18 did; the gamma loop must retain
            # authority to stop the +40m/s climbing spiral.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w19",
                guidance_rear_commit_deg=120.0,
                max_bank_deg=82.0,
                break_bank_deg=80.0,
                coarse_turn_ata_deg=45.0,
                coarse_turn_bank_deg=80.0,
                roll_rate_limit_degps=45.0,
                roll_cmd_limit=0.34,
                turn_pull_at_max_bank=-0.40,
                pitch_cmd_limit=0.50,
                initial_commit_duration_s=12.0,
                initial_commit_ata_deg=45.0,
                initial_commit_bank_deg=80.0,
                initial_commit_release_bank_deg=75.0,
                initial_commit_min_roll_cmd=0.18,
                corner_speed_mps=215.0,
                corner_throttle_base=0.35,
                corner_speed_gain=0.006,
                throttle_min=0.15,
                throttle_merge=0.55,
            ))
        if args.mode == "w18":
            # W18 keeps W17's fast bank commitment and prevents the vertical
            # gamma loop from relaxing pull while the target is still far
            # outside the nose.  Speed/altitude gates bound the energy risk.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w18",
                guidance_rear_commit_deg=120.0,
                turn_pull_at_max_bank=-0.34,
                pitch_cmd_limit=0.50,
                initial_commit_duration_s=12.0,
                initial_commit_ata_deg=60.0,
                initial_commit_bank_deg=75.0,
                initial_commit_release_bank_deg=65.0,
                initial_commit_min_roll_cmd=0.18,
                sustained_pull_rear_cmd=-0.34,
                sustained_pull_rear_ata_deg=90.0,
                sustained_pull_mid_cmd=-0.28,
                sustained_pull_mid_ata_deg=60.0,
                sustained_pull_min_bank_deg=55.0,
                sustained_pull_min_speed_mps=210.0,
                sustained_pull_min_altitude_m=1200.0,
            ))
        if args.mode == "w17":
            # W17 keeps W16's vertical/load-factor settings and changes only
            # initial bank acquisition.  Hold useful roll authority until
            # 65deg bank instead of tapering almost immediately.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w17",
                guidance_rear_commit_deg=120.0,
                turn_pull_at_max_bank=-0.34,
                pitch_cmd_limit=0.50,
                initial_commit_duration_s=12.0,
                initial_commit_ata_deg=60.0,
                initial_commit_bank_deg=75.0,
                initial_commit_release_bank_deg=65.0,
                initial_commit_min_roll_cmd=0.18,
            ))
        if args.mode == "w16":
            # W16 changes only the banked-turn load authority from W15.
            # W15 reached 70-78deg bank but produced only ~4.8deg/s yaw and
            # lost 692m, showing that bank angle was not converted into
            # enough pull/load factor.
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w16",
                guidance_rear_commit_deg=120.0,
                turn_pull_at_max_bank=-0.34,
                pitch_cmd_limit=0.50,
            ))
        if args.mode == "w15":
            return IntegratedBFMController(IntegratedBFMConfig(
                controller_name="w15",
                guidance_rear_commit_deg=120.0,
            ))
        return IntegratedBFMController(IntegratedBFMConfig(controller_name="w14"))

    if args.mode in ("w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13"):
        from dogfight.ai.w1_controller import W1ControllerActionProvider, W1Config
        if args.mode == "w13":
            # W13 is W12 with the invalid vertical extrapolation removed:
            # predict only horizontal N/E motion and track current altitude.
            return W1ControllerActionProvider(W1Config(
                controller_name="w13",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=180.0,
                intercept_horizon_min_s=0.5,
                intercept_horizon_max_s=8.0,
                intercept_velocity_clip_mps=400.0,
                intercept_predict_vertical=False,
            ))
        if args.mode == "w12":
            # W12 is the non-throttle candidate: predict a target intercept
            # point from measured target motion with a bounded time-to-range
            # horizon. Rear sign is no longer permanently locked.
            return W1ControllerActionProvider(W1Config(
                controller_name="w12",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=180.0,
                intercept_horizon_min_s=0.5,
                intercept_horizon_max_s=8.0,
                intercept_velocity_clip_mps=400.0,
            ))
        if args.mode == "w11":
            # W11 adds absolute corner-speed pressure and a close-threat
            # acceleration override to W10's relative-speed schedule.
            return W1ControllerActionProvider(W1Config(
                controller_name="w11",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
                dynamic_throttle_min=0.25,
                dynamic_speed_excess_start_mps=10.0,
                dynamic_speed_excess_full_mps=70.0,
                dynamic_throttle_slew_per_s=0.40,
                dynamic_absolute_speed_start_mps=280.0,
                dynamic_absolute_speed_full_mps=360.0,
                threat_override_range_m=1800.0,
                threat_override_closure_mps=20.0,
                threat_override_target_advantage_mps=15.0,
            ))
        if args.mode == "w10":
            # W10 generalizes W9: throttle reduction is proportional to
            # own-minus-target speed and ATA, and restores automatically when
            # the target becomes faster. No time/opponent-specific trigger.
            return W1ControllerActionProvider(W1Config(
                controller_name="w10",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
                dynamic_throttle_min=0.40,
                dynamic_speed_excess_start_mps=10.0,
                dynamic_speed_excess_full_mps=70.0,
                dynamic_throttle_slew_per_s=0.30,
            ))
        if args.mode == "w9":
            # W9 returns to W5's stable one-direction commitment and changes
            # one variable: cut throttle in the rear hemisphere to reduce
            # speed and turn radius instead of orbiting faster than target.
            return W1ControllerActionProvider(W1Config(
                controller_name="w9",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
                throttle_rear=0.40,
            ))
        if args.mode == "w8":
            # W8 preserves W7's trend trigger but prevents the bank-reversal
            # transient from immediately triggering a reversal back. A new
            # commitment must be held for 45 seconds before reassessment.
            return W1ControllerActionProvider(W1Config(
                controller_name="w8",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
                rear_trend_flip_s=10.0,
                rear_trend_worsen_deg=3.0,
                rear_trend_min_hold_s=45.0,
            ))
        if args.mode == "w7":
            # W7 changes W5's permanent rear commitment into a trend-based
            # recommit: reverse only when ATA worsens while range closes over
            # a ten-second rear-hemisphere window.
            return W1ControllerActionProvider(W1Config(
                controller_name="w7",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
                rear_trend_flip_s=10.0,
                rear_trend_worsen_deg=3.0,
            ))
        if args.mode == "w6":
            # W6 changes exactly one control behavior from W5: a permanent
            # rear commitment becomes progress-aware. Recommit only after
            # ATA has failed to improve by 1 degree for eight seconds.
            return W1ControllerActionProvider(W1Config(
                controller_name="w6",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
                rear_stall_flip_s=8.0,
                rear_progress_deg=1.0,
            ))
        if args.mode == "w5":
            # W5 changes exactly one variable from W4: target bank authority.
            # W4's stronger pull was cancelled by the gamma loop after 10 s
            # and mostly became climb, so do not increase pull again. Use a
            # steeper bank to create more horizontal turn acceleration.
            return W1ControllerActionProvider(W1Config(
                controller_name="w5",
                bank_step_test=args.w1_bank_step_test,
                max_bank_deg=65.0,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
            ))
        if args.mode == "w4":
            # W4 changes exactly one variable from W3: stronger banked-turn
            # pull. Live run0052 proved rear commitment works, but a roughly
            # 50-degree bank still produced only about 0.87 deg/s of turn.
            return W1ControllerActionProvider(W1Config(
                controller_name="w4",
                bank_step_test=args.w1_bank_step_test,
                turn_pitch_feedforward_at_45_deg=-0.20,
                rear_commit_az_deg=120.0,
            ))
        if args.mode == "w3":
            return W1ControllerActionProvider(W1Config(
                controller_name="w3",
                bank_step_test=args.w1_bank_step_test,
                turn_pitch_feedforward_at_45_deg=-0.12,
                rear_commit_az_deg=120.0,
            ))
        if args.mode == "w2":
            return W1ControllerActionProvider(W1Config(
                controller_name="w2",
                bank_step_test=args.w1_bank_step_test,
                turn_pitch_feedforward_at_45_deg=-0.12,
            ))
        return W1ControllerActionProvider(W1Config(bank_step_test=args.w1_bank_step_test))

    if args.mode == "pulse":
        from dogfight.ai.pulse_test_provider import (
            PITCH_PULSE_SEQUENCE,
            PITCH_TRIM_SWEEP_SEQUENCE,
            ROLL_INERTIA_SEQUENCE,
            YAW_PULSE_SEQUENCE,
            PulseTestActionProvider,
            PulseTestConfig,
        )
        sequence_map = {
            "pitch": PITCH_PULSE_SEQUENCE,
            "pitch_trim": PITCH_TRIM_SWEEP_SEQUENCE,
            "roll_inertia": ROLL_INERTIA_SEQUENCE,
            "yaw": YAW_PULSE_SEQUENCE,
        }
        sequence = sequence_map.get(args.pulse_sequence)
        cfg = PulseTestConfig(loop=args.pulse_loop) if sequence is None else PulseTestConfig(
            sequence=sequence, loop=args.pulse_loop
        )
        return PulseTestActionProvider(cfg)

    if args.bundle_dir is None:
        raise ValueError("--bundle-dir is required for rl and hybrid modes")

    rl_provider = RLActionProvider(
        bundle_dir=args.bundle_dir,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=args.explore,
    )

    if args.mode == "rl":
        if args.tactical_wrapper and args.pursuit_controller:
            raise ValueError("--tactical-wrapper and --pursuit-controller are mutually exclusive")
        if args.pursuit_controller:
            from dogfight.ai.pursuit_controller import (
                PursuitControllerActionProvider,
                PursuitControllerConfig,
            )
            return PursuitControllerActionProvider(
                rl_provider,
                PursuitControllerConfig(
                    world_frame_pitch_gate=args.pc_world_frame_pitch_gate,
                    turn_pull_decomposition=args.pc_turn_pull_decomposition,
                ),
            )
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
