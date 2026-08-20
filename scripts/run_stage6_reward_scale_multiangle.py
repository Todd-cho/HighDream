"""Parameterized reward-scale sweep for the Stage4-based, 4-angle-pool
stability line. Reintroduces engagement/aggression reward terms gradually
(as a fraction --scale of stage6e_delta_down_400iter's original, validated
values) on top of stage6ll's safe stability-only base, per user direction
(2026-08-19): "이어서 진행해주고 전에 쌓은 리워드 파라미터가 좋았던 걸 참고해서
빠르게 성능 올려보자" -- reuse the historically-good stage6e reward magnitudes
rather than reinventing them, but reintroduce them incrementally instead of
all at once (which is exactly the combination -- full engagement reward +
chase-habit-laden weights -- that produced stage6hh/ii/jj's crashes).

Background chain: stage6ll (Stage4 source, 4-angle pool [0,8,15,22]deg,
stability-only reward -- every engagement term at 0.0) achieved crash=0%
(20/20) with minimum_altitude_worst=988.8m, this line's best safety
margin so far, but wez_episode_rate is only 5% (mean_distance ~47000m --
policy flees far since there's zero incentive to stay close). Each
engagement-driving reward field is scaled linearly between 0.0 (stage6ll)
and its stage6e_delta_down_400iter value (validated over hundreds of
loiter-opponent iterations, and used byte-for-byte by every
stage6u-6ee experiment in this session): range_scale=2.4,
overshoot_penalty=4.0, overshoot_quadratic_scale=1.5,
inside_min_range_penalty=-3.0, ata_scale=0.15, aa_scale=0.04,
wez_bonus=1.0, damage_scale=20.0, attack_range_bonus=0.9,
far_range_penalty=1.05, range_delta_scale=0.03, ata_delta_scale=0.15.
Safety/control terms (altitude floors+bonuses, nose_down, roll/pitch
limits, crash_penalty, step_penalty, survival_bonus) stay fixed at
stage6cc/6ll's values regardless of scale.

Each invocation is a FRESH restart from Stage4
(altitude_native_curriculum_v1_stage4_banked_descent_recovery_C10) --
never a continuation of a previous scale level's checkpoint -- per this
project's repeatedly-confirmed finding that continuation/restore
introduces its own instability (rule 14) independent of reward content.
Same 4-angle scenario_pool (0/8/15/22deg) and eased scripted_pursuit
opponent config as stage6cc/6ll throughout.

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6_reward_scale_multiangle.py --scale 0.2 --seed 261855 --tag-suffix stage6mm_scale20 [--iterations 50] [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.run_altitude_attack_followup import RUN_EXPERIMENT, attack_window_metrics, execute
from scripts.run_altitude_native_curriculum import (
    checkpoint_final,
    model_bundle,
    save_rows,
)
from scripts.run_altitude_reward_search import load_yaml, write_yaml
from scripts.run_stage6f_turnback_pilot import overshoot_recovery_metrics, passed_gate

BASE_YAML = (
    ROOT / "artifacts" / "altitude_attack_followup_v1" / "generated"
    / "stage6e_delta_down_400iter.yaml"
)
SOURCE_TAG = "altitude_native_curriculum_v1_stage4_banked_descent_recovery_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6_reward_scale_sweep_results.csv"

WARMUP_STEPS = 2000

_ORIGINAL_DISTANCE_M = math.hypot(530.0, 530.0)  # 749.53..., same as close_quarters_750
_ANGLES_DEG = [0.0, 8.0, 15.0, 22.0]

OWNSHIP_START = [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0]

# stage6e_delta_down_400iter's original, validated engagement-reward values.
# Every other field (safety/control) stays fixed regardless of scale.
FULL_ENGAGEMENT_VALUES = {
    "range_scale": 2.4,
    "overshoot_penalty": 4.0,
    "overshoot_quadratic_scale": 1.5,
    "inside_min_range_penalty": -3.0,
    "ata_scale": 0.15,
    "aa_scale": 0.04,
    "wez_bonus": 1.0,
    "damage_scale": 30.0,  # 2026-08-20: boosted 20->30 per user direction (emphasize damage-dealing)
    "attack_range_bonus": 0.9,
    "far_range_penalty": 1.05,
    "range_delta_scale": 0.03,
    "ata_delta_scale": 0.15,
}

FIXED_SAFETY_VALUES = {
    # 2026-08-20: halved (not zeroed, per user direction) after the timeout
    # win/loss fix -- with terminal reward now correctly scoring a timeout
    # by relative damage (see my_reward_delta_v1.py), the per-step penalty's
    # "hurry up and engage" job is now mostly handled by the terminal signal,
    # but keeping a small nonzero value still discourages needless stalling.
    "step_penalty": -0.0015,
    "survival_bonus": 0.0,
    "too_close_m": 460.0,
    "ideal_range_min_m": 450.0,
    "ideal_range_max_m": 1100.0,
    "altitude_soft_floor_m": 1200.0,
    "altitude_hard_floor_m": 600.0,
    "low_altitude_penalty": 1.15,
    "very_low_altitude_penalty": 3.45,
    "altitude_bonus_high_min_m": 3400.0,
    "altitude_bonus_high_max_m": 9000.0,
    "altitude_bonus_mid_min_m": 1800.0,
    "altitude_bonus_high": 0.5,
    "altitude_bonus_mid": 0.31,
    "nose_down_altitude_m": 4000.0,
    "nose_down_pitch_deg": -10.0,
    "nose_down_penalty": -1.13,
    "roll_limit_deg": 80.0,
    "roll_limit_penalty": 0.15,
    "pitch_down_limit_deg": -12.0,
    "pitch_down_penalty": 1.2,
    "pitch_up_limit_deg": 35.0,
    "pitch_up_penalty": 0.25,
    "attack_range_min_m": 250.0,
    "attack_range_max_m": 2200.0,
    "far_range_penalty_start_m": 2500.0,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "crash_penalty": -280.0,
}

# Original (pre-easing) difficulty from stage6j, the first scripted_pursuit
# opponent this project ever trained against, before stage6m/6n eased it
# down to 0.4/25 because -- on the OLD stage6e-sourced/chase-habit-laden
# base -- it caused instability. Kept here for reference/CLI default
# comparison; ScriptedPursuitActionProvider's own defaults match this.
ORIGINAL_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 1.5,
    "max_bank_deg": 70.0,
}

EASED_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 0.4,
    "max_bank_deg": 25.0,
}


def _target_start(angle_deg: float, start_distance_m: float = _ORIGINAL_DISTANCE_M) -> list[float]:
    return [
        1000.0 + start_distance_m * math.cos(math.radians(angle_deg)),
        0.0 + start_distance_m * math.sin(math.radians(angle_deg)),
        -4500.0,
        0.0,
        0.0,
        90.0,
        250.0,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, required=True,
                         help="Fraction (0.0-1.0+) of stage6e's engagement reward values to apply.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag-suffix", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--opponent-bank-gain", type=float, default=0.4,
        help="ScriptedPursuitActionProvider.heading_to_bank_gain (eased default 0.4; "
             "original/harder pre-easing value was 1.5).",
    )
    parser.add_argument(
        "--opponent-max-bank-deg", type=float, default=25.0,
        help="ScriptedPursuitActionProvider.max_bank_deg (eased default 25.0; "
             "original/harder pre-easing value was 70.0).",
    )
    parser.add_argument(
        "--ideal-range-min-m", type=float, default=450.0,
        help=(
            "reward.ideal_range_min_m. Default 450.0 was the untouched value "
            "across every scale-sweep variant (0-200%%) -- diagnosed 2026-08-19 "
            "as a standoff-distance bug: with ideal_mid=775m sitting well "
            "outside the WEZ's own max_range_m=914.4/min_range_m=152.4 "
            "engagement band, the policy gets its single largest per-step "
            "reward for loitering at ~775m regardless of angle, and only a "
            "small ata bonus for actually pointing at the target -- so it "
            "never commits to closing into WEZ range. Pass a lower value "
            "(e.g. 300.0) to move the peak inside the WEZ band instead."
        ),
    )
    parser.add_argument(
        "--ideal-range-max-m", type=float, default=1100.0,
        help="reward.ideal_range_max_m -- see --ideal-range-min-m.",
    )
    parser.add_argument(
        "--too-close-m", type=float, default=460.0,
        help=(
            "reward.too_close_m -- overshoot penalty threshold. Default 460.0 "
            "is well above the WEZ's own min_range_m=152.4, so the policy is "
            "punished for entering most of the actual WEZ envelope. Pass a "
            "lower value (e.g. 250.0) to only penalize distances tighter than "
            "the WEZ itself needs."
        ),
    )
    parser.add_argument(
        "--train-max-engage-time-s", type=float, default=60.0,
        help=(
            "env.max_engage_time during training (default 60.0, matches "
            "every prior run this session). Diagnosed 2026-08-19: only "
            "~8-9 episodes complete per 100 iterations because most run the "
            "full 60s without a win/loss/crash, so the ~1.4s merge window "
            "that actually determines wez_episode_rate only gets 8-9 "
            "gradient updates worth of terminal-outcome signal per run -- "
            "identical mean_target_damage across several very different "
            "reward configs suggests that window's behavior hasn't moved "
            "at all. Shortening this lets many more episodes (and thus "
            "many more samples of the merge window) complete per iteration "
            "budget."
        ),
    )
    parser.add_argument(
        "--safety-override-enabled", action="store_true",
        help=(
            "Enable single_agent_env.py's hard safety override DURING "
            "TRAINING (not just at eval time). Diagnosed 2026-08-19: every "
            "prior training run this session had this OFF (default False), "
            "but adhoc_scripted_pursuit_eval_stage6j.py hardcodes it ON at "
            "eval time (altitude_m=1500/time_horizon_s=25) -- a train/eval "
            "mismatch where the policy never experienced the override "
            "hijacking its controls (up to 38%% of an episode's steps in "
            "stage6rr_phasewez_100iter's frozen eval, despite worst-case "
            "altitude never dropping below 2765m -- real margin was huge, "
            "the override was just trigger-happy) until eval time."
        ),
    )
    parser.add_argument("--safety-override-altitude-m", type=float, default=700.0)
    parser.add_argument("--safety-override-time-horizon-s", type=float, default=10.0)
    parser.add_argument("--safety-override-hard-floor-m", type=float, default=400.0)
    parser.add_argument(
        "--time-pressure-start-s", type=float, default=0.0,
        help="reward.time_pressure_start_s -- grace period before the stalling penalty ramps up.",
    )
    parser.add_argument(
        "--time-pressure-ramp-s", type=float, default=12.0,
        help="reward.time_pressure_ramp_s -- seconds to ramp from 0 to full penalty.",
    )
    parser.add_argument(
        "--time-pressure-scale", type=float, default=0.0,
        help=(
            "reward.time_pressure_scale -- max per-step penalty (bounded, not "
            "accumulated) applied once ramped, while outside the WEZ. 0.0 "
            "(default) disables it. Added to break the safe-orbit-forever "
            "equilibrium seen in stage6pp_engageband_fix_100iter."
        ),
    )
    parser.add_argument(
        "--ata-delta-scale-override", type=float, default=None,
        help="If set, overrides reward.ata_delta_scale directly instead of scale*FULL_ENGAGEMENT_VALUES['ata_delta_scale'].",
    )
    parser.add_argument(
        "--angles-deg", type=str, default=None,
        help=(
            "Comma-separated angle pool in degrees, overriding the default "
            "4-angle mix [0,8,15,22]. Added 2026-08-20 after a 100-episode "
            "frozen eval of stage6mm_scale100_100iter broke wez_episode_rate "
            "down by scenario angle and found the '5%%' figure was a mix of "
            "two totally different regimes: 0deg genuinely succeeds 52.9%% "
            "of the time (9/17, real policy skill, different wez_steps/"
            "target_health per seed), while 8/15/22deg succeed 0-4.3%% (the "
            "8deg 4.3%% being a single scenario-determined 'free hit' "
            "reproduced identically even with zero policy input -- see "
            "memory). Use e.g. '--angles-deg 0.0' to test whether training "
            "focused entirely on the angle that already works pushes its "
            "success rate even higher, unlocking on the 3/4 of training "
            "budget the 4-angle mix currently spends on angles that never "
            "succeed."
        ),
    )
    parser.add_argument(
        "--start-distance-m", type=float, default=_ORIGINAL_DISTANCE_M,
        help=(
            "Initial ownship-target separation (default 749.53, the "
            "'close_quarters_750' convention used by every stage6 experiment "
            "this project has ever run). Added 2026-08-20 after "
            "adhoc_action_saturation_probe.py showed the RL policy's raw "
            "roll/pitch/yaw commands are FULLY SATURATED (near +-1.0 on "
            "multiple axes simultaneously, for 1+ continuous seconds) during "
            "episode#18's merge window, identically across 6 wildly "
            "different reward configs (rr/ss/tt/uu/vv/ww) that all still "
            "produced the exact same wez_episode_rate=0.05 and "
            "mean_target_damage=0.0001874499999999779 -- i.e. the policy is "
            "already commanding maximum control authority and reward "
            "shaping has no more room to work with. A larger starting "
            "distance gives more time before the merge's closure rate "
            "exceeds the airframe's turn authority, which reward tuning "
            "cannot fix but geometry might."
        ),
    )
    parser.add_argument(
        "--pursuit-heading-scale", type=float, default=0.0,
        help=(
            "reward.pursuit_heading_scale (ported into my_reward_delta_v1.py "
            "2026-08-20). Rewards each step's actual flight-path direction "
            "for pointing at the target's CURRENT position, every step, "
            "independent of whether range/ata have already improved -- "
            "continuous 'turn the nose toward the opponent' instead of a "
            "one-shot merge bet. This was the single best WEZ-engagement "
            "signal this project ever found (2026-08-11, on the old "
            "stage6e/turnback line: crash 0%%, first-ever real WEZ damage "
            "hit) but was never tried on the current Stage4-based safe "
            "foundation. 0.0 (default) disables it; 0.2 is the "
            "historically-validated starting point."
        ),
    )
    parser.add_argument(
        "--ata-scale-override", type=float, default=None,
        help=(
            "If set, overrides reward.ata_scale directly instead of "
            "scale*FULL_ENGAGEMENT_VALUES['ata_scale'] -- use to make pointing "
            "at the target compete with the (still much larger) range-holding "
            "reward instead of being dominated by it."
        ),
    )
    parser.add_argument(
        "--source-tag", type=str, default=None,
        help=(
            "Output tag to restore-checkpoint from, overriding SOURCE_TAG "
            "(the Stage4 banked-descent-recovery checkpoint). Added "
            "2026-08-20 to allow continuing training from a tactical19 "
            "checkpoint (Stage4 is tactical16-shaped and can't be restored "
            "into once --observation-mode is changed)."
        ),
    )
    parser.add_argument(
        "--observation-mode", type=str, default="tactical16",
        choices=["classic12", "relative14", "tactical16", "tactical19"],
        help=(
            "Env observation mode. Default tactical16 (matches every prior "
            "experiment, restorable from SOURCE_TAG). tactical19 (added "
            "2026-08-20) adds the target's relative velocity N/E/D so the "
            "policy can compute a lead-pursuit solution instead of pure "
            "pursuit -- see observation.py's describe_observation. "
            "IMPORTANT: any mode other than tactical16 changes the "
            "network's input layer shape, so --restore-checkpoint is "
            "automatically skipped (fresh random-init training instead) "
            "regardless of SOURCE_TAG."
        ),
    )
    parser.add_argument(
        "--ata-precision-scale", type=float, default=0.0,
        help="reward.ata_precision_scale -- Gaussian bonus peaked at ata=0, see my_reward_delta_v1.py. 0.0 disables.",
    )
    parser.add_argument(
        "--ata-precision-width-deg", type=float, default=8.0,
        help="reward.ata_precision_width_deg -- Gaussian width (std-dev-like) for --ata-precision-scale.",
    )
    parser.add_argument(
        "--evasion-penalty-scale", type=float, default=0.0,
        help=(
            "reward.evasion_penalty_scale -- extra per-step penalty charged "
            "ON TOP of range_delta_scale only while distance is increasing "
            "(never while closing), directly targeting the fleeing/standoff "
            "habit. 0.0 (default) disables."
        ),
    )
    parser.add_argument(
        "--far-range-penalty-span-m", type=float, default=None,
        help=(
            "reward.far_range_penalty_span_m -- distance over which the "
            "far_range penalty ramps from 0 to full past "
            "far_range_penalty_start_m. Default (unset) falls back to "
            "far_range_penalty_start_m*6 (my_reward_delta_v1.py's own "
            "default), which lets a policy drift very far before feeling "
            "much penalty. Pass a smaller span (e.g. 3000) to punish "
            "straying far away much sooner."
        ),
    )
    parser.add_argument(
        "--lead-pursuit-scale", type=float, default=0.0,
        help=(
            "reward.lead_pursuit_scale (2026-08-20) -- rewards the ownship's "
            "nose for pointing at the target's PREDICTED position (current "
            "position + measured per-step displacement * "
            "--lead-pursuit-horizon-steps), not its current position. 0.0 "
            "disables. See student/my_reward_delta_v1.py's config comment."
        ),
    )
    parser.add_argument(
        "--lead-pursuit-horizon-steps", type=float, default=20.0,
        help="reward.lead_pursuit_horizon_steps -- see --lead-pursuit-scale.",
    )
    parser.add_argument(
        "--action-rate-limit", type=float, default=None,
        help=(
            "env_config.action_rate_limit -- max |delta| per env-step on the "
            "ownship's own raw roll/pitch/yaw command (throttle unrestricted). "
            "None (default) disables. Added 2026-08-20: every live-Unreal "
            "test this session showed the policy commanding full +-1.0 "
            "roll/pitch from frame 1 and staying saturated 75-84% of the "
            "flight -- a hard per-step cap (e.g. 0.3) directly prevents that "
            "regardless of what the policy tries to output."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=None)
    return parser.parse_args()


def build_reward_overrides(
    scale: float,
    ideal_range_min_m: float = 450.0,
    ideal_range_max_m: float = 1100.0,
    too_close_m: float = 460.0,
    ata_scale_override: float | None = None,
    ata_delta_scale_override: float | None = None,
    time_pressure_start_s: float = 0.0,
    time_pressure_ramp_s: float = 12.0,
    time_pressure_scale: float = 0.0,
    pursuit_heading_scale: float = 0.0,
    ata_precision_scale: float = 0.0,
    ata_precision_width_deg: float = 8.0,
    evasion_penalty_scale: float = 0.0,
    far_range_penalty_span_m: float | None = None,
    lead_pursuit_scale: float = 0.0,
    lead_pursuit_horizon_steps: float = 20.0,
) -> dict[str, float]:
    overrides = dict(FIXED_SAFETY_VALUES)
    overrides["ideal_range_min_m"] = ideal_range_min_m
    overrides["ideal_range_max_m"] = ideal_range_max_m
    overrides["too_close_m"] = too_close_m
    overrides["time_pressure_start_s"] = time_pressure_start_s
    overrides["time_pressure_ramp_s"] = time_pressure_ramp_s
    overrides["time_pressure_scale"] = time_pressure_scale
    overrides["pursuit_heading_scale"] = pursuit_heading_scale
    overrides["ata_precision_scale"] = ata_precision_scale
    overrides["ata_precision_width_deg"] = ata_precision_width_deg
    overrides["evasion_penalty_scale"] = evasion_penalty_scale
    overrides["lead_pursuit_scale"] = lead_pursuit_scale
    overrides["lead_pursuit_horizon_steps"] = lead_pursuit_horizon_steps
    if far_range_penalty_span_m is not None:
        overrides["far_range_penalty_span_m"] = far_range_penalty_span_m
    for key, full_value in FULL_ENGAGEMENT_VALUES.items():
        overrides[key] = full_value * scale
    if ata_scale_override is not None:
        overrides["ata_scale"] = ata_scale_override
    if ata_delta_scale_override is not None:
        overrides["ata_delta_scale"] = ata_delta_scale_override
    return overrides


def build_variant_experiment(base: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_{args.tag_suffix}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = "student.my_reward_delta_v1"
    experiment["env"]["target_mode"] = "scripted_pursuit"
    experiment["env"]["max_engage_time"] = args.train_max_engage_time_s
    experiment["env_config"]["reward"] = build_reward_overrides(
        args.scale,
        ideal_range_min_m=args.ideal_range_min_m,
        ideal_range_max_m=args.ideal_range_max_m,
        too_close_m=args.too_close_m,
        ata_scale_override=args.ata_scale_override,
        ata_delta_scale_override=args.ata_delta_scale_override,
        time_pressure_start_s=args.time_pressure_start_s,
        time_pressure_ramp_s=args.time_pressure_ramp_s,
        time_pressure_scale=args.time_pressure_scale,
        pursuit_heading_scale=args.pursuit_heading_scale,
        ata_precision_scale=args.ata_precision_scale,
        ata_precision_width_deg=args.ata_precision_width_deg,
        evasion_penalty_scale=args.evasion_penalty_scale,
        far_range_penalty_span_m=args.far_range_penalty_span_m,
        lead_pursuit_scale=args.lead_pursuit_scale,
        lead_pursuit_horizon_steps=args.lead_pursuit_horizon_steps,
    )
    experiment["env"]["observation_mode"] = args.observation_mode
    experiment["env_config"]["observation_mode"] = args.observation_mode
    experiment["env_config"]["target_scripted_pursuit"] = {
        "cruise_altitude_m": 4500.0,
        "heading_to_bank_gain": args.opponent_bank_gain,
        "max_bank_deg": args.opponent_max_bank_deg,
    }
    if args.safety_override_enabled:
        experiment["env_config"]["safety_override_enabled"] = True
        experiment["env_config"]["safety_override_altitude_m"] = args.safety_override_altitude_m
        experiment["env_config"]["safety_override_time_horizon_s"] = args.safety_override_time_horizon_s
        experiment["env_config"]["safety_override_hard_floor_m"] = args.safety_override_hard_floor_m
    if args.action_rate_limit is not None:
        experiment["env_config"]["action_rate_limit"] = args.action_rate_limit

    angles_deg = (
        [float(x) for x in args.angles_deg.split(",")]
        if args.angles_deg is not None
        else _ANGLES_DEG
    )
    scenarios = [
        {
            "name": f"scale{args.scale:.2f}_{angle_deg:04.1f}deg_750",
            "weight": 1.0,
            "ownship": list(OWNSHIP_START),
            "target": _target_start(angle_deg, args.start_distance_m),
        }
        for angle_deg in angles_deg
    ]
    experiment["env_config"]["initial_scenario"]["mode"] = "scenario_pool"
    experiment["env_config"]["initial_scenario"]["scenarios"] = scenarios

    experiment["runtime"]["iterations"] = args.iterations
    experiment["runtime"]["seed"] = args.seed
    if args.source_tag is not None:
        # Explicit continuation from a prior checkpoint (2026-08-20) -- used
        # to keep building on a tactical19 checkpoint, since Stage4 itself
        # is tactical16-shaped and can't be restored into once
        # --observation-mode changes the input layer size.
        experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(args.source_tag))
        experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    elif args.observation_mode == "tactical16":
        experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
        experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    else:
        # Observation shape changed -> Stage4's network input layer no
        # longer matches, so restoring would error. Fresh random-init
        # training instead (2026-08-20, tactical19 rollout) -- no replay
        # warmup needed either since there's no restored buffer to protect.
        experiment["runtime"].pop("restore_checkpoint", None)
    experiment["notes"] = (
        f"reward-scale sweep: scale={args.scale}, seed={args.seed}, "
        f"iterations={args.iterations}, fresh restart from Stage4 "
        f"({SOURCE_TAG}), 4-angle pool (0/8/15/22deg). Engagement reward "
        f"fields scaled to {args.scale*100:.0f}% of stage6e_delta_down_400iter's "
        f"values; safety/control fields fixed. Opponent difficulty: "
        f"heading_to_bank_gain={args.opponent_bank_gain}, "
        f"max_bank_deg={args.opponent_max_bank_deg} "
        f"(eased default 0.4/25, original pre-easing 1.5/70). "
        f"Standoff-band fix (2026-08-19): ideal_range=[{args.ideal_range_min_m},"
        f"{args.ideal_range_max_m}]m (was [450,1100] every prior run), "
        f"too_close_m={args.too_close_m} (was 460), "
        f"ata_scale_override={args.ata_scale_override}. "
        f"Time pressure: start_s={args.time_pressure_start_s}, "
        f"ramp_s={args.time_pressure_ramp_s}, "
        f"scale={args.time_pressure_scale}. "
        f"Pursuit heading (2026-08-20, ported from "
        f"my_reward_delta_v1_wez_proximity.py's 2026-08-11 term, first try "
        f"on the Stage4-based line): pursuit_heading_scale={args.pursuit_heading_scale} "
        f"-- continuous per-step reward for flight-path pointing at the "
        f"target's current position, independent of a clean merge. "
        f"Start distance (2026-08-20, after adhoc_action_saturation_probe.py "
        f"found the RL policy's raw controls FULLY SATURATED during episode "
        f"#18's merge across 6 different reward configs): "
        f"start_distance_m={args.start_distance_m} (default/prior "
        f"convention: {_ORIGINAL_DISTANCE_M:.2f}). "
        f"ATA precision bonus (2026-08-20): ata_precision_scale="
        f"{args.ata_precision_scale}, width_deg={args.ata_precision_width_deg} "
        f"-- Gaussian bonus peaked at ata=0, additive on top of the linear "
        f"ata term. Observation mode (2026-08-20): {args.observation_mode}"
        + (
            " (fresh random-init, no restore_checkpoint -- input shape "
            "differs from Stage4)."
            if args.observation_mode != "tactical16"
            else " (restored from Stage4, unchanged)."
        )
    )
    return tag, experiment


def main() -> int:
    args = parse_args()
    if args.evaluation_window is None:
        args.evaluation_window = args.iterations
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    if not checkpoint_final(SOURCE_TAG).exists():
        raise FileNotFoundError(f"Source checkpoint missing: {checkpoint_final(SOURCE_TAG)}")
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = (
        list(csv.DictReader(RESULT_PATH.open("r", encoding="utf-8-sig", newline="")))
        if RESULT_PATH.exists() else []
    )

    tag, experiment = build_variant_experiment(base, args)
    yaml_path = CAMPAIGN_DIR / "generated" / f"{args.tag_suffix}.yaml"
    write_yaml(yaml_path, experiment)
    code = execute(
        [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"[dry-run] would evaluate {tag}")
        return 0

    gate_metrics = attack_window_metrics(tag, args.evaluation_window)
    recovery_metrics = overshoot_recovery_metrics(tag, args.evaluation_window)
    row = {
        "variant": args.tag_suffix,
        "scale": args.scale,
        "seed": args.seed,
        "iterations": args.iterations,
        "tag": tag,
        "reward_module": "student.my_reward_delta_v1",
        "status": "success" if code == 0 else "failed",
        "replay_warmup_steps": WARMUP_STEPS,
        **gate_metrics,
        **recovery_metrics,
        "gate_passed": passed_gate(gate_metrics) if code == 0 else False,
        "checkpoint_final": str(checkpoint_final(tag)),
        "model_bundle": str(model_bundle(tag)),
    }
    results.append(row)
    save_rows(RESULT_PATH, results)
    print(f"[reward-scale] {tag}: scale={args.scale} gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[reward-scale] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] reward-scale sweep result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
