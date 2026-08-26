"""Frozen (deterministic) JSBSim evaluation of TerminalStabilizerActionProvider
(src/dogfight/ai/terminal_stabilizer.py) -- raw/unmodified RL v8 during
acquisition (the same as run_unreal_inference.py --mode rl --terminal-stabilizer),
with a LOS-rate PD smoothing loop blended in only once ATA/distance are
already small, to remove axis chatter/saturation without replacing RL's own
aim decisions. Only live-tested once before (run0221, 64.6s, cut short).

Same scenario/reward/opponent config as the other adhoc_*_eval.py scripts.

Usage:
  python scripts/adhoc_terminal_stabilizer_eval.py --tag <rl_tag> [--episodes 30]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.terminal_stabilizer import TerminalStabilizerActionProvider, TerminalStabilizerConfig
from dogfight.ai.scripted_pursuit_provider import ScriptedPursuitActionProvider
from dogfight.ai.student_hooks import load_reward_hook
from scripts.run_stage6g_frozen_eval import load_training_env_config

MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
FROZEN_EVAL_ROOT = ROOT / "artifacts" / "altitude_attack_followup_v1" / "frozen_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=269000)
    parser.add_argument("--enter-ata-deg", type=float, default=None, help="Override TerminalStabilizerConfig.enter_ata_deg (default 22.0).")
    parser.add_argument("--exit-ata-deg", type=float, default=None, help="Override TerminalStabilizerConfig.exit_ata_deg (default 35.0).")
    parser.add_argument("--enter-distance-m", type=float, default=None, help="Override TerminalStabilizerConfig.enter_distance_m (default 2200.0).")
    parser.add_argument("--exit-distance-m", type=float, default=None, help="Override TerminalStabilizerConfig.exit_distance_m (default 3000.0).")
    parser.add_argument("--tag-suffix", default="", help="Extra suffix for the output directory, so A/B variants don't overwrite each other.")
    parser.add_argument("--roll-commit-enabled", action="store_true", help="Enable the roll-direction commit filter during raw acquisition.")
    parser.add_argument("--roll-commit-deadband-deg", type=float, default=None)
    parser.add_argument("--roll-commit-hysteresis-deg", type=float, default=None)
    parser.add_argument("--energy-throttle-floor-enabled", action="store_true", help="Force min throttle during acquisition when under corner speed.")
    parser.add_argument("--corner-speed-mps", type=float, default=None)
    parser.add_argument("--energy-throttle-floor", type=float, default=None)
    parser.add_argument("--energy-pitch-unload-enabled", action="store_true", help="Cap pull magnitude when well under corner speed (unload to accelerate).")
    parser.add_argument("--energy-pitch-unload-deficit-mps", type=float, default=None)
    parser.add_argument("--energy-pitch-unload-cap", type=float, default=None)
    parser.add_argument("--energy-pitch-scale-enabled", action="store_true", help="Smooth (continuous) pull taper as speed drops, instead of a hard cap.")
    parser.add_argument("--energy-pitch-scale-deficit-mps", type=float, default=None)
    parser.add_argument("--energy-pitch-scale-floor", type=float, default=None)
    parser.add_argument("--defensive-break-enabled", action="store_true", help="Hard override: break toward the threat when they hold a good angle on us.")
    parser.add_argument("--defensive-break-threat-ata-deg", type=float, default=None)
    parser.add_argument("--defensive-break-range-m", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag
    suffix = "terminal_stabilizer" + (f"_{args.tag_suffix}" if args.tag_suffix else "")
    out_dir = FROZEN_EVAL_ROOT / f"{tag}__scripted_pursuit__{suffix}"
    reward_module, env_config = load_training_env_config(tag)
    reward_fn, _ = load_reward_hook(reward_module)

    env_config["max_engage_time"] = args.max_engage_time
    env_config["safety_override_enabled"] = True
    env_config["safety_override_altitude_m"] = env_config.get("safety_override_altitude_m", 1500.0)
    env_config["safety_override_time_horizon_s"] = env_config.get("safety_override_time_horizon_s", 25.0)

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("episode_summary*.csv"):
        stale.unlink()
    env_config["episode_summary_path"] = str(out_dir / "episode_summary.csv")

    rl_provider = RLActionProvider(
        bundle_dir=MODEL_ROOT / tag,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )
    ts_kwargs = {}
    if args.enter_ata_deg is not None:
        ts_kwargs["enter_ata_deg"] = args.enter_ata_deg
    if args.exit_ata_deg is not None:
        ts_kwargs["exit_ata_deg"] = args.exit_ata_deg
    if args.enter_distance_m is not None:
        ts_kwargs["enter_distance_m"] = args.enter_distance_m
    if args.exit_distance_m is not None:
        ts_kwargs["exit_distance_m"] = args.exit_distance_m
    if args.roll_commit_enabled:
        ts_kwargs["roll_commit_enabled"] = True
    if args.roll_commit_deadband_deg is not None:
        ts_kwargs["roll_commit_deadband_deg"] = args.roll_commit_deadband_deg
    if args.roll_commit_hysteresis_deg is not None:
        ts_kwargs["roll_commit_hysteresis_deg"] = args.roll_commit_hysteresis_deg
    if args.energy_throttle_floor_enabled:
        ts_kwargs["energy_throttle_floor_enabled"] = True
    if args.corner_speed_mps is not None:
        ts_kwargs["corner_speed_mps"] = args.corner_speed_mps
    if args.energy_throttle_floor is not None:
        ts_kwargs["energy_throttle_floor"] = args.energy_throttle_floor
    if args.energy_pitch_unload_enabled:
        ts_kwargs["energy_pitch_unload_enabled"] = True
    if args.energy_pitch_unload_deficit_mps is not None:
        ts_kwargs["energy_pitch_unload_deficit_mps"] = args.energy_pitch_unload_deficit_mps
    if args.energy_pitch_unload_cap is not None:
        ts_kwargs["energy_pitch_unload_cap"] = args.energy_pitch_unload_cap
    if args.energy_pitch_scale_enabled:
        ts_kwargs["energy_pitch_scale_enabled"] = True
    if args.energy_pitch_scale_deficit_mps is not None:
        ts_kwargs["energy_pitch_scale_deficit_mps"] = args.energy_pitch_scale_deficit_mps
    if args.energy_pitch_scale_floor is not None:
        ts_kwargs["energy_pitch_scale_floor"] = args.energy_pitch_scale_floor
    if args.defensive_break_enabled:
        ts_kwargs["defensive_break_enabled"] = True
    if args.defensive_break_threat_ata_deg is not None:
        ts_kwargs["defensive_break_threat_ata_deg"] = args.defensive_break_threat_ata_deg
    if args.defensive_break_range_m is not None:
        ts_kwargs["defensive_break_range_m"] = args.defensive_break_range_m
    ownship_provider = TerminalStabilizerActionProvider(rl_provider, TerminalStabilizerConfig(**ts_kwargs))
    target_provider = ScriptedPursuitActionProvider(**pursuit_cfg)
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=ownship_provider,
        target_action_provider=target_provider,
        reward_fn=reward_fn,
    )
    active_steps = 0
    total_steps = 0
    try:
        for episode_number in range(1, args.episodes + 1):
            env.reset(seed=args.seed + episode_number)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
                total_steps += 1
                if ownship_provider._active:
                    active_steps += 1
            print(f"[terminal-stabilizer-eval] episode {episode_number}/{args.episodes}")
    finally:
        env.close()

    paths = sorted(out_dir.glob("episode_summary*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    def num(row: dict[str, str], key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    crashes = sum(row.get("outcome") == "crash" for row in rows)
    wins = sum(row.get("outcome") == "win" for row in rows)
    losses = sum(row.get("outcome") == "loss" for row in rows)
    draws = sum(row.get("outcome") == "draw" for row in rows)
    altitudes = [v for row in rows if (v := num(row, "minimum_altitude_m")) is not None]
    distances = [v for row in rows if (v := num(row, "mean_distance_m")) is not None]
    min_distances = [v for row in rows if (v := num(row, "min_distance_m")) is not None]
    wez_steps = [v for row in rows if (v := num(row, "wez_steps")) is not None]
    target_healths = [v for row in rows if (v := num(row, "target_health")) is not None]

    print(f"\n[done] mode={suffix} episodes={len(rows)} crash={crashes} win={wins} loss={losses} draw={draws}")
    print(f"  minimum_altitude_mean_m={sum(altitudes)/len(altitudes) if altitudes else None:.1f}")
    print(f"  minimum_altitude_worst_m={min(altitudes) if altitudes else None}")
    print(f"  mean_distance_m={sum(distances)/len(distances) if distances else None:.1f}")
    print(f"  min_distance_mean_m={sum(min_distances)/len(min_distances) if min_distances else None:.1f}")
    print(f"  wez_episode_rate={sum(v > 0.0 for v in wez_steps)/len(wez_steps) if wez_steps else None}")
    print(f"  mean_target_damage={1 - sum(target_healths)/len(target_healths) if target_healths else None}")
    if total_steps:
        print(f"  terminal_stabilizer_active_fraction={active_steps/total_steps:.3f}")
    print()
    for i, row in enumerate(rows, 1):
        print(i, row.get("scenario_name"), row.get("outcome"), row.get("end_condition"),
              row.get("minimum_altitude_m"), row.get("min_distance_m"), row.get("mean_distance_m"),
              row.get("wez_steps"), row.get("target_health"), row.get("ownship_health"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
