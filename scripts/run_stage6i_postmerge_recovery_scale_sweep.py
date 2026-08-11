"""Fine-grained sweep of post_merge_climb_scale around the 0.15 sweet spot,
fresh from stage6e_delta_down_400iter for each point (same pattern as the
0.15 run that first found the sweet spot), followed immediately by frozen
eval at both 120s and 200s for each.

Background: three points collected so far (0.0 = stage6e_delta_down_400iter
original, 0.15, 0.4) show climb_scale is NOT a simple safety<->firepower
tradeoff -- 0.0->0.15 trades wez_episode_rate (60-70% -> 30%) for crash
(10-25% -> 0%), but 0.15->0.4 loses on BOTH axes (wez_rate -> 0%, crash
eventually collapses to 100% at 200s with more training) because too strong
a bonus makes "avoid merging entirely" the dominant strategy. This sweeps a
finer grid between 0.0 and 0.4, closer to 0.15, to see whether wez_episode_rate
can be pushed higher than 30% without reintroducing crash risk.

Each point is a fresh 50iter run from stage6e_delta_down_400iter (not chained
off each other, for a clean single-variable comparison against both 0.0 and
0.4), same reward config otherwise, same evaluation protocol (frozen 20
episodes at both --max-engage-time 120 (trained value) and 200 (real
competition value)).

Usage: python scripts/run_stage6i_postmerge_recovery_scale_sweep.py [--dry-run] [--scales 0.10,0.125,0.175,0.20]
"""
from __future__ import annotations

import argparse
import copy
import csv
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6e_delta_down_400iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6i_ab_results.csv"
FROZEN_EVAL_SCRIPT = ROOT / "scripts" / "run_stage6g_frozen_eval.py"

ITERATIONS = 50
WARMUP_STEPS = 2000
# next unused seeds in the 2618xx family (261818 used by the scale015 pilot).
FIRST_SEED = 261819

BASE_REWARD_OVERRIDES: dict[str, Any] = {
    "step_penalty": -0.003,
    "survival_bonus": 0.0,
    "too_close_m": 460.0,
    "ideal_range_min_m": 450.0,
    "ideal_range_max_m": 1100.0,
    "range_scale": 2.4,
    "overshoot_penalty": 4.0,
    "overshoot_quadratic_scale": 1.5,
    "inside_min_range_penalty": -3.0,
    "ata_scale": 0.15,
    "aa_scale": 0.04,
    "wez_bonus": 1.0,
    "damage_scale": 20.0,
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
    "attack_range_bonus": 0.9,
    "far_range_penalty_start_m": 2500.0,
    "far_range_penalty": 1.05,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "crash_penalty": -280.0,
    "range_delta_scale": 0.03,
    "ata_delta_scale": 0.15,
    "post_merge_trigger_m": 150.0,
    "post_merge_recovery_window_steps": 150,
    "post_merge_climb_clip_m": 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scales", type=str, default="0.10,0.125,0.175,0.20",
        help="Comma-separated post_merge_climb_scale values to sweep.",
    )
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def suffix_for(scale: float) -> str:
    return f"stage6i_postmerge_recovery_scale{round(scale * 1000):04d}_50iter"


def build_variant_experiment(base: dict[str, Any], scale: float, seed: int) -> tuple[str, dict[str, Any]]:
    suffix = suffix_for(scale)
    tag = f"altitude_attack_followup_v1_{suffix}_C10"
    overrides = dict(BASE_REWARD_OVERRIDES)
    overrides["post_merge_climb_scale"] = scale
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = "student.my_reward_delta_v1_postmerge_recovery"
    experiment["env_config"]["reward"] = overrides
    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = seed
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    experiment["notes"] = (
        f"Fine-grained sweep point: post_merge_climb_scale={scale} (fresh 50iter "
        "from stage6e_delta_down_400iter, same protocol as the 0.15 sweet-spot run). "
        "Everything else identical to stage6e_delta_down_400iter's reward config."
    )
    return tag, experiment


def main() -> int:
    args = parse_args()
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    if not checkpoint_final(SOURCE_TAG).exists():
        raise FileNotFoundError(f"Source checkpoint missing: {checkpoint_final(SOURCE_TAG)}")
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = (
        list(csv.DictReader(RESULT_PATH.open("r", encoding="utf-8-sig", newline="")))
        if RESULT_PATH.exists() else []
    )

    for i, scale in enumerate(scales):
        seed = FIRST_SEED + i
        tag, experiment = build_variant_experiment(base, scale, seed)
        suffix = suffix_for(scale)
        yaml_path = CAMPAIGN_DIR / "generated" / f"{suffix}.yaml"
        write_yaml(yaml_path, experiment)
        print(f"\n=== sweep point: post_merge_climb_scale={scale} (seed={seed}) tag={tag} ===")
        code = execute(
            [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)],
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"[dry-run] would evaluate {tag}")
            continue

        gate_metrics = attack_window_metrics(tag, args.evaluation_window)
        recovery_metrics = overshoot_recovery_metrics(tag, args.evaluation_window)
        row = {
            "variant": suffix,
            "tag": tag,
            "reward_module": "student.my_reward_delta_v1_postmerge_recovery",
            "status": "success" if code == 0 else "failed",
            "replay_warmup_steps": WARMUP_STEPS,
            "post_merge_climb_scale": scale,
            **gate_metrics,
            **recovery_metrics,
            "gate_passed": passed_gate(gate_metrics) if code == 0 else False,
            "checkpoint_final": str(checkpoint_final(tag)),
            "model_bundle": str(model_bundle(tag)),
        }
        results.append(row)
        save_rows(RESULT_PATH, results)
        print(f"[stage6i-sweep] {tag}: gate_passed={row['gate_passed']} "
              f"wez_episode_rate={row.get('wez_episode_rate')} "
              f"episode_crash_rate={row.get('episode_crash_rate')}")

        # Frozen eval at both the trained (120s) and real-competition (200s)
        # engagement time, matching the protocol used for every other candidate.
        execute([sys.executable, str(FROZEN_EVAL_SCRIPT), "--tag", tag, "--episodes", "20"], dry_run=False)
        execute(
            [sys.executable, str(FROZEN_EVAL_SCRIPT), "--tag", tag, "--episodes", "20",
             "--max-engage-time", "200.0"],
            dry_run=False,
        )

    if not args.dry_run:
        print(f"\n[done] Stage6i climb_scale sweep results appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
