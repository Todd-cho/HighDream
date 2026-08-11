"""Continue stage6g_turnback_pursuit_heading_50iter for 50 more iterations
(cumulative 100iter from stage6f_turnback_200iter), to get a bigger post-warmup
episode sample before judging gate status.

Background: the first pursuit_heading pilot (run_stage6g_turnback_pursuit_heading.py)
added a new directional-shaping term (this step's actual flight-path direction
vs. the target's live position, +-scale bounded) on top of turnback_wez_hold_50iter.
Its frozen (deterministic, 20-episode) evaluation came closer to the Stage6
safe_wez_geometry gate than anything else this session: crash_rate=0.0,
minimum_altitude_mean_m=4207 (gate >=1500 passed), mean_distance_m=7512.6
(gate <=7500, missed by only 12.6m), wez_episode_rate=0.05 (gate >=0.10,
half the target) -- and this line's first real WEZ damage engagement
(episode 15, target_health=0.975). Both restore fixes are active
(--replay-warmup-steps=2000, auto-inferred --initial-alpha). Per rule 13
(small increments, re-verify every step -- a 200iter jump collapsed
stage6f_turnback and stage6g_wez_proximity_100iter in this same session),
this extends the same line by another 50 iterations rather than jumping
further, then requires a fresh frozen eval before any gate-passed judgment.

Usage: python scripts/run_stage6g_turnback_pursuit_heading_continue.py [--dry-run]
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6g_turnback_pursuit_heading_50iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261813  # next unused seed (261812 used by the pursuit_heading_50iter pilot).
MAX_ENGAGE_TIME_S = 120.0  # unchanged from the 50iter pilot.

VARIANT = {
    "suffix": "turnback_pursuit_heading_100iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.5,
        "wez_proximity_use_aa": True,
        "wez_hold_scale": 0.4,
        "wez_hold_break_penalty_scale": 0.3,
        "pursuit_heading_scale": 0.2,
    },
    "notes": (
        "50iter continuation from stage6g_turnback_pursuit_heading_50iter "
        "(cumulative 100iter from stage6f_turnback_200iter), same reward "
        "params (unchanged pursuit_heading_scale=0.2 on top of wez_hold/"
        "wez_proximity). The 50iter pilot's frozen eval landed closest to "
        "the Stage6 gate this session (mean_distance 7512.6m vs 7500 target, "
        "wez_episode_rate 0.05 vs 0.10 target) with 0% crash and this line's "
        "first real WEZ damage hit -- extending by a small increment (rule 13) "
        "to see whether wez_episode_rate/mean_distance keep improving without "
        "regressing crash rate."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_variant_experiment(base: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_stage6g_{VARIANT['suffix']}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = VARIANT["reward_module"]
    experiment["env_config"]["reward"].update(VARIANT["reward_overrides"])
    experiment["env_config"]["max_engage_time"] = MAX_ENGAGE_TIME_S
    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    # initial_alpha deliberately left unset: train_rllib.py auto-infers it
    # from SOURCE_TAG's own training_log.csv (--auto-restore-alpha, default on).
    experiment["notes"] = VARIANT["notes"]
    return tag, experiment


def main() -> int:
    args = parse_args()
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    if not checkpoint_final(SOURCE_TAG).exists():
        raise FileNotFoundError(f"Source checkpoint missing: {checkpoint_final(SOURCE_TAG)}")
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = (
        list(csv.DictReader(RESULT_PATH.open("r", encoding="utf-8-sig", newline="")))
        if RESULT_PATH.exists() else []
    )

    tag, experiment = build_variant_experiment(base)
    yaml_path = CAMPAIGN_DIR / "generated" / f"stage6g_{VARIANT['suffix']}.yaml"
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
        "variant": VARIANT["suffix"],
        "tag": tag,
        "reward_module": VARIANT["reward_module"],
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
    print(f"[stage6g] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print(f"[done] Stage6g turnback_pursuit_heading continuation appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
