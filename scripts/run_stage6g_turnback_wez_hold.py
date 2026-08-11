"""Test the new wez_hold duration-aware shaping term (added 2026-08-10 to
my_reward_delta_v1_wez_proximity.py) against the "approach then flee" pattern.

Background: a 20-episode frozen (deterministic, explore=False) evaluation of
turnback_wez_proximity_aa_150iter found the policy reliably closes to
450-760m (well inside attack_range_min/max_m 250-1500m) every single episode,
but then drifts back out to 14000-23000m by episode end instead of holding
position long enough to also align ata -- range and ata alignment are
achieved sequentially, never simultaneously, so wez_episode_rate stayed 0%
across all 20 tries despite always getting close. Bumping wez_proximity_scale
0.5 -> 0.8 (turnback_wez_proximity_scale08_50iter) made this WORSE, not
better: crash dropped to 0% but mean_distance nearly tripled (5334m ->
14753m) and final_distance regularly hit 20000-46000m -- the policy became
even more committed to a single close pass followed by a long retreat.

Hypothesis for why: range_delta/ata_delta (already in this reward stack) pay
for the *motion* of closing, not the *state* of being close -- once distance
stops decreasing, delta reward drops to ~0, but retreating and re-approaching
earns a fresh burst of positive delta reward again. Repeated
bail-and-reapproach cycles could therefore earn *more* cumulative reward than
settling into one hold, directly explaining the observed pattern. The new
wez_hold term is designed to counter exactly this: it rewards consecutive
steps spent in a good range x ata (x aa) "quality" state with a bonus that
*grows* with streak length (capped), and applies a one-time penalty when an
established streak (>= wez_hold_break_min_streak) is broken -- making
bail-and-reapproach cost something instead of being free. Full design
rationale and a synthetic-scenario sanity check are in
my_reward_delta_v1_wez_proximity.py's module docstring and this session's
transcript.

This restarts from turnback_wez_proximity_aa_150iter (not the scale08
checkpoint, which had already learned the more-committed-to-fleeing
behavior) with wez_proximity_scale reverted to its last non-regressed value
(0.5) plus wez_hold_scale=0.4 / wez_hold_break_penalty_scale=0.3 (modest,
same order of magnitude as wez_proximity_scale -- this project has a
documented precedent for shaping-reward magnitude increases turning into
exploits when pushed too far, see my_reward_delta_v1.py's 2026-08-06
dive-exploit fix note, so start conservative and re-verify in small
increments per rule 13). Both restore fixes apply automatically
(--replay-warmup-steps=2000, auto-inferred --initial-alpha).

Usage: python scripts/run_stage6g_turnback_wez_hold.py [--dry-run]
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6g_turnback_wez_proximity_aa_150iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261810  # next unused seed (261809 used by the scale08 run).

VARIANT = {
    "suffix": "turnback_wez_hold_50iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.5,  # reverted from the regressed 0.8 trial
        "wez_proximity_use_aa": True,
        "wez_hold_scale": 0.4,
        "wez_hold_break_penalty_scale": 0.3,
    },
    "notes": (
        "First test of the new wez_hold duration-aware shaping term (streak "
        "bonus for consecutive steps in good range x ata x aa geometry, "
        "one-time penalty for breaking an established streak), restarted "
        "from turnback_wez_proximity_aa_150iter with wez_proximity_scale "
        "reverted to 0.5. Targets the diagnosed 'approach then flee' pattern "
        "(0% wez_episode_rate across 20 frozen episodes despite always "
        "closing to 450-760m) that a simple wez_proximity_scale increase "
        "(0.5->0.8) made worse rather than better."
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
    print(f"[done] Stage6g wez_hold result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
