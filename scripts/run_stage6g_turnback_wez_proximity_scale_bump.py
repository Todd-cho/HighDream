"""Bump wez_proximity_scale (0.5 -> 0.8) on top of stage6g_turnback_wez_proximity_aa_150iter,
to test whether stronger joint-condition shaping fixes the "always approaches
but never holds/aligns" pattern the frozen evaluation exposed.

Background: a 20-episode frozen (deterministic, explore=False) evaluation of
turnback_wez_proximity_aa_150iter (run_stage6g_frozen_eval.py) gave a much
cleaner read than any of the noisy 4-6 episode training windows: crash_rate
25% (5/20, right at the Stage6 gate boundary), wez_episode_rate 0% (zero true
WEZ engagements across all 20 tries). Every single episode closed to
450-760m (well inside the attack_range_min/max_m 250-1500m band) but then
drifted back out to 14000-23000m by episode end instead of holding position
long enough to also align ata -- range and ata alignment are being achieved
sequentially, never simultaneously, which is exactly the joint-condition gap
wez_proximity shaping was designed to close in the first place (see
my_reward_delta_v1_wez_proximity.py's docstring). Three separate 50iter
continuations (50/100/150iter) all reproduced this same "approach then flee"
shape without landing the joint condition, and actor_loss/alpha stayed
healthy throughout (no divergence) -- so this is not a training-duration or
restore-bug problem, it looks like the shaping magnitude simply isn't strong
enough yet to make holding+aligning worth more than closing distance alone.

This run raises wez_proximity_scale from 0.5 to 0.8 (a ~60% increase, kept
deliberately modest rather than a big jump -- this project has a documented
precedent for shaping-reward magnitude increases turning into reward-hacking
exploits when pushed too far, see my_reward_delta_v1.py's 2026-08-06
dive-exploit fix note) while keeping every other reward term, the altitude
safety gate, and wez_proximity_use_aa=True unchanged. Both restore fixes
apply automatically (--replay-warmup-steps=2000, auto-inferred
--initial-alpha from the source run's own converged alpha).

Usage: python scripts/run_stage6g_turnback_wez_proximity_scale_bump.py [--dry-run]
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
SEED = 261809  # next unused seed (261808 used by the 100->150iter continuation).

VARIANT = {
    "suffix": "turnback_wez_proximity_scale08_50iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.8,
        "wez_proximity_use_aa": True,
    },
    "notes": (
        "50iter continuation from stage6g_turnback_wez_proximity_aa_150iter "
        "with wez_proximity_scale raised 0.5 -> 0.8. The 150iter frozen eval "
        "(20 deterministic episodes) showed the policy always closes to "
        "450-760m but never holds/aligns long enough for a true WEZ episode "
        "(wez_episode_rate 0%, crash 25%). Testing whether stronger joint "
        "range x ata x aa shaping makes holding position worth more than "
        "closing and leaving."
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
    print(f"[done] Stage6g scale-bump result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
