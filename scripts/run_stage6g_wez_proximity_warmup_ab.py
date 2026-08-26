"""Stage6g warmup A/B: does the replay-buffer warmup fix (train_rllib.py
--replay-warmup-steps, added 2026-08-10) prevent the crash collapse that hit
stage6g_wez_proximity_150iter, and does folding aa into the wez_proximity
joint term (range x ata x aa instead of range x ata) help further?

Background: stage6g_wez_proximity_100iter got wez_episode_rate to 0.10 (this
project's first real WEZ engagement, episode 7) but a 50iter continuation
(stage6g_wez_proximity_150iter, no warmup) collapsed to crash 100%(4/4). Root
cause traced to a Ray 2.54 new-API-stack gap: Algorithm.restore() restores
policy/critic weights but never the SAC local_replay_buffer contents, and
RLlib's own num_steps_sampled_before_learning_starts warmup gate is keyed off
lifetime env steps (which the restored checkpoint already exceeds via its
restored metrics logger state), so it never re-arms after a restore either --
training resumes into a near-empty buffer immediately. train_rllib.py now
accepts --replay-warmup-steps N, which raises that gate relative to the
restore-time lifetime step count so the buffer gets N fresh steps of
rollout-only refill before learner updates resume.

At this run's ~100 env steps/iteration throughput (see training_log.csv of
prior stage6g runs), WARMUP_STEPS=2000 buys roughly the first 20 of 50
iterations as pure buffer refill, leaving ~30 iterations of actual learning
on a partially-refilled buffer to observe whether crash behavior stays sane.
This is deliberately a small pilot (rule 13: verify in small increments, not
a single large jump) -- if this passes cleanly, a longer follow-up with more
post-warmup training iterations is the next step, not a bigger warmup value.

Two variants, same source checkpoint/seed so it is a fair A/B (rule 7):
  A (warmup_50iter):    existing my_reward_delta_v1_wez_proximity, unchanged
                         reward params (range x ata joint term only) + warmup.
                         Isolates "does warmup alone stop the collapse".
  B (warmup_aa_50iter):  same, plus wez_proximity_use_aa=True (range x ata x
                         aa joint term, added 2026-08-10). Isolates whether
                         requiring aa alignment too helps AA stabilize
                         without hurting crash/wez_episode_rate.

Usage: python scripts/run_stage6g_wez_proximity_warmup_ab.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6g_wez_proximity_100iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261805  # stage6g control/wez_proximity 100iter pair used 261803, noisecheck used 261804.

VARIANTS = [
    {
        "suffix": "warmup_50iter",
        "reward_module": "student.my_reward_delta_v1_wez_proximity",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
            "wez_proximity_scale": 0.5,
        },
        "notes": (
            "Stage6g warmup check - 50iter continuation from "
            "stage6g_wez_proximity_100iter with --replay-warmup-steps=2000, "
            "same reward params as the 150iter run that collapsed (crash "
            "0%->100%) without warmup. Isolates whether the replay-buffer "
            "warmup fix alone prevents the collapse."
        ),
    },
    {
        "suffix": "warmup_aa_50iter",
        "reward_module": "student.my_reward_delta_v1_wez_proximity",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
            "wez_proximity_scale": 0.5,
            "wez_proximity_use_aa": True,
        },
        "notes": (
            "Stage6g warmup+aa check - same as warmup_50iter but the "
            "wez_proximity joint term also requires aa alignment (range x "
            "ata x aa instead of range x ata), same seed/iterations/warmup. "
            "Isolates whether folding aa into the joint condition improves "
            "aa stability without regressing crash/wez_episode_rate."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_variant_experiment(base: dict[str, Any], variant: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_stage6g_{variant['suffix']}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = variant["reward_module"]
    experiment["env_config"]["reward"].update(variant["reward_overrides"])
    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    experiment["notes"] = variant["notes"]
    return tag, experiment


def main() -> int:
    args = parse_args()
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    if not checkpoint_final(SOURCE_TAG).exists():
        raise FileNotFoundError(f"Source checkpoint missing: {checkpoint_final(SOURCE_TAG)}")
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        tag, experiment = build_variant_experiment(base, variant)
        yaml_path = CAMPAIGN_DIR / "generated" / f"stage6g_{variant['suffix']}.yaml"
        write_yaml(yaml_path, experiment)
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
            "variant": variant["suffix"],
            "tag": tag,
            "reward_module": variant["reward_module"],
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

    if not args.dry_run:
        print(f"[done] Stage6g warmup A/B results appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
