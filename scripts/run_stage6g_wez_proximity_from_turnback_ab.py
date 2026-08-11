"""Stage6g wez_proximity, restarted from the safest checkpoint instead of the
already-fragile stage6g_wez_proximity_100iter line.

Background: three consecutive continuations of stage6g_wez_proximity_100iter
(wez_proximity_150iter, warmup_50iter/warmup_aa_50iter, alpha_fix_50iter/
alpha_fix_aa_50iter) all collapsed to ~100% crash on the same close_quarters_750
scenario, with the same low_altitude_crash/altitude_nose_down_crash signature
(minimum altitude ~285-300m). The alpha_fix pair proved this is NOT a training-
dynamics artifact: with both the replay-buffer warmup fix and the SAC
temperature (alpha) restore fix in place, actor_loss stayed bounded (437-610,
no divergence) and alpha correctly continued its natural decay from the source
run's own converged 0.309 -- yet crash rate was unchanged (100%). It was also
not the aa multiplicative term: the no-aa and aa variants failed identically in
every pairing. Conclusion: stage6g_wez_proximity_100iter itself is a fragile
checkpoint (its own original eval was already 40% crash, far higher than any
other Stage6 candidate) and continuing to train it, with or without the two
restore fixes, keeps reproducing that fragility rather than curing it.

Pivot: apply the same wez_proximity idea (dense range x ata joint shaping,
optionally x aa) to `stage6f_turnback_200iter` instead -- this project's
safest checkpoint that still retains real attack-approach behavior (0% crash
over 12 episodes, clears 3/4 Stage6 gates, only wez_episode_rate falls short).
Both restore fixes are active automatically (train_rllib.py auto-infers
--initial-alpha from this source's own training_log.csv, which converged to
alpha=0.226; --replay-warmup-steps=2000 delays learner updates ~20 iterations
post-restore). Same reward recipe and iteration budget as every prior
stage6g pilot (rule 13: small increments, re-verify each time) for direct
comparability.

Two variants, same source checkpoint/seed (rule 7, fair A/B):
  A (turnback_wez_proximity_50iter):    my_reward_delta_v1_wez_proximity,
                                         range x ata joint term only.
  B (turnback_wez_proximity_aa_50iter): same + wez_proximity_use_aa=True
                                         (range x ata x aa joint term).

Usage: python scripts/run_stage6g_wez_proximity_from_turnback_ab.py [--dry-run]
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6f_turnback_200iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261806  # next unused seed in the stage6f/6g incrementing convention
# (261800/261801 turnback pilot+continue, 261803 wez_proximity pair,
# 261804 noisecheck, 261805 warmup + alpha_fix pairs).

VARIANTS = [
    {
        "suffix": "turnback_wez_proximity_50iter",
        "reward_module": "student.my_reward_delta_v1_wez_proximity",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
            "wez_proximity_scale": 0.5,
        },
        "notes": (
            "Stage6g wez_proximity restarted from stage6f_turnback_200iter "
            "(0% crash, safest Stage6 checkpoint) instead of the fragile "
            "stage6g_wez_proximity_100iter line. Both restore fixes active: "
            "--replay-warmup-steps=2000 and auto-inferred --initial-alpha "
            "(source converged to 0.226). range x ata joint term only."
        ),
    },
    {
        "suffix": "turnback_wez_proximity_aa_50iter",
        "reward_module": "student.my_reward_delta_v1_wez_proximity",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
            "wez_proximity_scale": 0.5,
            "wez_proximity_use_aa": True,
        },
        "notes": (
            "Same as turnback_wez_proximity_50iter but the wez_proximity "
            "joint term also requires aa alignment (range x ata x aa). Same "
            "seed/iterations/warmup for a fair A/B against the no-aa variant."
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
    # initial_alpha deliberately left unset: train_rllib.py auto-infers it
    # from SOURCE_TAG's own training_log.csv (--auto-restore-alpha, default on).
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
        print(f"[done] Stage6g from-turnback A/B results appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
