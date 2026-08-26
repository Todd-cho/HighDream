"""Stage6g alpha-fix A/B: does fixing SAC's temperature (alpha) reset on
restore -- on top of the replay-buffer warmup fix -- finally prevent the
crash collapse that hit both stage6g_wez_proximity_150iter (no fixes) and
stage6g_warmup_50iter/stage6g_warmup_aa_50iter (warmup only, still 100%
crash, 6/6, both variants)?

Background: the warmup-only run (run_stage6g_wez_proximity_warmup_ab.py)
confirmed the buffer-refill warmup executed exactly as designed (actor_loss
stayed nan through iter18, first real update at iter19 exactly when
replay_buffer_size hit 2000) but actor_loss still diverged monotonically from
that very first update (545 -> -362 by iter49), and both variants ended 100%
crash (6/6) with the same failure signature (low_altitude_crash, minimum
altitude ~290-294m) as the earlier unwarmed collapse.

Comparing training_log.csv of the restored runs against the SOURCE
checkpoint's (stage6g_wez_proximity_100iter) own last iteration found the
likely deeper cause: that source run had converged alpha down to 0.309 by its
own iter99, but every continuation (warmed up or not) restarts alpha at 1.0.
Root cause (confirmed by reading RLlib 2.54 source,
ray/rllib/algorithms/sac/sac_learner.py + torch_learner.py): SAC's temperature
(curr_log_alpha) is a raw tensor owned directly by the Learner, not a
parameter of the RLModule and not captured by torch's optimizer.state_dict()
(which only holds per-parameter Adam momentum buffers, not values) --
Algorithm.restore() has no path that ever restores this value, so it silently
resets to config.initial_alpha (1.0) every time, reintroducing ~3x the
entropy pressure the actor/critic weights were actually tuned for right as
training resumes.

train_rllib.py now auto-infers --initial-alpha from the restored checkpoint's
own training_log.csv when --restore-checkpoint is used and --initial-alpha is
not explicitly set (see --auto-restore-alpha, default on) -- no YAML/script
change needed to pick this up, so this script is otherwise identical to
run_stage6g_wez_proximity_warmup_ab.py (same source checkpoint, same seed,
same 50iter/2000-step-warmup budget) with new output tags so the prior
(collapsed) warmup-only run's data is preserved for comparison.

Usage: python scripts/run_stage6g_wez_proximity_alpha_fix_ab.py [--dry-run]
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
SEED = 261805  # same seed as the warmup-only pair, for direct comparability.

VARIANTS = [
    {
        "suffix": "alpha_fix_50iter",
        "reward_module": "student.my_reward_delta_v1_wez_proximity",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
            "wez_proximity_scale": 0.5,
        },
        "notes": (
            "Stage6g alpha-fix check - 50iter continuation from "
            "stage6g_wez_proximity_100iter with --replay-warmup-steps=2000 "
            "AND auto-restored initial_alpha (source run's own converged "
            "alpha=0.309 instead of resetting to 1.0). Same reward params as "
            "stage6g_warmup_50iter, which still collapsed to 100% crash with "
            "the buffer warmup alone. Isolates whether the alpha-reset fix is "
            "what was actually missing."
        ),
    },
    {
        "suffix": "alpha_fix_aa_50iter",
        "reward_module": "student.my_reward_delta_v1_wez_proximity",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
            "wez_proximity_scale": 0.5,
            "wez_proximity_use_aa": True,
        },
        "notes": (
            "Stage6g alpha-fix+aa check - same as alpha_fix_50iter but the "
            "wez_proximity joint term also requires aa alignment (range x "
            "ata x aa). If alpha_fix_50iter shows the collapse is actually "
            "resolved, this isolates whether folding aa into the joint "
            "condition helps aa stability without regressing crash/"
            "wez_episode_rate."
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
        print(f"[done] Stage6g alpha-fix A/B results appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
