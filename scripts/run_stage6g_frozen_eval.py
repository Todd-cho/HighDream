"""Frozen (deterministic, no exploration noise) evaluation of a Stage6g
turnback_wez_proximity lightweight bundle on its own training scenario.

Background: every stage6g continuation so far has only ever been judged on
its own *training* episodes (SAC's stochastic/exploring rollout, 4-6
episodes per 50iter window). Crash rate has bounced around a lot between
windows (50iter: 2/6, 100iter: 0/4, 150iter: 3/5) with actor_loss/alpha
staying healthy throughout (no divergence) -- consistent with sampling noise
from exploration rather than a real regression, but that has never actually
been checked. This script freezes the policy (RLActionProvider explore=False
-> action_dist.to_deterministic(), see src/dogfight/ai/rl_action_provider.py)
and reruns it on the exact same scenario/reward config the checkpoint was
trained under (close_quarters_750, student.my_reward_delta_v1_wez_proximity
with the turnback+wez_proximity overrides), for a much larger episode count,
to get a noise-free read on crash rate and wez_episode_rate that isn't
contaminated by exploration.

Unlike scripts/evaluate_altitude_bundle.py (which is hardcoded to the
Stage4 curriculum's level_5000 scenario pool and plain student.my_reward --
wrong scenario/reward for this Stage6 attack-approach lineage), this script
reads the exact env/env_config the target bundle was trained with directly
from its own generated experiment YAML in
artifacts/altitude_attack_followup_v1/generated/, so scenario geometry and
reward weights always match the checkpoint being evaluated -- no need to
hand-copy config that could drift out of sync.

Usage:
  python scripts/run_stage6g_frozen_eval.py --tag <output_tag> [--episodes 20] [--seed 269000]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_reward_hook

GENERATED_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1" / "generated"
MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
FROZEN_EVAL_ROOT = ROOT / "artifacts" / "altitude_attack_followup_v1" / "frozen_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", required=True,
        help="Output tag of the run to evaluate, e.g. "
             "altitude_attack_followup_v1_stage6g_turnback_wez_proximity_aa_150iter_C10",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=269000)
    parser.add_argument(
        "--max-engage-time", type=float, default=None,
        help="Override max_engage_time (seconds) for this eval only, e.g. 200.0 "
             "for the real competition value. Default: use the tag's trained value. "
             "Output is written to a separate <tag>__engageNNN subdirectory so it "
             "never overwrites the trained-value frozen eval.",
    )
    parser.add_argument(
        "--target-mode", type=str, default=None,
        help="Override target_mode for this eval only, e.g. 'behavior_tree' to "
             "test against the active AIP_BASE_target.dll AI instead of the "
             "trained-value 'loiter' (passive, predictable) target. Output goes "
             "to a <tag>__target<mode> subdirectory.",
    )
    parser.add_argument(
        "--safety-override", action="store_true",
        help="Enable the hard, non-learned recovery override "
             "(single_agent_env.py._apply_safety_override): forces wings-level "
             "pull-up + full throttle whenever altitude < --safety-override-altitude-m "
             "and pitch <= --safety-override-pitch-deg. Off by default (matches "
             "training). Output goes to a <tag>__safety subdirectory.",
    )
    parser.add_argument("--safety-override-altitude-m", type=float, default=500.0)
    parser.add_argument("--safety-override-pitch-deg", type=float, default=-5.0)
    parser.add_argument("--safety-override-roll-deg", type=float, default=45.0)
    parser.add_argument("--safety-override-time-horizon-s", type=float, default=15.0)
    parser.add_argument("--safety-override-hard-floor-m", type=float, default=400.0)
    return parser.parse_args()


def load_training_env_config(tag: str) -> tuple[str, dict]:
    """Reconstruct the exact env_config train_rllib.py used for this tag.

    Mirrors train_rllib.py's own precedence: CLI-derived env.* fields first,
    then env_config: section deep-updated on top (matching
    dogfight.ai.training.config_io.deep_update / load_experiment_env_config).
    """
    # Generic prefix/suffix strip so any campaign tag (stage6e/6f/6g/...)
    # resolves to its own generated YAML stem, not just stage6g ones.
    yaml_stem = tag
    if yaml_stem.startswith("altitude_attack_followup_v1_"):
        yaml_stem = yaml_stem[len("altitude_attack_followup_v1_"):]
    if yaml_stem.endswith("_C10"):
        yaml_stem = yaml_stem[: -len("_C10")]
    yaml_path = GENERATED_DIR / f"{yaml_stem}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Generated experiment YAML not found: {yaml_path}")
    experiment = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    env = experiment.get("env", {})
    env_config: dict = {
        "observation_mode": env.get("observation_mode", "tactical16"),
        "target_mode": env.get("target_mode", "loiter"),
        "target_behavior_dll": env.get("target_behavior_dll", "AIP_BASE_target.dll"),
        "ownship_control_mode": "rl",
        "max_engage_time": env.get("max_engage_time", 300.0),
        "episode_step_limit": env.get("episode_step_limit", 18000),
    }
    reward_module = env.get("reward_module", "")

    def deep_update(base: dict, updates: dict) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value

    deep_update(env_config, experiment.get("env_config", {}))
    return reward_module, env_config


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    reward_module, env_config = load_training_env_config(args.tag)
    reward_fn, _default_reward_config = load_reward_hook(reward_module)
    # env_config["reward"] already carries the full resolved reward dict from
    # the generated YAML (base config + variant overrides), so it is used
    # as-is -- do not fall back to _default_reward_config here.

    output_dir_name = args.tag
    if args.max_engage_time is not None:
        env_config["max_engage_time"] = args.max_engage_time
        output_dir_name = f"{args.tag}__engage{int(args.max_engage_time)}"
    if args.target_mode is not None:
        env_config["target_mode"] = args.target_mode
        # The trained scenario pool hard-codes its own target_mode (e.g.
        # "loiter" for close_quarters_750) which silently overrides this
        # top-level setting every reset unless disabled (single_agent_env.py
        # ._apply_scenario_pool_initial_scenario reads
        # sampled.get("target_mode", self._base_target_mode) only when
        # scenario_pool_apply_target_mode is falsy -- found 2026-08-11 when a
        # behavior_tree eval produced bit-identical results to loiter).
        env_config["scenario_pool_apply_target_mode"] = False
        output_dir_name = f"{output_dir_name}__target{args.target_mode}"
    if args.safety_override:
        env_config["safety_override_enabled"] = True
        env_config["safety_override_altitude_m"] = args.safety_override_altitude_m
        env_config["safety_override_pitch_deg"] = args.safety_override_pitch_deg
        env_config["safety_override_roll_deg"] = args.safety_override_roll_deg
        env_config["safety_override_time_horizon_s"] = args.safety_override_time_horizon_s
        env_config["safety_override_hard_floor_m"] = args.safety_override_hard_floor_m
        output_dir_name = f"{output_dir_name}__safety"

    bundle = MODEL_ROOT / args.tag
    if not bundle.exists():
        raise FileNotFoundError(f"Lightweight bundle not found: {bundle}")

    output_dir = FROZEN_EVAL_ROOT / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    env_config["episode_summary_path"] = str(output_dir / "episode_summary.csv")

    provider = RLActionProvider(
        bundle_dir=bundle,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,  # deterministic: action_dist.to_deterministic(), no exploration noise
    )
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=provider,
        reward_fn=reward_fn,
    )
    try:
        for episode_number in range(1, args.episodes + 1):
            env.reset(seed=args.seed + episode_number)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            print(f"[frozen-eval] {args.tag} episode {episode_number}/{args.episodes}")
    finally:
        env.close()

    paths = sorted(output_dir.glob("episode_summary*.csv"))
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
    altitudes = [v for row in rows if (v := num(row, "minimum_altitude_m")) is not None]
    distances = [v for row in rows if (v := num(row, "mean_distance_m")) is not None]
    wez_steps = [v for row in rows if (v := num(row, "wez_steps")) is not None]
    override_steps = [v for row in rows if (v := num(row, "safety_override_steps")) is not None]

    summary = {
        "tag": args.tag,
        "max_engage_time_s": env_config["max_engage_time"],
        "safety_override_enabled": env_config.get("safety_override_enabled", False),
        "episodes": len(rows),
        "crash_rate": crashes / len(rows) if rows else None,
        "minimum_altitude_mean_m": sum(altitudes) / len(altitudes) if altitudes else None,
        "minimum_altitude_worst_m": min(altitudes) if altitudes else None,
        "mean_distance_m": sum(distances) / len(distances) if distances else None,
        "wez_episode_rate": (
            sum(v > 0.0 for v in wez_steps) / len(wez_steps) if wez_steps else None
        ),
        "safety_override_episode_rate": (
            sum(v > 0.0 for v in override_steps) / len(override_steps) if override_steps else None
        ),
        "safety_override_steps_mean": (
            sum(override_steps) / len(override_steps) if override_steps else None
        ),
    }
    summary_path = output_dir / "frozen_eval_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"[done] frozen eval summary: {summary_path}")
    print(
        f"[frozen-eval] {args.tag}: crash_rate={summary['crash_rate']} "
        f"wez_episode_rate={summary['wez_episode_rate']} "
        f"minimum_altitude_mean_m={summary['minimum_altitude_mean_m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
