"""Probe the RL policy's RAW action outputs (roll/pitch/yaw/throttle cmd, in
[-1,1]) during the merge window of a given episode, to test whether the
episode#18 wez_episode_rate=0.05 / mean_target_damage=0.0001874499999999779
signature -- byte-identical across 6 wildly different reward configs
(rr/ss/tt/uu/vv/ww, 2026-08-19~20) -- is caused by reward-signal dilution
(actions should differ across configs) or by control/physics SATURATION
(actions pin near +-1.0 regardless of reward, because the 8deg merge's
closure geometry demands more roll/pitch authority than the airframe has,
so any policy that's broadly "try to point at the target" converges to the
same saturated commands in that instant).

Usage: python scripts/adhoc_action_saturation_probe.py --tag <tag> --episode-number 18
"""
from __future__ import annotations

import argparse
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
from dogfight.ai.scripted_pursuit_provider import ScriptedPursuitActionProvider
from dogfight.ai.student_hooks import load_reward_hook
from scripts.run_stage6g_frozen_eval import load_training_env_config

MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
BASE_SEED = 269000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--episode-number", type=int, default=18)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--window-s", type=float, default=3.0, help="Print actions for t in [0, window_s].")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag
    reward_module, env_config = load_training_env_config(tag)
    reward_fn, _ = load_reward_hook(reward_module)
    env_config["max_engage_time"] = args.max_engage_time
    env_config["episode_summary_path"] = ""
    env_config["safety_override_enabled"] = False  # raw policy tendency

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}
    seed = BASE_SEED + args.episode_number

    ownship_provider = RLActionProvider(
        bundle_dir=MODEL_ROOT / tag,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )

    actions_log: list[tuple[float, np.ndarray]] = []
    original_compute_action = ownship_provider.compute_action

    def logging_compute_action(context):
        result = original_compute_action(context)
        sim_time = None
        if context.ownship_state is not None:
            try:
                from dogfight.sim.state_schema import StateIndex
                sim_time = float(context.ownship_state[StateIndex.SIM_TIME])
            except Exception:
                sim_time = None
        actions_log.append((sim_time, np.asarray(result.action, dtype=np.float64).copy()))
        return result

    ownship_provider.compute_action = logging_compute_action

    target_provider = ScriptedPursuitActionProvider(**pursuit_cfg)
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=ownship_provider,
        target_action_provider=target_provider,
        reward_fn=reward_fn,
    )
    try:
        env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    finally:
        env.close()

    print(f"[probe] tag={tag} episode={args.episode_number} seed={seed}")
    print(f"[probe] total steps logged: {len(actions_log)}")
    print(f"[probe] roll_cmd/pitch_cmd/yaw_cmd/throttle_cmd for t in [0, {args.window_s}]s:")
    for sim_time, action in actions_log:
        if sim_time is None or sim_time > args.window_s:
            continue
        roll, pitch, yaw, throttle = action[:4]
        sat_roll = "SAT" if abs(roll) > 0.98 else "   "
        sat_pitch = "SAT" if abs(pitch) > 0.98 else "   "
        print(f"    t={sim_time:6.3f}s roll={roll:+.4f}{sat_roll} pitch={pitch:+.4f}{sat_pitch} "
              f"yaw={yaw:+.4f} throttle={throttle:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
