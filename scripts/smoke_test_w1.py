"""Quick JSBSim smoke test for W1ControllerActionProvider -- crash/exception
check only, NOT a performance/tuning eval (the whole point of the pivot is
to stop trusting JSBSim for tuning and measure live directly instead)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.scripted_pursuit_provider import ScriptedPursuitActionProvider
from dogfight.ai.student_hooks import load_reward_hook
from dogfight.ai.w1_controller import W1ControllerActionProvider, W1Config
from scripts.run_stage6g_frozen_eval import load_training_env_config

TAG = "altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10"

reward_module, env_config = load_training_env_config(TAG)
reward_fn, _ = load_reward_hook(reward_module)
env_config["max_engage_time"] = 15.0
env_config["safety_override_enabled"] = True
env_config.pop("episode_summary_path", None)
pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}

ownship_provider = W1ControllerActionProvider(W1Config())
target_provider = ScriptedPursuitActionProvider(**pursuit_cfg)
env = DogFightWrapper(
    env_config=env_config,
    ownship_action_provider=ownship_provider,
    target_action_provider=target_provider,
    reward_fn=reward_fn,
)
try:
    env.reset(seed=1)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 300:
        _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        steps += 1
    print(f"[smoke] OK -- {steps} steps, terminated={terminated} truncated={truncated}")
    last = ownship_provider.info_log[-1]
    print(f"[smoke] last info: ata={last['ata']:.1f} dist={last['distance']:.0f} "
          f"bank={last['current_bank']:.1f} pitch={last['own_pitch']:.1f} alt={last['own_alt']:.0f}")
finally:
    env.close()
