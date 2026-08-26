from __future__ import annotations

from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dogfight.ai.pure_torch_sac_provider import PureTorchSACActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider

BUNDLE = ROOT / "artifacts/models/highdream/altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10"

algo_provider = RLActionProvider(bundle_dir=str(BUNDLE), algorithm_factory=build_algorithm_from_bundle, explore=False)
torch_provider = PureTorchSACActionProvider(BUNDLE)
rng = np.random.default_rng(260824)
max_delta = 0.0
max_clipped_delta = 0.0
for _ in range(32):
    obs = rng.uniform(-1.0, 1.0, size=19).astype(np.float32)
    expected = algo_provider._compute_module_action(obs)
    actual = torch_provider.compute_raw_action(obs)
    max_delta = max(max_delta, float(np.max(np.abs(expected - actual))))
    max_clipped_delta = max(
        max_clipped_delta,
        float(np.max(np.abs(np.clip(expected, -1.0, 1.0) - actual))),
    )
print(f"max_abs_delta={max_delta:.9g}")
print(f"max_clipped_delta={max_clipped_delta:.9g}")
algo_provider.close()
if max_clipped_delta > 1e-5:
    raise SystemExit("pure Torch actor does not match RLlib inference")
