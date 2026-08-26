"""Create a goal27 v8 initializer with exactly preserved tactical19 behavior."""
from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/models/highdream/altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10"
DEST = ROOT / "artifacts/models/highdream/v8_bfm_goal27_zero_init"


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    metadata["metadata"]["obs_mode"] = "bfm_goal27"
    metadata["metadata"]["observation_mode"] = "bfm_goal27"
    metadata["metadata"]["observation_module"] = "student.bfm_goal_observation"
    metadata["metadata"]["observation_size"] = 27
    metadata["metadata"]["parent_bundle"] = str(SOURCE)
    metadata["metadata"]["input_expansion"] = "tactical19 columns preserved; columns 19:27 initialized to zero"
    env_cfg = metadata["algorithm_config"].setdefault("env_config", {})
    env_cfg["observation_mode"] = "bfm_goal27"
    env_cfg["observation_module"] = "student.bfm_goal_observation"
    env_cfg["observation_size"] = 27

    with gzip.open(SOURCE / "policy_weights.pkl.gz", "rb") as handle:
        weights = pickle.load(handle)
    key = "pi_encoder.net.mlp.0.weight"
    old = np.asarray(weights[key], dtype=np.float32)
    if old.shape != (256, 19):
        raise ValueError(f"unexpected v8 input layer: {old.shape}")
    expanded = np.zeros((256, 27), dtype=np.float32)
    expanded[:, :19] = old
    weights[key] = expanded

    (DEST / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(DEST / "policy_weights.pkl.gz", "wb") as handle:
        pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(DEST)


if __name__ == "__main__":
    main()
