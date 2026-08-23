"""Create an immutable-content manifest for the selected W56 residual candidate."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "frozen" / "w56_cp50_pitch225_manifest.json"

ARTIFACTS = [
    "artifacts/checkpoints/highdream/residual_w56_gain15_wezcurr_200iter_seed01/checkpoint_000050",
    "artifacts/models/highdream/residual_w56_gain15_wezcurr_200iter_seed01/bundle_000050",
    "artifacts/records/highdream/residual_w56_gain15_wezcurr_200iter_seed01/best_checkpoint.json",
    "artifacts/logs/highdream/residual_w56_gain15_wezcurr_200iter_seed01/training_log.csv",
    "artifacts/logs/highdream/residual_w56_gain15_wezcurr_200iter_seed01/episode_summary_runner_1_env_0.csv",
    "artifacts/logs/highdream/residual_w56_gain15_wezcurr_200iter_seed01/holdout_eval_log.csv",
    "artifacts/evaluations/w56_pitch225_confirm_8ep.json",
    "experiments/residual_w56_gain15_wezcurr_200iter.yaml",
    "student/my_reward_residual_w56_v1.py",
    "src/dogfight/ai/callbacks.py",
    "src/dogfight/ai/rule_profiles.py",
    "src/dogfight/envs/residual_w56_env.py",
    "train_rllib.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_path(relative: str) -> list[dict]:
    target = ROOT / relative
    if not target.exists():
        raise FileNotFoundError(target)
    files = sorted(target.rglob("*")) if target.is_dir() else [target]
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
        if path.is_file()
    ]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_state() -> dict:
    probe = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return {"available": False, "reason": "workspace is not a git repository"}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    return {
        "available": True,
        "commit": commit,
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "working_tree_diff_size_bytes": len(diff),
    }


def main() -> None:
    entries = []
    for artifact in ARTIFACTS:
        entries.extend(describe_path(artifact))
    manifest = {
        "manifest_schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "checkpoint_iteration": 50,
            "observation_dimension": 36,
            "observation_normalization": "ResidualW56Env normalized/clipped feature construction; no external filter",
            "action_dimension": 3,
            "action_scale": {"roll": 0.15, "pitch": 0.225, "throttle": 0.15},
            "terminal_pitch_los_rate_gain": 1.5,
            "gate": {"distance_m_lte": 3000.0, "ata_deg_lte": 20.0, "threat_ata_deg_gte": 40.0},
            "action_order": "W56 rule action -> gated scaled residual -> clip; environment safety override has final authority",
            "action_rate_limit": "not enabled by the selected experiment YAML",
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "ray": package_version("ray"),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "gymnasium": package_version("gymnasium"),
            "jsbsim": package_version("jsbsim"),
        },
        "git": git_state(),
        "files": entries,
        "policy": "Do not overwrite or continue training in this checkpoint directory.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"files={len(entries)}")


if __name__ == "__main__":
    main()
