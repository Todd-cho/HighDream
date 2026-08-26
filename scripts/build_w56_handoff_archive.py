"""Build the final W56 residual RL desktop handoff archive."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "handoff" / "W56_RESIDUAL_RL_HANDOFF_20260823.zip"

EXPLICIT = [
    "run_unreal_inference.py",
    "train_rllib.py",
    "src/dogfight/ai/integrated_bfm_controller.py",
    "src/dogfight/ai/rule_profiles.py",
    "src/dogfight/ai/callbacks.py",
    "src/dogfight/ai/opponent_pool_provider.py",
    "src/dogfight/ai/w56_residual_action_provider.py",
    "src/dogfight/envs/residual_w56_env.py",
    "src/dogfight/envs/single_agent_env.py",
    "src/dogfight/envs/initial_scenario.py",
    "src/dogfight/envs/observation.py",
    "src/dogfight/unreal/policies.py",
    "student/my_reward_residual_w56_v1.py",
    "scripts/run_experiment.py",
    "scripts/verify_w53_factory.py",
    "scripts/verify_w56_factory.py",
    "scripts/verify_residual_w56_env.py",
    "scripts/verify_residual_w56_zero_trajectory.py",
    "scripts/verify_opponent_pool_distribution.py",
    "scripts/sweep_w56_residual_authority.py",
    "scripts/evaluate_w56_final_holdout.py",
    "scripts/diagnose_w56_scripted_pursuit.py",
    "scripts/inspect_w56_action_pipeline.py",
    "scripts/freeze_w56_candidate.py",
    "scripts/build_w56_handoff_archive.py",
    "RL_TRAINING_HANDOFF_KO.md",
    "RL_TRAINING_SESSION_SUMMARY_20260823_KO.txt",
    "RL_W56_EXPERIMENT_REPORT_20260822_NIGHT_TO_20260823_KO.txt",
]

NEW_FILES = {
    "src/dogfight/ai/rule_profiles.py",
    "src/dogfight/ai/opponent_pool_provider.py",
    "src/dogfight/ai/w56_residual_action_provider.py",
    "src/dogfight/envs/residual_w56_env.py",
    "student/my_reward_residual_w56_v1.py",
    "scripts/verify_w56_factory.py",
    "scripts/verify_residual_w56_env.py",
    "scripts/verify_residual_w56_zero_trajectory.py",
    "scripts/verify_opponent_pool_distribution.py",
    "scripts/sweep_w56_residual_authority.py",
    "scripts/evaluate_w56_final_holdout.py",
    "scripts/diagnose_w56_scripted_pursuit.py",
    "scripts/inspect_w56_action_pipeline.py",
    "scripts/freeze_w56_candidate.py",
    "scripts/build_w56_handoff_archive.py",
    "RL_TRAINING_SESSION_SUMMARY_20260823_KO.txt",
    "RL_W56_EXPERIMENT_REPORT_20260822_NIGHT_TO_20260823_KO.txt",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed/unknown"


def role(path: str) -> str:
    if "checkpoint_000050" in path:
        return "Frozen native RLlib checkpoint selected at iteration 50."
    if "bundle_000050" in path:
        return "Lightweight policy bundle for inference/live transfer."
    if path.endswith("best_checkpoint.json"):
        return "Best-checkpoint selection record and deployment overrides."
    if path.startswith("artifacts/logs/"):
        return "W56 training or holdout CSV/JSON evidence."
    if path.startswith("artifacts/evaluations/"):
        return "W56 evaluation, sweep, final-test, or diagnosis result."
    if path.startswith("artifacts/records/"):
        return "W56 experiment record metadata."
    if path.startswith("artifacts/frozen/"):
        return "Frozen candidate integrity manifest."
    if path.startswith("experiments/"):
        return "W56 experiment configuration YAML."
    if "reward" in path:
        return "W56 residual reward definition."
    if "residual_w56_env" in path:
        return "W56 residual training/evaluation environment."
    if "opponent_pool" in path:
        return "Opponent-pool implementation or distribution verifier."
    if "rule_profiles" in path or "integrated_bfm" in path:
        return "Frozen W53/W56 rule profile and integrated BFM controller."
    if "w56_residual_action_provider" in path:
        return "Partially implemented live W56 residual provider (not fully parity-tested)."
    if path.endswith("callbacks.py"):
        return "RLlib callbacks and residual metrics logging."
    if path.endswith("observation.py"):
        return "Observation construction including tactical19 history reset."
    if path.endswith("policies.py"):
        return "Unreal command policy and extended W56RL CSV telemetry."
    if path.startswith("scripts/"):
        return "W56 verification, experiment, evaluation, diagnosis, or packaging utility."
    if path.endswith("run_unreal_inference.py"):
        return "Unreal inference entry point including unfinished --mode w56rl wiring."
    if path.endswith("train_rllib.py"):
        return "RLlib training, restore, holdout, and W56 environment wiring."
    if path.endswith((".md", ".txt")):
        return "Handoff/session/experiment documentation."
    return "Supporting W56 residual RL artifact."


def gather() -> list[Path]:
    selected: set[Path] = set()
    for relative in EXPLICIT:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.add(path)
    for path in (ROOT / "experiments").glob("residual_w56*.yaml"):
        selected.add(path)
    for base in (
        ROOT / "artifacts/logs/highdream",
        ROOT / "artifacts/records/highdream",
    ):
        for directory in base.glob("residual_w56*"):
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".yaml"}:
                    selected.add(path)
    for path in (ROOT / "artifacts/evaluations").glob("w56*.json"):
        selected.add(path)
    frozen = ROOT / "artifacts/frozen/w56_cp50_pitch225_manifest.json"
    if frozen.is_file():
        selected.add(frozen)
    for directory in (
        ROOT / "artifacts/checkpoints/highdream/residual_w56_gain15_wezcurr_200iter_seed01/checkpoint_000050",
        ROOT / "artifacts/models/highdream/residual_w56_gain15_wezcurr_200iter_seed01/bundle_000050",
    ):
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(ROOT).as_posix())


def status(path: str) -> str:
    if path in NEW_FILES or path.startswith(("experiments/residual_w56", "artifacts/")):
        return "NEW"
    return "MODIFIED"


def build_manifest(entries: list[tuple[str, bytes]]) -> str:
    lines = [
        "W56 RESIDUAL RL FINAL HANDOFF MANIFEST",
        f"Created: {datetime.now().astimezone().isoformat()}",
        "Baseline: files created or modified after the W53 ZIP handoff.",
        "",
        "RUNTIME",
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"Executable: {sys.executable}",
        f"Platform: {platform.platform()}",
        f"Ray/RLlib: {package_version('ray')} (RLlib ships with Ray)",
        f"Torch: {package_version('torch')}",
        f"NumPy: {package_version('numpy')}",
        "",
        "LOAD COMMANDS",
        "Native checkpoint evaluation/load:",
        r"  python -c \"from ray.rllib.algorithms.algorithm import Algorithm; a=Algorithm.from_checkpoint(r'artifacts\\checkpoints\\highdream\\residual_w56_gain15_wezcurr_200iter_seed01\\checkpoint_000050'); print(a)\"",
        "Lightweight bundle live command (after finishing parity validation):",
        r"  python run_unreal_inference.py --mode w56rl --bundle-dir artifacts\\models\\highdream\\residual_w56_gain15_wezcurr_200iter_seed01\\bundle_000050 --team-name FDSA --action-repeat 6 --safety-override --log-csv artifacts\\logs\\w56rl_live.csv",
        "Zero-residual smoke command:",
        r"  python run_unreal_inference.py --mode w56rl --bundle-dir artifacts\\models\\highdream\\residual_w56_gain15_wezcurr_200iter_seed01\\bundle_000050 --w56rl-zero-residual --team-name FDSA --action-repeat 6 --safety-override --log-csv artifacts\\logs\\w56rl_zero_live.csv",
        "",
        "REPRODUCTION COMMANDS",
        r"  python scripts\\verify_w56_factory.py",
        r"  python scripts\\verify_residual_w56_env.py",
        r"  python scripts\\verify_residual_w56_zero_trajectory.py",
        r"  python scripts\\verify_opponent_pool_distribution.py",
        r"  python scripts\\sweep_w56_residual_authority.py --checkpoint artifacts\\checkpoints\\highdream\\residual_w56_gain15_wezcurr_200iter_seed01\\checkpoint_000050 --experiment experiments\\residual_w56_gain15_wezcurr_200iter.yaml --episodes 4",
        r"  python scripts\\evaluate_w56_final_holdout.py",
        r"  python scripts\\diagnose_w56_scripted_pursuit.py",
        r"  python scripts\\inspect_w56_action_pipeline.py",
        "WARNING: final-holdout seeds are one-shot final-test data; do not tune against them.",
        "",
        "NOT YET IMPLEMENTED / NOT COMPLETE",
        "- --mode w56rl provider code exists but bundle-load, observation parity, bundle parity, gate-boundary, action-composition, and zero-residual parity tests are not complete.",
        "- Unreal 30-60s zero-residual and active-residual smoke tests are not run.",
        "- Unreal 0deg/91deg rule-only versus residual live matrix is not run.",
        "- 91deg acquisition policy/manager is not implemented; current track residual gate is nearly never reached there.",
        "- 0deg overshoot/min-range/closure manager is not implemented.",
        "- MT_Damage packet decoding/live HP telemetry remains unavailable.",
        "- No final integration with a W59 defense manager is implemented.",
        "",
        "EXCLUSIONS",
        "- __pycache__, .pyc, Ray/session caches, full replay buffers, temporary files, and unrelated historical models are excluded.",
        "- Only checkpoint_000050 and bundle_000050 are included among model binaries.",
        "",
        "FILES",
        "STATUS | SIZE_BYTES | SHA256 | RELATIVE_PATH | ROLE",
    ]
    for relative, data in entries:
        lines.append(f"{status(relative)} | {len(data)} | {sha256_bytes(data)} | {relative} | {role(relative)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    paths = gather()
    entries = [(path.relative_to(ROOT).as_posix(), path.read_bytes()) for path in paths]
    manifest = build_manifest(entries).encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("MANIFEST.txt", manifest)
        for relative, data in entries:
            archive.writestr(relative, data)
    archive_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    with zipfile.ZipFile(OUTPUT, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC verification failed: {bad}")
        names = archive.namelist()
    print(json.dumps({
        "output": str(OUTPUT),
        "size_bytes": OUTPUT.stat().st_size,
        "sha256": archive_hash,
        "archive_entries": len(names),
        "manifested_files": len(entries),
        "crc_test": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
