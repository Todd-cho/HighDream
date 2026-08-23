"""Screen residual authority scales on a frozen W56 SAC checkpoint."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

import train_rllib


CANDIDATES = {
    # roll, pitch, throttle, terminal gain, gate ATA, gate range, min threat ATA
    "baseline_15_20_15": (0.15, 0.20, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "pitch25": (0.15, 0.25, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "pitch225": (0.15, 0.225, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "pitch21": (0.15, 0.21, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "pitch22": (0.15, 0.22, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "pitch23": (0.15, 0.23, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "pitch24": (0.15, 0.24, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "p225_roll125": (0.125, 0.225, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "p225_roll140": (0.14, 0.225, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "p225_roll160": (0.16, 0.225, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "p225_roll175": (0.175, 0.225, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "p225_thr100": (0.15, 0.225, 0.10, 1.5, 20.0, 3000.0, 40.0),
    "p225_thr125": (0.15, 0.225, 0.125, 1.5, 20.0, 3000.0, 40.0),
    "p225_thr175": (0.15, 0.225, 0.175, 1.5, 20.0, 3000.0, 40.0),
    "p225_thr200": (0.15, 0.225, 0.20, 1.5, 20.0, 3000.0, 40.0),
    "p225_gain100": (0.15, 0.225, 0.15, 1.0, 20.0, 3000.0, 40.0),
    "p225_gain125": (0.15, 0.225, 0.15, 1.25, 20.0, 3000.0, 40.0),
    "p225_gain175": (0.15, 0.225, 0.15, 1.75, 20.0, 3000.0, 40.0),
    "p225_gain200": (0.15, 0.225, 0.15, 2.0, 20.0, 3000.0, 40.0),
    "pitch275": (0.15, 0.275, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "pitch30": (0.15, 0.30, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "roll20": (0.20, 0.20, 0.15, 1.5, 20.0, 3000.0, 40.0),
    "throttle10": (0.15, 0.20, 0.10, 1.5, 20.0, 3000.0, 40.0),
    "low_10_15_10": (0.10, 0.15, 0.10, 1.5, 20.0, 3000.0, 40.0),
    "pitch25_gain125": (0.15, 0.25, 0.15, 1.25, 20.0, 3000.0, 40.0),
    "pitch25_gain175": (0.15, 0.25, 0.15, 1.75, 20.0, 3000.0, 40.0),
    "pitch25_gain200": (0.15, 0.25, 0.15, 2.0, 20.0, 3000.0, 40.0),
    "pitch25_gate_ata15": (0.15, 0.25, 0.15, 1.5, 15.0, 3000.0, 40.0),
    "pitch25_gate_ata25": (0.15, 0.25, 0.15, 1.5, 25.0, 3000.0, 40.0),
    "pitch25_gate_range2000": (0.15, 0.25, 0.15, 1.5, 20.0, 2000.0, 40.0),
    "pitch25_gate_threat30": (0.15, 0.25, 0.15, 1.5, 20.0, 3000.0, 30.0),
    "pitch25_gate_threat50": (0.15, 0.25, 0.15, 1.5, 20.0, 3000.0, 50.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=990000001)
    parser.add_argument("--output")
    parser.add_argument(
        "--candidates",
        help="Comma-separated candidate names; default evaluates all.",
    )
    args = parser.parse_args()

    selected = CANDIDATES
    if args.candidates:
        names = [item.strip() for item in args.candidates.split(",") if item.strip()]
        unknown = [name for name in names if name not in CANDIDATES]
        if unknown:
            raise ValueError(f"unknown candidates: {unknown}")
        selected = {name: CANDIDATES[name] for name in names}

    experiment = yaml.safe_load(Path(args.experiment).read_text(encoding="utf-8"))
    base_config = copy.deepcopy(experiment["env_config"])
    base_config.update(experiment["env"])
    checkpoint = Path(args.checkpoint).resolve()
    register_env("dogfight-single-agent-v0", train_rllib.env_creator)
    train_rllib._ensure_ray_runtime_env()
    algorithm = Algorithm.from_checkpoint(str(checkpoint))
    rows = []
    try:
        for name, values in selected.items():
            roll, pitch, throttle, terminal_gain, gate_ata, gate_range, gate_threat = values
            env_config = copy.deepcopy(base_config)
            residual = env_config.setdefault("residual_w56", {})
            residual["max_roll_scale"] = roll
            residual["max_pitch_scale"] = pitch
            residual["max_throttle_scale"] = throttle
            residual["gate_ata_deg"] = gate_ata
            residual["gate_range_m"] = gate_range
            residual["gate_min_threat_ata_deg"] = gate_threat
            residual.setdefault("w56_overrides", {})[
                "terminal_pitch_los_rate_gain"
            ] = terminal_gain
            result = train_rllib._run_holdout_evaluation(
                algorithm,
                env_config,
                max_engage_time=180.0,
                base_seed=args.seed,
                num_episodes=args.episodes,
            )
            row = {
                "name": name,
                "roll": roll,
                "pitch": pitch,
                "throttle": throttle,
                "terminal_pitch_los_rate_gain": terminal_gain,
                "gate_ata_deg": gate_ata,
                "gate_range_m": gate_range,
                "gate_min_threat_ata_deg": gate_threat,
                **result,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    finally:
        algorithm.stop()

    rows.sort(key=lambda row: row["mean_damage_exchange"], reverse=True)
    payload = {"checkpoint": str(checkpoint), "episodes": args.episodes, "results": rows}
    print("RANKING")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
