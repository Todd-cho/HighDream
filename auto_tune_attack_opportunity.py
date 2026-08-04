# -*- coding: utf-8 -*-
"""Attack Opportunity(거리 줄이기) 전용 장시간 자동 튜닝 루프 (crash 완전 해결 이후 단계).

로컬(Release 폴더, conda activate aip 상태)에서 실행해야 합니다.
Claude가 대신 실행해줄 수 없습니다.

사용법 (자기 전에 실행해두고 자면 됨):
    python auto_tune_attack_opportunity.py

지금까지 확인된 것:
- altitude_soft_floor_m을 4500->3800으로 낮추고, 파라미터 안 바꾸고 200 iteration씩
  여러 번 이어서 돌리니 crash_rate가 68% -> 25% -> 14% -> **0%**까지 완전히 수렴했다.
- 근데 crash가 0%가 된 마지막 라운드에서 오히려 Mean Range가 다시 커졌다
  (정책이 "이제 안전하니 좀 멀어져도 된다"는 쪽으로 풀어진 것으로 보임).
- crash는 이제 병목이 아니므로, 이번 버전은 attack_range_bonus/far_range_penalty를
  이전 상한(0.40/0.55)보다 훨씬 더 밀어붙일 수 있게 범위를 확장했다.

이전 버전과 달라진 점:
- BOUNDS를 crash가 해결된 걸 전제로 훨씬 공격적인 범위까지 확장
  (attack_range_bonus 최대 0.55, far_range_penalty 최대 0.80,
   far_range_penalty_start_m 최소 1200까지)
- 라운드당 iteration을 200 -> 300으로 늘려서 한 번에 더 오래 학습 (자는 동안 활용)
- 시작값 = 지금까지 나온 가장 진전된 상태(continue3 결과: 전부 상한이었던 값들)
- init_bundle 기본값 = altitude_ease_v1_continue2_200iter (crash 0% 찍은 그 체크포인트)
- crash 안전 기준을 CRASH_SAFE_MAX=0.10으로 더 엄격하게 (0%까지 갔던 걸 알고 있으므로
  조금이라도 다시 늘면 바로 후퇴)

조정 대상 (attack opportunity만, altitude/control은 템플릿 값 그대로 유지):
    ata_scale, aa_scale, wez_bonus,
    attack_range_bonus, far_range_penalty_start_m, far_range_penalty
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Attack opportunity 조정 범위 / 스텝 — crash 해결을 전제로 더 공격적인 범위까지 확장
# ---------------------------------------------------------------------------
BOUNDS: dict[str, tuple[float, float]] = {
    "ata_scale": (0.10, 0.18),
    "aa_scale": (0.03, 0.06),
    "wez_bonus": (0.05, 0.35),
    "attack_range_bonus": (0.40, 0.55),
    "far_range_penalty_start_m": (1200.0, 1800.0),
    "far_range_penalty": (0.55, 0.80),
}

STEP: dict[str, float] = {
    "ata_scale": 0.02,
    "aa_scale": 0.01,
    "wez_bonus": 0.05,
    "attack_range_bonus": 0.03,
    "far_range_penalty_start_m": -150.0,  # 낮추는 방향이 기본
    "far_range_penalty": 0.05,
}

# continue3까지 도달한 최종 상태를 그대로 시작값으로 사용
INITIAL_PARAMS: dict[str, float] = {
    "ata_scale": 0.10,
    "aa_scale": 0.03,
    "wez_bonus": 0.05,
    "attack_range_bonus": 0.40,
    "far_range_penalty_start_m": 1800.0,
    "far_range_penalty": 0.55,
}

MEAN_DIST_TARGET_M = 3000.0  # 1차 수치 목표.txt 기준
CRASH_SAFE_MAX = 0.10  # crash 0%까지 가봤으므로 더 엄격한 안전선
CRASH_REGRESSION_MARGIN = 0.10  # 직전 성공 라운드 대비 이 이상 나빠지면 후퇴


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def clamp(value: float, key: str) -> float:
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, value))


def parse_training_log(csv_path: Path, last_n: int = 20) -> dict[str, float] | None:
    """training_log.csv에서 episode가 실제로 완료된 행만 골라 최근 last_n개 평균."""
    if not csv_path.exists():
        return None

    wanted_cols = [
        "crash_rate",
        "ep_wez_steps",
        "final_ata_deg",
        "final_aa_deg",
        "ep_mean_distance",
        "ep_min_distance",
        "action_sat_rate",
        "reward_mean",
    ]
    valid_rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                crash_val = float(row.get("crash_rate", ""))
            except (TypeError, ValueError):
                continue
            if math.isnan(crash_val):
                continue
            parsed: dict[str, float] = {"crash_rate": crash_val}
            ok = True
            for col in wanted_cols[1:]:
                try:
                    v = float(row.get(col, ""))
                except (TypeError, ValueError):
                    ok = False
                    break
                if math.isnan(v):
                    ok = False
                    break
                parsed[col] = v
            if ok:
                valid_rows.append(parsed)

    if not valid_rows:
        return None

    recent = valid_rows[-last_n:]
    agg: dict[str, float] = {}
    for col in wanted_cols:
        vals = [r[col] for r in recent]
        agg[col] = sum(vals) / len(vals)
    agg["n_episodes_seen"] = float(len(recent))
    return agg


def score_metrics(m: dict[str, float]) -> float:
    s = 0.0
    s -= m["crash_rate"] * 150.0
    s += m["ep_wez_steps"] * 200.0
    s -= max(0.0, m["final_ata_deg"] - 90.0) * 0.3
    s -= max(0.0, m["final_aa_deg"] - 30.0) * 0.3
    s -= max(0.0, m["ep_mean_distance"] - MEAN_DIST_TARGET_M) * 0.02
    s -= max(0.0, 250.0 - m.get("ep_min_distance", 250.0)) * 0.05
    return s


def is_excellent(m: dict[str, float], wez_prev: float) -> bool:
    return (
        m["crash_rate"] <= 0.20
        and m["ep_mean_distance"] <= MEAN_DIST_TARGET_M
        and m["final_ata_deg"] <= 90.0
        and m["final_aa_deg"] <= 40.0
        and (m["ep_wez_steps"] > 0.0 and wez_prev > 0.0)
    )


def decide_next_params(history: list[dict[str, Any]]) -> tuple[dict[str, float], list[str]]:
    cur = history[-1]
    prev = history[-2] if len(history) > 1 else None

    params = dict(cur["params"])
    reasons: list[str] = []

    m = cur["metrics"]
    crash_now = m["crash_rate"]
    crash_prev = prev["metrics"]["crash_rate"] if prev else 0.0
    wez_now = m["ep_wez_steps"]
    wez_prev = prev["metrics"]["ep_wez_steps"] if prev else 0.0
    mean_dist_now = m["ep_mean_distance"]
    ata_now = m["final_ata_deg"]

    # 1) crash가 다시 의미 있게 재발하면: attack 파라미터 후퇴 (안전 우선)
    if crash_now > CRASH_SAFE_MAX and crash_now > crash_prev + CRASH_REGRESSION_MARGIN:
        params["ata_scale"] = clamp(params["ata_scale"] - STEP["ata_scale"], "ata_scale")
        params["aa_scale"] = clamp(params["aa_scale"] - STEP["aa_scale"], "aa_scale")
        params["attack_range_bonus"] = clamp(
            params["attack_range_bonus"] - STEP["attack_range_bonus"], "attack_range_bonus"
        )
        params["far_range_penalty_start_m"] = clamp(
            params["far_range_penalty_start_m"] - STEP["far_range_penalty_start_m"],
            "far_range_penalty_start_m",
        )
        reasons.append(
            f"crash_rate 재발(now={crash_now:.2f}, prev={crash_prev:.2f}, 안전선={CRASH_SAFE_MAX:.2f}) "
            "-> attack 파라미터 후퇴"
        )
        return params, reasons

    # 2) crash는 안전. Mean Range가 아직 너무 크면: far_range 계열 + attack_range_bonus 전진
    if mean_dist_now > MEAN_DIST_TARGET_M:
        params["far_range_penalty_start_m"] = clamp(
            params["far_range_penalty_start_m"] + STEP["far_range_penalty_start_m"],
            "far_range_penalty_start_m",
        )
        params["far_range_penalty"] = clamp(
            params["far_range_penalty"] + STEP["far_range_penalty"], "far_range_penalty"
        )
        params["attack_range_bonus"] = clamp(
            params["attack_range_bonus"] + STEP["attack_range_bonus"], "attack_range_bonus"
        )
        reasons.append(
            f"crash 안전(={crash_now:.2f}), Mean Range 큼({mean_dist_now:.0f}m > {MEAN_DIST_TARGET_M:.0f}m) "
            "-> far_range/attack_range_bonus 전진"
        )
        return params, reasons

    # 3) crash 안전 + Mean Range 양호. WEZ Steps가 아직 0이면: ata/aa_scale 전진
    if wez_now == 0.0:
        params["ata_scale"] = clamp(params["ata_scale"] + STEP["ata_scale"], "ata_scale")
        params["aa_scale"] = clamp(params["aa_scale"] + STEP["aa_scale"], "aa_scale")
        reasons.append(
            f"crash/Mean Range 양호, WEZ Steps=0, final_ata_deg={ata_now:.1f} -> ata/aa_scale 전진"
        )
        return params, reasons

    # 4) 다 좋고 WEZ Steps도 생김: wez_bonus만 소폭 강화하며 값 유지
    if wez_prev > 0.0:
        params["wez_bonus"] = clamp(params["wez_bonus"] + STEP["wez_bonus"], "wez_bonus")
        reasons.append("WEZ Steps 2라운드 연속 >0 -> wez_bonus 소폭 강화, 나머지 유지")
    else:
        reasons.append("WEZ Steps>0 (1라운드째) -> 값 보존")

    return params, reasons


def build_round_yaml(
    template: dict[str, Any],
    round_idx: int,
    reward_params: dict[str, float],
    iterations: int,
    init_bundle: str,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(template))  # deep copy
    tag = f"auto_range_r{round_idx:03d}"
    cfg["output"]["tag"] = tag
    cfg["runtime"]["iterations"] = iterations
    cfg["runtime"]["init_bundle"] = init_bundle
    for key, value in reward_params.items():
        cfg["env_config"]["reward"][key] = value
    cfg["notes"] = f"auto_tune(range-push) round {round_idx}. init_bundle={init_bundle}"
    return cfg


def run_round(python_exe: str, release_dir: Path, yaml_rel_path: Path, attempts: int = 2) -> bool:
    cmd = [python_exe, "scripts/run_experiment.py", str(yaml_rel_path)]
    for attempt in range(1, attempts + 1):
        print(f"\n[auto_tune] 실행 (시도 {attempt}/{attempts}): {' '.join(cmd)} (cwd={release_dir})")
        result = subprocess.run(cmd, cwd=str(release_dir))
        if result.returncode == 0:
            return True
        print(f"[auto_tune] 학습 실패 (returncode={result.returncode}).")
    return False


def write_summary_row(summary_path: Path, row: dict[str, Any], write_header: bool) -> None:
    fieldnames = list(row.keys())
    mode = "w" if write_header else "a"
    with summary_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="crash 해결 이후: Mean Range를 줄이기 위해 far_range 계열을 장시간 자동 튜닝"
    )
    parser.add_argument("--release-dir", default=".", help="Release 폴더 경로 (기본: 현재 폴더)")
    parser.add_argument(
        "--template",
        default="experiments/team_base_R08_original_continue.yaml",
        help="템플릿 yaml 경로 (altitude/control은 이 파일 값 그대로 유지됨)",
    )
    parser.add_argument("--python", default=sys.executable, help="사용할 python 실행파일")
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="라운드당 iteration 수 (200보다 늘려서 한 번에 더 오래 학습)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=15,
        help="최대 라운드 수 (라운드당 300 iteration이면 대략 30분 내외 예상, 15라운드로 여유있게)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=12.0,
        help="최대 실행 시간(시간 단위) 안전장치",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="최고 점수가 이만큼 연속 '성공' 라운드 동안 갱신 안 되면 조기 종료 (0=비활성화)",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="이만큼 연속으로 라운드가 실패하면 구조적 문제로 보고 완전히 중단",
    )
    parser.add_argument(
        "--attempts-per-round",
        type=int,
        default=2,
        help="라운드 하나당 몇 번까지 재시도할지 (크래시 대비)",
    )
    parser.add_argument(
        "--last-n-episodes",
        type=int,
        default=20,
        help="라운드 끝난 뒤 지표 집계에 쓸 최근 episode 개수",
    )
    parser.add_argument(
        "--start-init-bundle",
        default="artifacts/models/highdream/altitude_ease_v1_continue2_200iter",
        help="1라운드에서 이어서 학습할 체크포인트 (crash 0% 찍은 그 지점)",
    )
    args = parser.parse_args()

    release_dir = Path(args.release_dir).resolve()
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = release_dir / template_path
    if not template_path.exists():
        print(f"[auto_tune] 템플릿 yaml을 찾을 수 없습니다: {template_path}")
        sys.exit(1)

    template = load_yaml(template_path)
    output_name = template["output"]["name"]

    current_params = dict(INITIAL_PARAMS)
    summary_path = release_dir / "artifacts" / "auto_tune_range_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    init_bundle = args.start_init_bundle
    start_time = time.time()

    best_score = float("-inf")
    best_round: dict[str, Any] | None = None
    rounds_since_best = 0
    consecutive_failures = 0
    completed_rounds = 0
    round_idx = 0

    for round_idx in range(1, args.max_rounds + 1):
        elapsed_h = (time.time() - start_time) / 3600.0
        if elapsed_h >= args.max_hours:
            print(
                f"\n[auto_tune] 최대 실행 시간({args.max_hours}시간) 도달. 라운드 {round_idx} 시작 전 종료합니다."
            )
            break

        print(f"\n{'=' * 70}")
        print(f"[auto_tune] 라운드 {round_idx}/{args.max_rounds} 시작 (경과 {elapsed_h:.2f}시간)")
        print(f"{'=' * 70}")
        print(f"[auto_tune] 파라미터(attack만): {current_params}")
        print(f"[auto_tune] init_bundle: {init_bundle}")

        round_yaml = build_round_yaml(
            template, round_idx, current_params, args.iterations, init_bundle
        )
        tag = round_yaml["output"]["tag"]
        yaml_rel_path = Path("experiments") / "auto" / f"range_round_{round_idx:03d}.yaml"
        dump_yaml(round_yaml, release_dir / yaml_rel_path)

        ok = run_round(args.python, release_dir, yaml_rel_path, attempts=args.attempts_per_round)

        summary_row: dict[str, Any] = {
            "round": round_idx,
            "tag": tag,
            "run_ok": ok,
            "elapsed_hours": round(elapsed_h, 3),
            **{f"param.{k}": v for k, v in current_params.items()},
        }

        if not ok:
            consecutive_failures += 1
            summary_row["consecutive_failures"] = consecutive_failures
            write_summary_row(summary_path, summary_row, write_header=(round_idx == 1))
            print(
                f"[auto_tune] 라운드 {round_idx} 실패 ({consecutive_failures}/{args.max_consecutive_failures} 연속). "
                f"같은 파라미터/bundle로 다음 라운드를 계속 시도합니다."
            )
            if consecutive_failures >= args.max_consecutive_failures:
                print(
                    f"\n[auto_tune] {args.max_consecutive_failures}회 연속 실패. 구조적 문제로 보고 완전히 중단합니다."
                )
                break
            continue

        consecutive_failures = 0
        completed_rounds += 1

        csv_path = release_dir / "artifacts" / "logs" / output_name / tag / "training_log.csv"
        metrics = parse_training_log(csv_path, last_n=args.last_n_episodes)
        if metrics is None:
            print(f"[auto_tune] {csv_path} 에서 유효한 지표를 못 찾았습니다. 이 라운드는 실패로 취급하고 재시도합니다.")
            summary_row["consecutive_failures"] = 0
            summary_row["run_ok"] = False
            write_summary_row(summary_path, summary_row, write_header=(round_idx == 1))
            continue

        summary_row.update({f"metric.{k}": v for k, v in metrics.items()})
        summary_row["consecutive_failures"] = 0
        print(
            f"[auto_tune] 이번 라운드 지표(최근 {int(metrics['n_episodes_seen'])}개 episode 평균): {metrics}"
        )

        history.append({"round": round_idx, "tag": tag, "params": dict(current_params), "metrics": metrics})

        score = score_metrics(metrics)
        summary_row["score"] = round(score, 3)
        print(f"[auto_tune] 이번 라운드 점수: {score:.2f} (최고 점수: {best_score:.2f})")

        if score > best_score:
            best_score = score
            best_round = history[-1]
            rounds_since_best = 0
            print(f"[auto_tune] *** 새로운 최고 점수 갱신! (라운드 {round_idx}, tag={tag}) ***")
        else:
            rounds_since_best += 1

        wez_prev = history[-2]["metrics"]["ep_wez_steps"] if len(history) > 1 else 0.0
        excellent = is_excellent(metrics, wez_prev)
        summary_row["is_best_so_far"] = history[-1] is best_round
        summary_row["is_excellent"] = excellent

        next_params, reasons = decide_next_params(history)
        summary_row["decision_reasons"] = " | ".join(reasons)
        write_summary_row(summary_path, summary_row, write_header=(round_idx == 1))

        for r in reasons:
            print(f"[auto_tune] 결정: {r}")

        if excellent:
            print(
                f"\n[auto_tune] 우수 목표 수준 도달! 라운드 {round_idx} (tag={tag})에서 종료합니다."
            )
            break

        if args.patience > 0 and rounds_since_best >= args.patience:
            print(
                f"\n[auto_tune] 최고 점수가 {args.patience}라운드 연속 갱신 안 됨 -> 정체로 판단, 종료합니다."
            )
            break

        init_bundle = f"artifacts/models/{output_name}/{tag}"
        current_params = next_params
    else:
        print(f"\n[auto_tune] 최대 라운드({args.max_rounds})까지 도달했습니다.")

    total_h = (time.time() - start_time) / 3600.0
    print(f"\n{'=' * 70}")
    print(
        f"[auto_tune] 전체 종료. 총 소요 시간: {total_h:.2f}시간, "
        f"성공 라운드 {completed_rounds}개 (시도한 라운드 번호 최대 {round_idx})"
    )
    print(f"[auto_tune] summary CSV: {summary_path}")
    if best_round is not None:
        print(f"\n[auto_tune] === 최고 점수 라운드 ===")
        print(f"  round: {best_round['round']}  tag: {best_round['tag']}  score: {best_score:.2f}")
        print(f"  params: {best_round['params']}")
        print(f"  metrics: {best_round['metrics']}")
        print(f"  bundle: artifacts/models/{output_name}/{best_round['tag']}")
        print(
            "  -> 이 bundle을 init_bundle로 쓰거나, 이 파라미터를 최종 yaml로 팀에 공유하시면 됩니다."
        )
    else:
        print("[auto_tune] 완료된(성공한) 라운드가 없습니다. summary CSV에서 실패 원인을 확인해주세요.")


if __name__ == "__main__":
    main()