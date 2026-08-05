# 팀 공유 모델 비교 폴더

여기는 각자 컴퓨터에서 서로의 모델을 바로 받아서 평가/비교하고, 필요하면 이어서 학습까지 할 수 있게 만든
폴더입니다. `artifacts/models_shared/`와 `artifacts/checkpoints_shared/`는 (다른 `artifacts/` 하위 폴더와
달리) git으로 추적됩니다 — 여기 넣은 파일은 `git add` → `git commit` → `git push`만 하면 팀원들이
`git pull`로 바로 받습니다.

## 두 폴더의 역할이 다름

- **`artifacts/models_shared/<이름>/`** — lightweight bundle (`metadata.json` + `policy_weights.pkl.gz`
  딱 2개 파일). actor 가중치만 있고 critic/replay buffer/optimizer 상태는 없음. **평가/추론 전용**. 항상 여기엔
  올릴 것.
- **`artifacts/checkpoints_shared/<이름>/checkpoint_final/`** — RLlib native checkpoint. **이어서 학습할
  때 필요**. 용량이 커서(보통 수 MB) 선택사항 — 다른 사람이 내 모델에서 이어서 학습해줬으면 할 때만 추가로
  올린다. 중간 체크포인트(`checkpoint_000030` 등)는 올리지 말고 `checkpoint_final`만.
- 두 폴더 다 git에 남기 때문에, 여기 넣는 건 실제로 남길 가치가 있는 결과물로만 제한할 것 — 실험 중 나온
  것 전부 올리면 저장소가 계속 불어남.

## 내 최고 성능 모델을 여기에 올리는 방법

1. lightweight bundle 위치를 찾는다 (보통 `artifacts/models/highdream/<내_실험_tag>/`).
2. `models_shared` 밑에 `<내이름>_<실험tag>` 형식으로 새 폴더를 만들고 그 두 파일만 복사한다:
   ```
   mkdir -p artifacts/models_shared/영인_stage5_ata015_150iter
   cp artifacts/models/highdream/stage5_ata015_150iter/metadata.json \
      artifacts/models/highdream/stage5_ata015_150iter/policy_weights.pkl.gz \
      artifacts/models_shared/영인_stage5_ata015_150iter/
   ```
3. (선택) 남이 이어서 학습할 수 있게 하려면, native checkpoint의 `checkpoint_final`도 **같은 이름**으로
   복사한다:
   ```
   mkdir -p artifacts/checkpoints_shared/영인_stage5_ata015_150iter
   cp -r artifacts/checkpoints/highdream/stage5_ata015_150iter/checkpoint_final \
         artifacts/checkpoints_shared/영인_stage5_ata015_150iter/checkpoint_final
   ```
4. 아래 "등록된 모델" 표에 한 줄 추가한다 (직접 학습 로그에서 확인한 실제 수치로 — 마지막 1개 episode가
   아니라 **마지막 10~20개 episode 평균**을 쓸 것, 초반 random exploration 구간이나 단일 episode는 착시를
   만듦).
5. `git add artifacts/models_shared/ artifacts/checkpoints_shared/ .gitignore` (수정했다면) →
   `git commit` → `git push`.

## 남이 올린 모델을 내 컴퓨터에서 평가하는 방법

```
python scripts/evaluate_altitude_bundle.py \
  --bundle artifacts/models_shared/<폴더이름> \
  --output-dir artifacts/eval_compare/<폴더이름> \
  --episodes-per-scenario 20 \
  --seed 260900 \
  --jitter
```

`artifacts/eval_compare/<폴더이름>/frozen_evaluation_summary.csv`에 시나리오별
`crash_rate`, `minimum_altitude_mean_m`, `minimum_altitude_worst_m` 등이 나옵니다. 5개 시나리오
(level_5000 / light_disturbance_4500 / mild_dive_4000 / low_dive_3500 / banked_descent_3800)에 대해
각각 나오니, 전체 평균만 보지 말고 시나리오별로도 비교할 것.

## 남이 올린 모델에서 이어서 학습하는 방법

`checkpoints_shared/<폴더이름>/checkpoint_final`이 있는 모델만 가능합니다 (lightweight bundle만 있는
모델은 critic/replay buffer가 없어서 이어서 학습하면 정책이 붕괴할 수 있음 — 반드시 native checkpoint로).

`scripts/run_experiment.py`로 실행할 실험 yaml의 `runtime.restore_checkpoint`에 절대경로를 넣는다:
```yaml
runtime:
  restore_checkpoint: <repo 절대경로>/artifacts/checkpoints_shared/<폴더이름>/checkpoint_final
```
(참고: [[experiment 규칙]] — 하나의 학습 실행 도중 reward를 바꾸지 않기, 후보당 최소 200~300iteration
권장, 다른 사람 결과와 공정 비교하려면 같은 시작 checkpoint/seed/iteration 수로 맞출 것.)

## 등록된 모델

| 폴더 | 올린 사람 | 베이스/커리큘럼 | 핵심 reward 변경점 | crash율(마지막20 평균) | mean_distance | 최저고도 평균 | checkpoint 있음 | 비고 | 날짜 |
|---|---|---|---|---|---|---|---|---|---|
| `영인_stage5_ata015_150iter` | 영인 | C10 Stage4 native checkpoint에서 이어서 학습 | `ata_scale=0.15`, `attack_range_bonus=0.9`, `far_range_penalty=1.05`, `far_range_penalty_start_m=4000` (승현 far_range span*6 fix 포함) | 5% | 8913m | 3300m | O | Stage5 안전/접근 이분탐색 최종 후보, 5시나리오 mixed pool 150iter | 2026-08-04 |
| `영인_stage5_safe_approach_C10_attack_followup` | 영인 | C10 Stage4 native checkpoint에서 이어서 학습 (`run_altitude_attack_followup.py` Stage5, 180iter, mixed 5시나리오 pool) | 위 150iter 후보와 동일 파라미터 + `range_scale=2.4` (버그로 누락됐다가 2026-08-05 수정 — C10 base엔 range_scale이 없어서 my_reward.py 기본값 0.8로 조용히 폴백되던 문제, 수정 전엔 mean_distance 10860m로 게이트 실패) | 10% | 8384m | 3315m | O | `run_altitude_attack_followup.py` 파이프라인의 frozen 안전검증 통과 후 정식 180iter 실행, 안전/접근 게이트(crash≤20%, 최저고도≥1800m, distance≤9000m) 전부 통과. 정식 180iter라 150iter 탐색판보다 iteration 수는 늘었지만 crash율은 소폭 상승(5%→10%, n=10 오차범위 가능), distance/고도는 비슷~약간 개선. episode 243개 중 crash 원인은 nose_down 1건, roll_instability 0건 (전체 crash율 6.6%). 다음 단계 Stage6(safe_wez_geometry)의 시작 checkpoint. | 2026-08-05 |

(새 모델 올릴 때 이 표에 이어서 한 줄씩 추가해주세요.)
