# Release_v2 잔차 RL 학습 인수인계서

작성일: 2026-08-22  
대상 프로젝트: `Release_v2_altitude/Release_v2`  
목적: 데스크톱에서는 Unreal 라이브 규칙 제어기를 계속 개선하고, 노트북에서는 검증된 규칙 제어기를 베이스로 **근거리 정밀 조준 전용 잔차 RL(residual RL)** 을 학습한다.

---

## 0. 새 작업자가 가장 먼저 읽을 결론

1. 기존 SAC를 그대로 더 오래 학습하거나 전체 조종을 다시 맡기지 않는다.
2. 규칙 제어기의 기준 베이스는 우선 **W53**으로 고정한다.
   - W54/W55는 라이브에서 사전 잠금(prelock)을 실험 중인 개발 버전이다.
   - 학습 도중 베이스 규칙을 바꾸지 않는다.
3. RL의 임무는 원거리 접근이 아니라, 규칙 명령에 작은 보정을 더해 ATA를 공격 원뿔 안에 오래 유지하는 것이다.
4. 현재 코드에는 추론 시 residual 합성 기능은 있지만, **W53을 포함한 잔차 RL 학습 환경은 아직 구현되어 있지 않다.** 먼저 이를 구현하고 검증해야 한다.
5. 학습 결과는 마지막 iteration이 아니라 고정 평가 세트에서 가장 좋은 체크포인트를 선택한다.
6. JSBSim 성적만으로 최종 성공을 선언하지 않는다. 데스크톱 Unreal 라이브 검증을 반드시 통과해야 한다.

---

## 1. 현재 프로젝트와 역할 분담

### 노트북 트랙

- JSBSim 기반 RL 학습 및 오프라인 평가
- W53 규칙 출력에 대한 작은 roll/pitch 잔차 학습
- 여러 상대와 초기조건을 사용한 일반화 검증
- 체크포인트, 메타데이터, 평가 CSV를 데스크톱으로 전달

### 데스크톱 트랙

- Unreal 주최 서버 라이브 테스트
- W55 이후 규칙 제어기 실험
- 실제 서버 좌표계, 기체 응답, 공격창 및 상대 전술 검증
- 노트북에서 온 잔차 RL을 낮은 비중부터 라이브 A/B

두 트랙은 같은 소스 폴더를 네트워크 공유 상태로 동시에 수정하지 않는다. 노트북은 별도 복사본에서 작업하고, 산출물과 명시적인 코드 패치만 다시 가져온다.

---

## 2. 핵심 파일

### 실행 및 학습

- `train_rllib.py`: RLlib 단일 실험 학습기
- `train_curriculum.py`: 커리큘럼 학습기
- `scripts/run_experiment.py`: YAML 실험 실행기
- `run_unreal_inference.py`: Unreal 라이브 추론 및 W 버전 구성
- `requirements.txt`: Python 패키지 버전

### 환경, 관측, 보상

- `src/dogfight/envs/single_agent_env.py`: JSBSim 단일 에이전트 환경, 행동공간, 데미지
- `src/dogfight/envs/observation.py`: tactical16/tactical19 관측
- `student/my_reward_delta_v1.py`: 기존 공격형 보상 계보의 중심
- `student/my_reward_delta_v1_*`: 기존 보상 실험들
- `student/my_curriculum.py`: 기존 커리큘럼

### 규칙 및 합성

- `src/dogfight/ai/integrated_bfm_controller.py`: W14 이후 통합 규칙 제어기. W53 로직의 본체
- `src/dogfight/ai/hybrid_action_provider.py`: 추론 시 rule/RL residual·blend·switch 합성
- `src/dogfight/ai/tactical_wrapper.py`: 과거 W 계열 래퍼. 현재 W53 본체와 혼동하지 말 것
- `src/dogfight/unreal/policies.py`: Unreal 텔레메트리 변환, CSV 기록
- `src/dogfight/unreal/policies.py`의 `SafetyOverrideCommandPolicy`: 라이브 최종 안전 계층

### 라이브 근거 로그

- `artifacts/logs/run0104_w50_saf1-live_ang091_seed01.csv`: W50 성공, 최소 ATA 0.70°
- `artifacts/logs/run0105_w50_saf1-live_ang091_seed02.csv`: W50 실패, 최소 ATA 9.87°
- `artifacts/logs/run0108_w53_saf1-live_ang091_seed01.csv`: W53 성공, 최소 ATA 0.71°
- `artifacts/logs/run0109_w53_saf1-live_ang091_seed02.csv`: 원거리 최소 ATA 1.42°이나 공격 거리 진입 전 정렬 상실
- `artifacts/logs/run0110_w54_saf1-live_ang091_seed01.csv`: prelock 조건 미발동, 최소 ATA 9.7°

`*_damage_raw.csv`는 현재 모두 헤더뿐이다. 주최 서버가 MT_Damage 패킷을 클라이언트에 보내지 않는 것으로 보이므로, 이 파일이 비었다는 사실만으로 데미지 0이라고 판정하지 않는다.

---

## 3. 현재까지 확인된 라이브 사실

### 기체 응답

- 제어·로그 주기는 약 10 Hz이며 Unreal 텔레메트리는 더 높은 빈도로 들어온다.
- roll은 pitch보다 매우 민감하며 중립에서도 한쪽으로 편향되는 경향이 관측됐다.
- pitch 명령은 대략 `음수 = nose-up`, `양수 = nose-down`으로 라이브에서 확인됐다.
- 고뱅크 상태에서 최대 pitch-up을 줘도 양력이 수평 선회에 사용되어 월드 피치가 늦게 변할 수 있다.
- 스로틀을 0으로 내려도 정면 교차 기하에서는 폐쇄율 350~500 m/s가 발생한다.

### W53이 해결한 것

- 월드 프레임 수평 유도
- 상대 회전율·LOS rate 활용
- 현재 표적 직접 조준을 사용하는 terminal track
- 수직 LOS 변화율 선행항
- 월드 프레임 수직 오차가 클 때만 선택적 bank unload
- 실제 라이브에서 두 차례 ATA 0.7°급 정렬 달성(W50 1회, W53 1회)

### 아직 해결되지 않은 것

- 같은 상대·같은 91° 시작에서도 결과 편차가 큼
- 유리한 정렬을 공격 거리까지 유지하지 못하는 경우가 있음
- 공격창이 대체로 정면 교차이며 폐쇄율이 너무 큼
- 상대 후방/측면에서 안정적으로 WEZ를 유지하는 능력 부족
- 실제 데미지 여부를 CSV만으로 판정하기 어려움

### 최신 라이브 기준 수치

W53 성공 판 `run0108`:

- 최소 ATA 0.71°
- ATA ≤3° 및 유효거리 체류 1.5초
- ATA ≤2° 체류 1.0초
- ATA ≤1° 체류 0.1초
- 공격 구간 평균 폐쇄율 약 504 m/s
- 공격 구간 평균 AA 약 177.5°, 상대 threat ATA 약 2.5°: 거의 상호 정면

W53 두 번째 판 `run0109`:

- 3.43 km에서 최소 ATA 1.42°
- 3 km 이내 4.2°, 2 km 이내 9.7°, 1.219 km 이내 13.7°
- 원거리에서 정렬 후 예측/lag 조준점이 점프하며 정렬 상실

---

## 4. 공격 판정과 각도 정의

`src/dogfight/config.py`의 현재 WEZ 단계는 다음과 같다. `angle_deg`는 전체 원뿔각이고 실제 판정은 코드에서 절반각을 사용한다.

| 단계 | 실제 ATA 조건 | 거리 | 계수 |
|---|---:|---:|---:|
| Phase 1 | ATA ≤1° | 500~3000 ft (152.4~914.4 m) | 1.0 |
| Phase 2 | ATA ≤2° | 500~3500 ft (152.4~1066.8 m) | 0.3 |
| Phase 3 | ATA ≤3° | 500~4000 ft (152.4~1219.2 m) | 0.1 |

학습과 평가에서 단순 `min ATA`만 사용하지 말고 반드시 각 단계의 **연속 체류시간**, 진입 횟수, 진입 직후 이탈률을 기록한다.

중요 용어:

- ATA: 우리 기수와 상대 LOS 사이 각도. 작을수록 우리가 상대를 조준.
- threat ATA: 상대 기수와 우리 LOS 사이 각도. 작을수록 상대가 우리를 조준.
- AA: 교전 자세를 구분하는 aspect angle. 코드의 부호·정의는 `GeoMathUtil.py`와 실제 로그를 기준으로 유지한다.
- LOS az/el 및 LOS rate: 현재 표적 시선과 그 변화율.

---

## 5. 좌표계와 관측에서 절대 다시 만들면 안 되는 버그

1. pitch에 `% 360`을 적용하지 않는다.
   - yaw는 0~360 wrap을 사용할 수 있다.
   - pitch는 -90~90 대칭 범위다.
2. roll/yaw 차분은 ±180 wrap을 고려해야 한다.
3. 상대 속도는 가능하면 실제 상태/텔레메트리를 사용한다.
   - tactical19의 JSBSim 구현은 상대 상대위치를 finite difference하여 16~18번에 상대속도를 넣는다.
   - 모듈 전역 캐시이므로 현재처럼 env runner 1개·env 1개를 전제로 한다. 병렬 env를 늘리기 전에 반드시 인스턴스별 캐시로 바꿔야 한다.
4. State의 D축/고도 부호와 Unreal의 z축을 혼동하지 않는다.
5. 바디 프레임 `aim_el`은 큰 roll에서 수평 오차가 섞일 수 있다. 월드 수직 판단은 `world_elevation - own_pitch`를 사용한다.
6. 미래 표적 위치를 모든 교전 자세에 무조건 적용하지 않는다.
   - 과거 RL9의 lead-pursuit 보상은 AA 게이트 없이 항상 미래 위치를 조준해 라이브에서 취약했다.
   - terminal gun track에서는 현재 표적 LOS 정렬을 우선한다.

---

## 6. 기존 관측공간

### tactical16

`src/dogfight/envs/observation.py` 기준:

| 인덱스 | 값 |
|---:|---|
| 0~5 | own roll, pitch, yaw, KCAS, altitude, health |
| 6~8 | target-own 상대 위치 N/E/D |
| 9~12 | ATA, AA, LOS az, LOS el |
| 13 | target health |
| 14 | WEZ flag |
| 15 | ATA×거리 pursuit score |

### tactical19

- 0~15: tactical16과 동일
- 16~18: 상대 상대속도 N/E/D
- JSBSim에서는 위치 finite difference
- 라이브 Unreal에서는 실제 velocity를 활용하는 편이 우선

### 잔차 RL용 권장 관측

기존 tactical19만으로도 첫 스모크 학습은 가능하지만 최종적으로는 다음 정보를 추가한 전용 관측을 권장한다.

- W53 rule action 4개 또는 최소 roll/pitch rule command
- 현재 roll/pitch/yaw rate
- LOS az/el rate
- closure rate
- own/target speed difference
- target yaw rate 및 world elevation rate
- W53 상태 플래그: terminal track, vertical unload, lag pursuit
- 이전 residual action 2개

목표는 RL이 규칙이 이미 무엇을 하려는지 알고 그 명령을 미세 보정하게 하는 것이다. 관측 크기와 순서는 메타데이터에 고정 기록한다.

---

## 7. 기존 RL 계보에 대한 판단

주요 비교 후보:

- RL8: `altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10`
- RL9: `...stage6obs19_v9_leadpursuit040_50iter_C10`
- RL13/RL16b: 로컬에서 명명했던 후보. 압축본에 해당 산출물이 실제 존재하는지 먼저 검색하고 전체 원본 태그를 기록한다.

주의:

- RL9은 AA 게이트 없는 고정 lead 보상 때문에 기준 후보로 권장하지 않는다.
- 기존 raw RL 라이브 시험들은 모델을 바꿔도 roll/pitch 포화와 조준 실패가 반복됐다.
- iteration을 늘릴수록 성능이 떨어진 사례가 있었다.
- 기존 체크포인트를 초기화에 쓰더라도 새 잔차 action head와 관측 구조가 다르면 그대로 복원할 수 없다.
- 따라서 1차 권장안은 W53 기반 잔차 정책을 새로 학습하는 것이다. 기존 모델은 행동 prior 비교 또는 representation 이식 가능성 조사 대상으로만 둔다.

---

## 8. 구현해야 할 잔차 RL 환경

현재 `HybridActionProvider`는 추론에서 다음과 비슷한 합성이 가능하다.

```text
final_action = rule_action + residual_scale * rl_action
```

그러나 기존 RL은 이 합성 결과로 환경을 경험하며 학습된 것이 아니다. 새 환경 wrapper 또는 action provider를 만들어 학습 step마다 아래 과정을 수행해야 한다.

```text
observation = residual_observation(state, W53 diagnostics)
raw_residual = policy(observation)
scaled_residual = state_dependent_scale * bounded(raw_residual)
rule_action = frozen_W53(state)
combined = clip(rule_action + scaled_residual)
safe_action = training_safety(combined)
sim.step(safe_action)
```

### 행동 권장안

1차 버전에서는 RL 행동을 2차원으로 제한한다.

- residual roll
- residual pitch
- yaw는 0 고정
- throttle은 W53 규칙 사용

권장 물리 범위:

- roll residual 최대 ±0.15
- pitch residual 최대 ±0.15
- 첫 파일럿이 안정적이면 ±0.20, 최대 ±0.25까지만 검토

전체 4축 SAC action을 그대로 더한 뒤 0.35를 곱하는 방식은 권장하지 않는다.

### 상태별 잔차 스케일

초기 권장값:

| 조건 | RL scale |
|---|---:|
| ATA >20° 또는 거리 >3500m | 0.00 |
| ATA 10~20°, 거리 ≤3000m | 0.05~0.10 |
| ATA 3~10°, 거리 ≤1800m | 0.15~0.20 |
| ATA ≤3°, 유효거리 | 0.10~0.15 |
| safety 개입 | RL 무시 |

ATA 3° 안에서는 scale을 다시 낮춰 채터링으로 공격창을 잃지 않게 한다. 경계에는 히스테리시스를 둔다.

### W53 고정 방법

- `IntegratedBFMController`를 학습 환경 내부에 인스턴스화한다.
- `run_unreal_inference.py`의 W53 구성값을 별도 factory/config 함수로 추출하는 것이 좋다.
- 라이브와 학습이 서로 다른 W53 파라미터 복사본을 갖지 않게 단일 factory를 공유한다.
- W54/W55 실험 설정을 W53 학습 베이스에 섞지 않는다.

---

## 9. 보상 설계

보상은 “순간 ATA가 작다”보다 “유효 거리에서 안정적으로 유지한다”를 우선한다.

### 권장 구성

```text
r =
  + survival_small
  + ATA_progress
  + LOS_rate_reduction
  + WEZ_phase_occupancy
  + WEZ_continuity
  + damage_delta
  - WEZ_exit_penalty
  - overshoot_penalty
  - excessive_closure_penalty
  - residual_magnitude_penalty
  - residual_delta/chattering_penalty
  - saturation_penalty
  - low_altitude/stall/safety_penalty
```

### 반드시 지킬 원칙

- Phase 1/2/3를 실제 거리·반각 조건 그대로 계산한다.
- 공격 원뿔에 들어오기 전 dense shaping과 들어온 뒤 체류 보상을 분리한다.
- ATA 절댓값 보상만 크게 주지 않는다. 멀리서 ATA 0°인 상태가 과대평가될 수 있다.
- lead point 보상은 AA·거리·교전 상태 게이트 없이는 사용하지 않는다.
- rule보다 좋아진 정도를 보상하는 residual 개선항을 고려한다.

예시:

```text
delta_ata_improvement = ATA(rule-only counterfactual) - ATA(combined)
```

단, 동일 step의 진짜 counterfactual 시뮬레이션은 비용이 크므로 첫 구현에서는 다음 대체치를 사용한다.

- ATA 감소량
- LOS rate 감소량
- WEZ 연속 체류 증가
- 작은 residual 정규화

### 권장 상대 가중치 시작점

- Phase 3 체류: +1 단위
- Phase 2 체류: Phase 3의 2~3배
- Phase 1 체류: Phase 3의 4~6배
- 공격 원뿔 진입 직후 0.5초 내 이탈: 명확한 패널티
- residual L2: 공격 보상보다 한 단계 작게
- action delta: roll chattering을 억제할 정도로 작게 시작

정확한 수치는 한 번에 확정하지 말고 A/B 2개만 비교한다.

---

## 10. 학습 초기조건과 일반화

주최 시작조건은 주로 91° 또는 0°, 약 10000 ft라고 알려져 있지만 잔차 RL은 근거리 상태 전문가이므로 전체 episode 시작만 복제해서는 데이터 효율이 낮다.

### 잔차 학습용 리셋 분포

- ATA: 0~20° 중심, 일부 20~35° 실패 복구 샘플
- 거리: 500~3500m
- AA: 0~180° 전체
- 고도: 주 경기 고도 중심 ±1500m, 안전 경계 샘플 일부
- own/target bank: -80~80°
- pitch: 현실 범위에서 상승/하강 모두
- closure: -100~500m/s
- 속도차: 느림/동속/빠름 모두
- target vertical rate: 상승·수평·하강
- target yaw rate: 좌/우 지속선회와 반전

### 상대 정책 최소 4종

1. 지속 수평선회: 좌/우, 서로 다른 bank
2. pursuit/intercept 상대
3. 수직 jink: 상승·하강·주기적 변환
4. turn reversal/불규칙 기동 상대

가능하면 다음도 추가한다.

- 이전 RL 정책 스냅샷 pool
- W53 또는 다른 rule 상대
- 상대 파라미터 랜덤화(게인, 반전 주기, 속도)

한 상대 DLL만 계속 사용하면 그 상대 궤적에 맞는 보정값을 외울 수 있다. 학습 episode마다 상대 유형과 파라미터를 샘플링하고 평가 세트에는 학습에서 보지 않은 조합을 남긴다.

---

## 11. JSBSim과 Unreal 차이 및 도메인 랜덤화

JSBSim은 로컬 학습 프록시이며 주최 Unreal 라이브와 동일한 시스템이 아니다.

알려진 차이:

- 상대 정책과 초기 교전 기하
- roll 중립 편향 및 실제 응답속도
- pitch 상하 비대칭
- 텔레메트리/제어 주기와 지연
- 좌표변환·속도 제공 방식
- 충돌/데미지/종료 판정

권장 랜덤화:

- roll/pitch command gain ±15~25%
- roll trim/bias
- pitch 상하 gain 비대칭
- action delay 0~2 control ticks
- observation delay/noise 소량
- 상대 속도/turn rate
- 질량·항력 또는 이에 해당하는 응답 파라미터
- 초기 위치·자세·속도

랜덤화를 처음부터 모두 크게 적용하지 않는다.

1. nominal 환경에서 잔차 학습 가능성 확인
2. 약한 랜덤화
3. 강한 랜덤화 및 미관측 상대 평가

---

## 12. 학습 프로토콜

### 단계 A: 구현 검증

1. W53 rule-only에서 residual=0일 때 기존 W53과 action이 일치하는지 단위 테스트
2. residual roll/pitch ±0.1 주입 시 최종 action 부호와 clip 검증
3. reset 시 W53 내부 rate/latched state가 완전히 초기화되는지 확인
4. tactical19 finite-difference 캐시가 episode 간 오염되지 않는지 확인
5. 1~2 iteration 스모크 학습

### 단계 B: 파일럿 A/B

고정 seed와 동일 평가 매트릭스를 사용한다.

- A: residual max ±0.15, 정밀 보상 보통
- B: residual max ±0.20, 정밀 보상 동일
- 각 10~20 iteration
- 5 iteration마다 lightweight bundle과 native checkpoint 저장

처음부터 100~300 iteration을 돌리지 않는다.

### 단계 C: 승자 연장

- 평가 승자만 20~30 iteration 연장
- 성능이 2회 평가 연속 하락하면 중단
- 마지막 모델이 아니라 최고 평가 모델을 보존

### SAC restore 주의

`train_rllib.py` 주석에 이미 다음 문제가 기록되어 있다.

- RLlib 2.54 SAC restore 시 entropy alpha가 체크포인트에서 정상 복구되지 않을 수 있음
- replay buffer도 복구되지 않아 이어 학습 직후 분포가 깨질 수 있음
- 코드에는 training log에서 alpha 복구 및 replay warmup 기능이 있음

이어 학습 시 반드시 확인:

- `--auto-restore-alpha` 기본 동작 또는 명시적 `--initial-alpha`
- `--replay-warmup-steps`
- 원본 training_log.csv가 체크포인트와 함께 존재하는지

새 잔차 정책은 가능하면 fresh run으로 먼저 성공시킨다.

---

## 13. 평가 매트릭스와 합격 기준

### 고정 평가 세트

최소 20 episode, 권장 40 episode:

- 시작각 0° / 91°
- 수평 좌/우 선회 상대
- 상승/하강 상대
- 반전 상대
- seed 고정 목록

### 반드시 기록할 지표

- Phase 1/2/3 episode 진입률
- 각 phase 총/최장 연속 체류시간
- 첫 ATA ≤20/10/6/3/2/1° 시간
- 유효거리 내 최소 ATA
- WEZ 진입 후 0.5초·1초 유지율
- 평균/최대 closure
- overshoot 횟수
- 우리/상대 damage 또는 HP 차이
- 저고도·실속·충돌·safety 개입
- roll/pitch saturation 비율
- residual RMS, saturation, action delta
- rule-only 대비 개선량

### 1차 승격 기준

잔차 모델은 rule-only W53 평가와 같은 seed에서 비교한다.

- Phase 3 episode rate가 명확히 증가
- Phase 2/1 체류가 감소하지 않거나 새로 발생
- 평균 ATA만 좋아지고 safety/overshoot가 악화된 모델은 탈락
- residual saturation이 지속적으로 높으면 행동 범위나 보상이 잘못된 것
- 최소 2개 미관측 상대에서도 개선

### 라이브 승격 기준

1. 노트북 평가 통과
2. 데스크톱에서 residual scale 0.05~0.10으로 1판 안전 확인
3. 같은 시작각 3판 이상 A/B
4. 문제가 없을 때 0.15, 필요 시 0.20
5. raw RL 100%는 사용하지 않음

---

## 14. 실행 환경과 명령 예시

Windows CMD 기준:

```cmd
cd C:\path\to\Release_v2
conda create -n aip python=3.11 -y
conda activate aip
pip install -r requirements.txt
```

기존 YAML dry-run/인자 확인:

```cmd
python scripts\run_experiment.py experiments\curriculum_altitude_sac_resume.yaml --dry-run
```

실제 실행 형식:

```cmd
python scripts\run_experiment.py experiments\YOUR_RESIDUAL_EXPERIMENT.yaml
```

잔차 환경 구현 후 반드시 새 YAML을 만든다. 예시 이름:

```text
experiments/residual_w53_tactical_v1_a.yaml
experiments/residual_w53_tactical_v1_b.yaml
```

산출물 태그 예시:

```text
resw53_v1_a_20iter_seed01
resw53_v1_b_20iter_seed01
```

실제 CLI 옵션은 `scripts/run_experiment.py --help`, `train_rllib.py --help`로 검증하고 문서의 예시를 맹목적으로 복사하지 않는다.

---

## 15. 노트북 AI가 우선 수행할 작업 순서

1. 이 문서와 아래 파일을 읽는다.
   - `src/dogfight/ai/integrated_bfm_controller.py`
   - `run_unreal_inference.py`의 W53 설정
   - `src/dogfight/ai/hybrid_action_provider.py`
   - `src/dogfight/envs/single_agent_env.py`
   - `src/dogfight/envs/observation.py`
   - `train_rllib.py`
   - `scripts/run_experiment.py`
2. W53 config를 공용 factory로 추출할 설계를 제안한다.
3. residual training wrapper의 관측·행동·reset·합성 흐름을 구현한다.
4. residual=0 동등성 테스트를 작성한다.
5. 전용 reward module과 experiment YAML A/B를 작성한다.
6. dry-run과 1~2 iteration 스모크 테스트를 실행한다.
7. 평가 스크립트를 만든 뒤 rule-only baseline을 먼저 측정한다.
8. A/B 각 10~20 iteration을 실행한다.
9. 최고 체크포인트를 고정 평가 세트로 비교한다.
10. 아래 인계 산출물을 데스크톱으로 전달한다.

AI는 첫 단계부터 장시간 학습을 시작하면 안 된다. 코드 동등성, action 부호, reset, 로그 지표가 먼저다.

---

## 16. 데스크톱으로 돌려줄 산출물

필수:

- 수정된 소스 파일 또는 patch
- 새 experiment YAML
- reward/observation module
- lightweight policy bundle
- native checkpoint
- metadata.json
- training_log.csv 전체
- 고정 평가 episode CSV/요약 JSON
- rule-only와 residual 비교표
- 실행에 사용한 정확한 CMD
- Git commit hash 또는 변경 파일 해시

요약표에는 최소 다음을 포함한다.

```text
model_tag
source/base
seed
iterations
best_iteration
Phase1/2/3 episode rate
Phase1/2/3 dwell time
in-range min ATA
overshoot rate
safety rate
action saturation
residual RMS
opponent matrix result
```

---

## 17. 압축 및 이동 체크리스트

포함:

- `src/`, `student/`, `scripts/`, `experiments/`
- 루트 Python 파일과 DLL/JSBSim 실행 의존성
- `requirements.txt`, README
- 후보 모델 bundle 및 필요한 native checkpoint
- 위 핵심 라이브 로그
- 이 인수인계서

선택 제외:

- `__pycache__/`
- 오래된 중복 로그
- 중복 checkpoint
- TensorBoard 임시 산출물
- 기존 ZIP

주의:

- 원본 폴더를 이동·삭제하지 말고 ZIP 복사본만 만든다.
- 압축 해제 후 DLL 경로와 한글 경로 문제를 확인한다.
- 가능하면 노트북에서는 짧고 영문인 경로를 쓴다. 예: `C:\AIP\Release_v2`.
- ZIP 전달 전 SHA-256을 기록하면 체크포인트 손상 확인에 유리하다.

---

## 18. 금지 사항

- raw SAC에게 전체 비행을 다시 맡기기
- 마지막 iteration을 자동 채택
- 한 상대·한 시작각만으로 모델 선택
- JSBSim 승률만 보고 라이브 성능 확정
- AA 게이트 없는 무조건 lead/lag 보상
- pitch `% 360`
- 큰 residual scale부터 라이브 적용
- 학습 중 W53 규칙 파라미터를 동시에 변경
- replay buffer/alpha 복구 확인 없이 SAC 이어 학습
- Phase 3 진입 한 프레임만으로 성공 판정

---

## 19. 최종 목표 구조

```text
상태/상대 정보
    ↓
Manager
    ├─ 원거리 접근·재획득: 최신 검증 규칙
    ├─ 근거리 기준 명령: 고정 W53 계열
    └─ 정밀 보정: residual RL
              ↓
       상태별 scale/clip
              ↓
        Safety override
              ↓
          Unreal command
```

최종 목표는 RL이 규칙을 대신하는 것이 아니라, 규칙이 반복적으로 만들어내는 3~15°의 남은 오차를 다양한 상대 기동에서 1~3° 공격 원뿔 안으로 안정적으로 밀어 넣고 유지하는 것이다.

---

## 20. 노트북 AI에게 전달할 첫 요청 문구

아래 문구와 함께 이 프로젝트 폴더를 작업공간으로 열어 준다.

> `RL_TRAINING_HANDOFF_KO.md`를 끝까지 읽고, 문서에 지정된 관련 소스도 직접 검증해라. 바로 장시간 학습을 시작하지 말고 먼저 W53 공용 config factory, residual training wrapper, residual=0 동등성 테스트, 전용 관측/보상, A/B YAML, 고정 평가 스크립트의 구현 계획을 제시하라. 계획 검토 후 구현하고 1~2 iteration 스모크 테스트까지만 먼저 실행하라. 기존 raw SAC 전체 조종과 W54/W55 개발 설정을 학습 베이스로 사용하지 마라.
