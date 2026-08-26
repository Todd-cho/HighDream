"""
[검증용 후보 제출 파일] stage6obs19_v8 + tactical_wrapper (재학습 없음)
=====================================================================
`student/my_submission.py`를 건드리지 않고 별도로 만든 라이브 검증용 파일.
기존 my_submission.py는 오늘 작업한 W-시리즈/RL 체크포인트와 전혀 연결되어
있지 않았음(순수 team01 스켈레톤) -- 이 파일은 frozen_eval에서 실제 상대
scripted_pursuit 기준 n=101, 승 90.1% / 패 2.0% / 무 7.9% / 추락 0%를 기록한
조합(stage6obs19_v8_angle090_opp055_50iter_C10 체크포인트 +
TacticalWrapperActionProvider, src/dogfight/ai/tactical_wrapper.py)을
그대로 재현한다.

재학습 필요 여부
----------------
NO. TacticalWrapperActionProvider는 이미 학습된 RL 체크포인트(policy_weights)를
그대로 불러와서, 그 출력 위에 규칙 기반 감독 레이어(ATA가 너무 커서 정책이
경험 못 한 구간이면 lead-pursuit 규칙으로 핸드오프)를 씌우기만 함.
가중치 업데이트/재학습 없음 -- tactical_wrapper.py 상단 docstring
("4.2 재학습 없는 감독 계층") 참고.

주의: 체크포인트 메타데이터(metadata.json)의 observation_mode가
"tactical19"로 학습되어 있어서, 기본값 "tactical16"을 그대로 쓰면 관측
차원이 안 맞음 -- 아래 OBSERVATION_MODE는 반드시 "tactical19"여야 함.

실행
----
  python student/my_submission_stage6obs19v8_wrapper.py

검증 후 이 조합이 실제로 제일 낫다고 확인되면, 이 파일 내용을 그때
`student/my_submission.py`에 반영하면 됨 (지금은 비교/검증 목적으로 분리).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.tactical_wrapper import TacticalWrapperActionProvider, TacticalWrapperConfig
from dogfight.unreal import AIType, MultiprocessUnrealAIPilotUDPClient, ProviderCommandPolicy
from dogfight.unreal.policies import SafetyOverrideCommandPolicy, SafetyOverrideConfig


# =============================================================================
# 설정 -- 매치마다 SERVER_IP만 바꾸면 됨.
# =============================================================================

TEAM_NAME = "team01"
SERVER_IP = "127.0.0.1"                             # 로컬 검증 서버
SERVER_PORT = 9999

# frozen_eval에서 90.1% 승률을 낸 그 체크포인트/관측모드 그대로.
BUNDLE_DIR = (
    "artifacts/models/highdream/"
    "altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10"
)
OBSERVATION_MODE = "tactical19"   # 체크포인트 metadata.json과 반드시 일치해야 함

# 연결 설정 -- my_submission.py와 동일한 관례값.
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6          # metadata.json의 step_ratio=6과 일치
DEBUG_ACTION_REPEAT = False

# 실기(라이브 Unreal)에는 JSBSim 학습 env가 갖고 있던 내부 안전장치가 없으므로
# 대회 규정 추락고도(1000ft=304.8m) 기준 dive-recovery override를 명시적으로 켬.
# run_unreal_inference.py --safety-override와 동일한 기본값.
ENABLE_SAFETY_OVERRIDE = True


def build_action_provider():
    bundle_path = ROOT / BUNDLE_DIR
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"모델 번들을 찾을 수 없습니다: {bundle_path}"
        )
    print(f"[{TEAM_NAME}] RL 모델 로드: {bundle_path}")
    rl_provider = RLActionProvider(
        bundle_dir=str(bundle_path),
        algorithm_factory=build_algorithm_from_bundle,
    )
    print(f"[{TEAM_NAME}] tactical_wrapper 적용 (재학습 없음)")
    # 2026-08-25 live diagnosis: default enter_offensive_ata_deg=5.0 is
    # unreachable from a 91deg start via this wrapper's own neutral-state
    # authority (best achieved live was ~61deg) -- the run stayed stuck in
    # "neutral" the entire flight and its rule-dominant lead-pursuit pitch
    # term drove a sustained climb (4566m->8673m over ~70s). Loosened here
    # only (not touched in tactical_wrapper.py's shared default, which the
    # validated 90.1%-win frozen_eval and other live scripts still use) so
    # "offensive" becomes reachable once real progress is made.
    wrapper_cfg = TacticalWrapperConfig(enter_offensive_ata_deg=30.0)
    return TacticalWrapperActionProvider(rl_provider, wrapper_cfg)


def main():
    log_dir = ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = log_dir / (
        f"live_{datetime.now():%Y%m%d_%H%M%S}_stage6obs19v8_wrapper.csv"
    )

    print(f"=== {TEAM_NAME} 검증용 클라이언트 시작 (stage6obs19_v8 + tactical_wrapper) ===")
    print(f"서버: {SERVER_IP}:{SERVER_PORT}")
    print(f"로그 CSV: {log_csv_path}")

    with activate_rule_xml("Rule_forTraining.xml", ROOT):
        action_provider = build_action_provider()
        command_policy = ProviderCommandPolicy(
            action_provider=action_provider,
            observation_mode=OBSERVATION_MODE,
            ownship_force_side=1,
            target_force_side=2,
            action_repeat=ACTION_REPEAT,
            debug_action_repeat=DEBUG_ACTION_REPEAT,
            log_csv_path=str(log_csv_path),
        )
        if ENABLE_SAFETY_OVERRIDE:
            command_policy = SafetyOverrideCommandPolicy(
                inner=command_policy,
                config=SafetyOverrideConfig(
                    enabled=True,
                    altitude_m=500.0,
                    pitch_deg=-5.0,
                    roll_deg=45.0,
                    time_horizon_s=15.0,
                    hard_floor_m=400.0,
                    min_altitude_m=304.8,
                ),
            )
            print("[safety-override] enabled (1000ft hard floor)")

        # Two processes, not one: the network process owns the UDP socket and
        # answers every 60Hz server frame immediately from shared memory; this
        # (parent) process owns the real RL+wrapper policy and just publishes
        # its newest completed command. A single-process client can't
        # guarantee this split under the GIL when the policy call itself
        # takes real wall-clock time (Ray/RLlib inference) -- exactly the
        # "command delay" this class exists to remove (see
        # MultiprocessUnrealAIPilotUDPClient's docstring, "Recommended for
        # W97 and V1.2 delay-count tests").
        client = MultiprocessUnrealAIPilotUDPClient(
            command_policy=command_policy,
            server_ip=SERVER_IP,
            server_port=SERVER_PORT,
            team_name=TEAM_NAME,
            ai_type=AI_TYPE,
            heartbeat_interval_sec=HEARTBEAT_SEC,
            command_delay_sec=COMMAND_DELAY_SEC,
            recv_timeout_sec=RECV_TIMEOUT_SEC,
            enable_terminal_monitor=True,
        )

        try:
            client.run()
        finally:
            action_provider.close()
            print(f"[{TEAM_NAME}] 클라이언트 종료")


if __name__ == "__main__":
    main()
