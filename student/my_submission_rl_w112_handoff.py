"""
[검증용 후보 제출 파일] stage6obs19_v8 (RL) <-> W112 (규칙기반) 핸드오프
=====================================================================
tactical_wrapper의 자체 "neutral" 규칙 로직이 91deg 시작에서 전혀 수렴을
못 하는 게 라이브로 확인됨(ata 87초 내내 60~170deg, 그동안 own_pitch 56deg
까지 치솟으며 4566m->8673m 상승). tactical_wrapper.py 내부는 안 건드리고,
별도 조합으로 새로 만든 후보:

  ATA가 RL 정책이 실제로 학습한 좁은 구간(대략 20~28deg 이내)일 때만 RL을
  쓰고, 그 밖(=지금 라이브에서 tactical_wrapper가 실패하던 바로 그 구간)은
  오늘 하루 라이브로 여러 번 검증된 W112 규칙 컨트롤러가 몰게 함
  (src/dogfight/ai/rl_rule_handoff.py). W109/W110/W111/W112/W113의 오늘자
  라이브 로그를 비교했을 때 W112가 W111의 모든 수정사항 + opening-pull
  보정 + side-shot 게이트 완화까지 포함한 가장 완성된 버전이라 이걸 씀
  (단독으로도 완벽하진 않지만, 지금 비교 가능한 것 중 가장 다듬어진 버전).

재학습 필요 여부: 없음. RL은 기존 stage6obs19_v8 체크포인트 그대로, W112도
기존 코드 그대로 -- 이 파일은 둘을 ATA 기준으로 스위칭만 함.

실행
----
  python student/my_submission_rl_w112_handoff.py
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
from dogfight.ai.rl_rule_handoff import RLRuleHandoffActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.unreal import AIType, MultiprocessUnrealAIPilotUDPClient, ProviderCommandPolicy
from dogfight.unreal.policies import SafetyOverrideCommandPolicy, SafetyOverrideConfig

import run_unreal_inference as rui


# =============================================================================
# 설정 -- 매치마다 SERVER_IP만 바꾸면 됨.
# =============================================================================

TEAM_NAME = "team01"
SERVER_IP = "127.0.0.1"                             # 로컬 검증 서버
SERVER_PORT = 9999

BUNDLE_DIR = (
    "artifacts/models/highdream/"
    "altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10"
)
OBSERVATION_MODE = "tactical19"

# RL <-> W112 전환 히스테리시스. RL은 대략 ata 0~22deg에서만 실제로 학습된
# 상태라(tactical_wrapper.py 자체 docstring 참고), 그보다 넉넉하게 잡음.
ENTER_RL_ATA_DEG = 20.0
EXIT_RL_ATA_DEG = 28.0

AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6
DEBUG_ACTION_REPEAT = False
ENABLE_SAFETY_OVERRIDE = True


def build_rule_provider():
    """W112 that's exactly the same object graph run_unreal_inference.py
    --mode w112 builds -- reused via parse_args()/build_action_provider()
    instead of hand-copying IntegratedBFMConfig fields (that config has
    ~150 fields cascading through w14->...->w109->w111->w112 overrides;
    copying them by hand risks silently missing one)."""
    saved_argv = sys.argv
    try:
        sys.argv = [
            "run_unreal_inference.py",
            "--mode", "w112",
            "--team-name", TEAM_NAME,
            "--server-ip", SERVER_IP,
        ]
        args = rui.parse_args()
    finally:
        sys.argv = saved_argv
    return rui.build_action_provider(args)


def build_action_provider():
    bundle_path = ROOT / BUNDLE_DIR
    if not bundle_path.exists():
        raise FileNotFoundError(f"모델 번들을 찾을 수 없습니다: {bundle_path}")
    print(f"[{TEAM_NAME}] RL 모델 로드: {bundle_path}")
    rl_provider = RLActionProvider(
        bundle_dir=str(bundle_path),
        algorithm_factory=build_algorithm_from_bundle,
    )
    print(f"[{TEAM_NAME}] W112 규칙 컨트롤러 빌드")
    rule_provider = build_rule_provider()
    print(
        f"[{TEAM_NAME}] RL<->W112 핸드오프 구성 "
        f"(enter={ENTER_RL_ATA_DEG}deg, exit={EXIT_RL_ATA_DEG}deg)"
    )
    return RLRuleHandoffActionProvider(
        rl_provider, rule_provider,
        enter_rl_ata_deg=ENTER_RL_ATA_DEG,
        exit_rl_ata_deg=EXIT_RL_ATA_DEG,
    )


def main():
    log_dir = ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = log_dir / (
        f"live_{datetime.now():%Y%m%d_%H%M%S}_rl_w112_handoff.csv"
    )

    print(f"=== {TEAM_NAME} 검증용 클라이언트 시작 (RL v8 <-> W112 핸드오프) ===")
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
