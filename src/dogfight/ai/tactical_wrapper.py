# -*- coding: utf-8 -*-
"""Non-learned tactical supervisory layer wrapping an inner RL ActionProvider,
per the design in C:\\Users\\2bpro\\Documents\\먀ㅔ\\주최조건_1대1_계층형전술_v3_v4_설계전략.txt
(2026-08-20, "4.2 재학습 없는 감독 계층" -- a supervisory layer added on top of
an existing checkpoint with NO retraining).

Motivation: every v5/v6b/v7/v8/v9 live test this session showed the SAC
policy commanding full +-1.0 roll/pitch from frame 1 at a ~91deg ATA start
and staying saturated 75-84% of the flight -- the policy was trained almost
entirely on ATA 0-22deg (Offensive-regime) starts and has essentially no
experience at high ATA. Rather than trying to train that experience in
(expensive, and lead_pursuit reward v9 showed a naive attempt can overfit to
the specific training opponent instead), this wrapper keeps the existing
policy as the "Offensive specialist" it already is, and hands off to a
simple rule-based lead-pursuit controller whenever ATA is too large for the
policy to have any real experience with, blending smoothly between the two
based on a hysteresis state machine (Neutral/Approach, Offensive, Overshoot
Prevention -- Recovery/Safety is deliberately NOT reimplemented here since
single_agent_env.py's _apply_safety_override / policies.py's
SafetyOverrideCommandPolicy already provide it as a final, higher-priority
layer applied AFTER this wrapper's output, matching the design doc's own
layering: tactical blend -> action rate limiter -> safety override).

Implements the shared ActionProvider interface (action_provider.py), so the
exact same class works for both JSBSim training/eval (DogFightWrapper's
ownship_action_provider=) and live Unreal inference (policies.py's
ProviderCommandPolicy(action_provider=...)) without any duplication -- both
paths already construct the same ActionContext/ActionResult shape.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex, position_ned


@dataclass
class TacticalWrapperConfig:
    # Hysteresis thresholds (deg) -- REVISED 2026-08-21 per a design-doc
    # review of our own run8/9/10 logs (주최조건_1대1_계층형전술_v3_v4_설계전략.txt):
    # the original 25/40 handed 80% control to SAC as soon as ATA reached
    # 25deg, but SAC has no training experience converging past there --
    # real damage needs ATA<=1-3deg (WEZ), so the old thresholds handed off
    # ~20+ degrees before the policy that received control could actually
    # finish the job. Matches the exact observed failure: logs repeatedly
    # approach ATA~20-25deg then stop converging.
    enter_offensive_ata_deg: float = 5.0
    exit_offensive_ata_deg: float = 10.0
    # New "Weapons Track" state (2026-08-21, same doc review): ATA<=3deg AND
    # distance<=weapons_track_distance_m (WEZ Phase3's own outer range
    # bound, config.py's wez.phases[-1].max_range_m=1219.2m) -- Neutral and
    # Offensive both optimize for closing/reducing ATA, but once actually in
    # a firing solution the goal flips to HOLDING it (small smooth
    # corrections, controlled closure, LOS rate near zero) rather than
    # continuing to aggressively close, which is what was overshooting the
    # solution and diving straight back out to Overshoot/Neutral.
    # Hysteresis (2026-08-21, design-doc fix for run12: the single 3.0deg
    # threshold made weapons_track drop out immediately the instant ATA
    # ticked up past it -- observed 2.12->1.98->3.68deg, gone after one
    # tick). enter_ata is tight (must actually acquire a real lock first);
    # exit_ata is looser so a brief wobble doesn't immediately kick it back
    # out, giving throttle/precision control a bit longer to hold the pass.
    weapons_track_enter_ata_deg: float = 3.0
    weapons_track_exit_ata_deg: float = 5.0
    weapons_track_distance_m: float = 1219.2
    # Overshoot Prevention trigger -- REVISED 2026-08-21: the fixed-distance
    # trigger (600m) was measured too late against the real live closure
    # rate (run8: closure=443.3m/s at distance=406.8m -- only
    # (600-152)/443=1.0s of warning from a 600m trigger, not enough time to
    # decelerate/switch to lag pursuit). Replaced with a time-to-min-range
    # estimate so the warning scales with how fast the merge is actually
    # happening instead of a single fixed distance.
    overshoot_closure_mps: float = 100.0
    overshoot_time_to_min_range_s: float = 3.0
    overshoot_min_range_m: float = 152.4
    # Blend weights (rl_weight, rule_weight) per state -- REVISED 2026-08-21:
    # offensive dropped 0.8->0.3 (SAC's own precision past ATA~10deg is
    # unproven, so keep the rule dominant even once "offensive"); overshoot
    # dropped 0.3->0.1 (same reasoning, this is also a precision-sensitive
    # state). weapons_track is new and kept rule-dominant for the same
    # reason, doc: "규칙 기반 정밀 추적 비중: 높게".
    neutral_rl_weight: float = 0.2
    offensive_rl_weight: float = 0.3
    weapons_track_rl_weight: float = 0.15
    overshoot_rl_weight: float = 0.1
    # Head-On Attack (2026-08-21, design-doc fix for run13/14): a fast
    # merge with aa>=head_on_attack_aa_deg is a head-on/high-aspect
    # crossing, not a tail-chase overtake -- geometrically you cannot
    # "overshoot" someone flying at you, so this is checked BEFORE Overshoot
    # and never falls into it. No lag pursuit (aims near the target's
    # CURRENT position via head_on_attack_lead_horizon_s, not a predicted
    # lead point), and throttle is NOT reduced (slowing down doesn't prevent
    # a head-on pass the way it prevents an overtake, so there's no benefit,
    # only lost energy).
    head_on_attack_aa_deg: float = 120.0
    head_on_attack_rl_weight: float = 0.15
    head_on_attack_lead_horizon_s: float = 0.15
    # Rule-based lead-pursuit controller gains: roll_cmd = clip(k_roll *
    # az_deg / norm_deg, -1, 1), pitch_cmd = clip(-k_pitch * el_deg / norm_deg, -1, 1).
    roll_gain: float = 1.0
    # Tried 30.0 (2026-08-21, user observation "선회가 좀 크게 도는 감이
    #있어") but the next live test got WORSE on every precision metric
    # (roll saturation 37.8%->46.8%, ata<=30deg 15.2%->7.7%, ata<=10deg
    # 2.8%->0%) -- reverted to the confirmed-better 45.0. The turning-radius
    # feel may need a different fix (e.g. the LOS-rate/throttle changes
    # below) rather than a blunter position gain.
    roll_norm_deg: float = 45.0
    pitch_gain: float = 1.0
    pitch_norm_deg: float = 60.0
    yaw_gain: float = 0.0
    # Lead-pursuit prediction horizon (seconds of extrapolation from the
    # target's measured per-second velocity, finite-differenced).
    #
    # REVISED 2026-08-21 (design-doc fix for a circular deadlock found in
    # run11): this used to be a FIXED 1.5s everywhere except inside
    # weapons_track (0.3s) -- but weapons_track can only be entered once
    # ata<=3deg, and the long 1.5s horizon (extrapolating up to
    # lead_velocity_clip_mps*1.5=~600m ahead) was itself preventing ata from
    # ever getting that low to begin with (run11's best in-range ATA was
    # 26.65deg vs run8/v3's 6.28deg) -- the entry condition and the control
    # mode were blocking each other. Now scheduled by DISTANCE instead of by
    # state, so it shortens automatically on approach regardless of which
    # state is currently active:
    # Four distance bands, tightest-first (doc's recommended values):
    #   <=1219.2m (WEZ outer bound): lead_horizon_wez_s (damage is judged
    #     against the target's CURRENT position, not a predicted one)
    #   <=2000m: lead_horizon_close_s
    #   <=4000m: lead_horizon_mid_s
    #   else: lead_horizon_far_s
    lead_horizon_wez_s: float = 0.1
    lead_horizon_close_s: float = 0.2
    lead_horizon_close_distance_m: float = 2000.0
    lead_horizon_mid_s: float = 0.5
    lead_horizon_mid_distance_m: float = 4000.0
    lead_horizon_far_s: float = 1.0
    lead_velocity_clip_mps: float = 400.0
    # Overshoot Prevention: reduced throttle (doc 3.3 "throttle 감소").
    overshoot_throttle: float = 0.3
    # Lag-pursuit aim point for Overshoot (2026-08-21, design-doc fix for
    # run12): previously Overshoot only cut throttle/RL weight but kept
    # aiming at the SAME lead point as every other state -- with a merge
    # closure of ~534m/s that's nowhere near enough to avoid blowing
    # through, since the nose is still being driven to point ahead of the
    # target's current position. Aims behind the target instead
    # (target_position - unit(target_velocity) * overshoot_lag_distance_m),
    # so the nose falls toward the target's SIX o'clock instead of its
    # twelve, converting a head-on-ish blow-through into a trailing
    # position -- textbook lag pursuit.
    overshoot_lag_distance_m: float = 500.0
    # Gate for the lag-pursuit aim point above (2026-08-21): only apply it
    # when aspect angle (target's orientation relative to the LOS) shows
    # we're already roughly behind the target, not in a head-on merge.
    # Starting loose per the doc's suggestion (90deg); can tighten toward
    # 60deg once there's data on how it behaves.
    tail_chase_aa_deg: float = 90.0
    # Weapons Track throttle (2026-08-21, doc: "throttle: closure를 줄이도록
    # 조절") -- proportional control around a target distance
    # (weapons_track_target_distance_m, doc's 300-900m band midpoint),
    # backing off throttle when closing in too fast and adding it back when
    # drifting out, instead of just flooring/holding a fixed value.
    weapons_track_target_distance_m: float = 600.0
    weapons_track_throttle_base: float = 0.6
    weapons_track_throttle_closure_gain: float = 0.4
    weapons_track_throttle_min: float = 0.2
    weapons_track_throttle_max: float = 0.9
    # doc: "roll/pitch 명령: 작고 부드럽게" -- scale down the rule
    # controller's roll/pitch magnitude specifically while holding a track,
    # since the point here is small corrections to maintain lock, not
    # aggressive pursuit.
    weapons_track_command_scale: float = 0.5

    # Altitude/energy-aware pitch ceiling (added 2026-08-21 after a live test
    # showed pitch_cmd pinned near -1.0 (nose up, per the sign convention
    # below) for ~116s while stuck at 600-700m -- the pure-geometry rule had
    # no altitude awareness at all, exactly the flaw 전술.txt critiques in
    # the base SAC policy. Caps how far NOSE-DOWN (positive pitch_cmd) the
    # FINAL blended command may go as altitude drops, linearly interpolating
    # from no restriction at altitude_safe_floor_m down to a forced climb
    # ceiling at altitude_critical_floor_m. Only clamps the upper (nose-down)
    # side -- nose-up/climb commands are never restricted.
    altitude_safe_floor_m: float = 1800.0
    altitude_critical_floor_m: float = 1000.0
    min_climb_pitch_ceiling: float = -0.5
    # Head-On Attack gets its OWN, much narrower band (2026-08-21, fix for a
    # live test showing pitch_cmd pinned near -0.5 for many consecutive
    # seconds during head_on_attack): the general 1000-1800m ramp above
    # covers a very common merge altitude, so it was constantly overriding
    # the aim solution's actual pitch computation, not just catching genuine
    # emergencies -- the aircraft couldn't track the target's elevation
    # properly while fighting this ceiling. The external --safety-override
    # layer (SafetyOverrideCommandPolicy, applied AFTER this wrapper, hard
    # trigger ~500m/400m) is a separate, already-proven safety net, so it's
    # safe to let Head-On Attack's own ceiling sit much closer to real
    # danger and otherwise leave pitch control to the aim solution.
    head_on_attack_altitude_safe_floor_m: float = 900.0
    head_on_attack_altitude_critical_floor_m: float = 550.0
    # Roll is also capped below this altitude (2026-08-21, same request) --
    # wings-level recovery takes priority over continuing to bank hard for
    # an attack while critically low.
    low_altitude_roll_limit: float = 0.4

    # "Stuck" detector + disengage fallback (added 2026-08-21, same live
    # test): if the FINAL blended roll or pitch command stays at/above
    # stuck_saturation_threshold for stuck_duration_s of sim time without
    # ATA improving by at least stuck_ata_improvement_deg, force a brief
    # wings-level/moderate-climb/low-aggression "disengage" for
    # disengage_duration_s before resuming normal logic -- mirrors real BFM
    # advice to separate and re-set up rather than keep grinding a geometry
    # that isn't resolving.
    stuck_saturation_threshold: float = 0.9
    stuck_duration_s: float = 5.0
    stuck_ata_improvement_deg: float = 5.0
    disengage_duration_s: float = 3.0
    disengage_pitch_cmd: float = -0.3
    # Raised 0.3->0.8 (2026-08-21, user observation "머지 후 지나갔을때
    # 상대쪽으로 가려는 경향이 너무 약함"): a live log showed disengage
    # firing right after several merges (ATA still climbing toward
    # 160deg+, target working into our six) and cutting roll authority to
    # 0.3 right when the hardest possible turn-back is needed -- this was
    # designed for a different problem (grinding at low altitude with no
    # progress) and was actively suppressing post-merge reacquisition here.
    disengage_roll_scale: float = 0.8

    # Fine-tracking precision refinement (added 2026-08-21 after a live test
    # showed Neutral state active 76.5% of live flight time (vs only 18% in
    # JSBSim against the training opponent) -- the real live opponent holds
    # ATA high far more than our training proxy, so this rule controller (not
    # SAC) does most of the actual flying live, and its plain fixed-gain P
    # law never precisely converged (WEZ needs ATA<=1-3deg; live only hit
    # <=1deg on 1/2000 frames). Two additions, both straight from
    # 전술.txt/design-doc's own recommendations:
    #  1. Gain scheduling: once |az| or |el| is already inside
    #     fine_track_threshold_deg, switch to a TIGHTER norm (more command
    #     per degree of remaining error) instead of the same coarse-tracking
    #     norm used for a 90deg initial offset -- a fixed-linear law under
    #     one norm is necessarily very gentle on the last few degrees.
    #  2. LOS-rate (PN-style) term: az_rate/el_rate (deg/s, finite-differenced)
    #     added on top of the position term -- pushes harder when the angle
    #     is opening (target maneuvering away) and damps the command as it
    #     closes (avoids overshoot right at convergence), per 전술.txt's
    #     "LOS angle은 작지만 계속 증가 -> 곧 벗어남 / 조금 크지만 빠르게
    #     감소 -> 수렴 중" point -- a pure P law on angle alone can't see this.
    # Roll stabilization once already close to a solution (2026-08-21, user
    # request): once ATA<=stabilize_ata_deg, cap the FINAL blended roll
    # command so a violent correction can't kick a near-converged aircraft
    # back out of alignment. Broader than weapons_track_command_scale (which
    # only applies inside weapons_track itself) -- this also covers
    # Offensive/Neutral moments that are already close but haven't formally
    # entered weapons_track yet.
    # Narrowed 10.0->5.0 (2026-08-21, user diagnosis from run16: stabilizing
    # this early was capping roll authority during ordinary maneuvering --
    # e.g. observed roll_cmd~0 despite a 60deg bank angle and ATA stuck at
    # 31deg not decreasing. Stabilization should only apply once a shot is
    # basically already lined up, not as a general damper.
    stabilize_ata_deg: float = 5.0
    stabilize_roll_limit: float = 0.5
    # Narrowed 15.0->5.0, same reasoning -- fine-tracking gain scheduling
    # and the stronger LOS-rate term should only kick in this close, not
    # during ordinary ATA>15deg turning.
    fine_track_threshold_deg: float = 5.0
    roll_norm_deg_fine: float = 15.0
    pitch_norm_deg_fine: float = 18.0
    # LOS-rate gain is SCHEDULED, not fixed (2026-08-21): weak far out (a
    # noisy single-tick rate estimate shouldn't dominate before the geometry
    # has stabilized), strong only once already inside fine_track_threshold_deg
    # (exactly where real PN-style guidance earns its keep -- damping the
    # final approach to convergence instead of overshooting past it).
    los_rate_gain_coarse: float = 0.15
    los_rate_gain_fine: float = 0.7
    los_rate_norm_degps: float = 30.0
    # EMA smoothing on the raw per-tick az/el rate estimate (2026-08-21) --
    # a single-tick finite difference over live telemetry is noisy; smooth
    # it before feeding the (now much stronger, in fine-track mode) rate
    # term so it doesn't jitter the command.
    los_rate_smoothing_alpha: float = 0.3
    # Neutral-state throttle is now rule-managed, not passed through from RL
    # (2026-08-21): RL has almost no training experience in this regime (the
    # live state-distribution telemetry showed Neutral dominates 76-90% of
    # live flight time vs only 18% in JSBSim), and a live log showed
    # throttle_cmd dropping to 0.0 for consecutive frames -- an energy-losing
    # RL artifact from an untrained regime, directly contradicting
    # 전술.txt's energy-management point. A fixed high-throttle default keeps
    # turn/climb performance available regardless of what RL's raw output does.
    neutral_throttle: float = 0.9

    # Roll commitment / anti-chattering (added 2026-08-21, user diagnosis
    # after run17: roll_cmd was observed flipping sign repeatedly within
    # ~2s during post-merge reacquisition -- a pure closed-loop P(+PN) law
    # on the instantaneous LOS azimuth has no memory, so it can reverse
    # direction every tick as az swings through fast-changing values at
    # close range, wasting time rebuilding bank angle each reversal instead
    # of completing one committed turn. Outside fine-track range
    # (|az|>=roll_deadband_deg), the SIGN of the desired turn direction
    # commits to ONE side, only flipping once az has clearly reversed past
    # deadband+hysteresis -- this part is unchanged and still prevents
    # direction chattering. Inside fine-track range the existing
    # proportional+PN law still applies unchanged (precision holding needs
    # small continuous corrections, not commitment).
    # roll_commit_hold_s REMOVED (2026-08-21 v14, user diagnosis from
    # run19/20): the old gate required BOTH reversed_clearly AND >=1.0s
    # elapsed before allowing a flip -- reversed_clearly alone (az past
    # deadband+hysteresis=13deg) is already a strong noise filter, so the
    # extra time-hold did nothing for genuine noise but actively forced the
    # WRONG committed direction to keep being flown whenever az reversed
    # decisively in under 1s (observed run19 frame2161: az=+170.2deg --
    # clearly calling for desired_sign=+1 -- but roll_cmd was still -0.6
    # because the previous commit hadn't held 1s yet). That 1s of banking
    # the wrong way is exactly what produced the near-inverted roll
    # excursions (own_roll_deg hit 178deg, then wrapped through +-180 to
    # recover) seen in both run19 and run20. reversed_clearly alone is kept
    # as the sole gate.
    roll_deadband_deg: float = 3.0
    roll_commit_hysteresis_deg: float = 10.0
    # Bank-angle hold (REPLACED 2026-08-21 v13, user diagnosis from run18:
    # the original design forced roll_cmd to +-roll_commit_magnitude=1.0 --
    # i.e. FULL roll input, not just a committed turn DIRECTION -- for the
    # entire hold window. That is "keep feeding in roll input", not "hold a
    # bank angle": once the aircraft actually reached a reasonable bank it
    # kept rolling straight through it (run18 live: median bank 83.6deg, 43%
    # of frames >90deg, well past a normal turn bank), bleeding lift/altitude
    # (4572m->726m over the run) and overshooting the LOS on every commit
    # instead of settling into a turn, exactly why ATA never converged
    # (v11->v12 was WORSE on ata/altitude despite direction-chatter being
    # fixed). Now only the direction (sign) is committed via the
    # deadband/hysteresis gate above; the target is a BANK ANGLE
    # (target_bank_deg on the committed side) and roll_cmd is a
    # proportional correction toward it (bank_error_deg = target - current,
    # wrap-safe). This self-limits: roll_cmd is large while far from the
    # target bank and shrinks toward 0 as the aircraft actually banks up to
    # it, instead of always being pinned at max until a timer expires.
    # target_bank_deg is now SCALED BY |az| (2026-08-21 v14, user diagnosis
    # from run19/20: a fixed 70deg target held the SAME hard bank all the
    # way through convergence -- both live runs show ATA genuinely
    # converging from ~127deg down to ~55deg over a sustained turn, then
    # blowing back open past 150deg because the wings never came up as the
    # LOS closed, overshooting straight through the solution). Now the
    # target bank ramps from target_bank_min_deg (right at the fine-track
    # boundary, easing toward wings-level for the precision handoff) up to
    # target_bank_max_deg (once |az| is at/above target_bank_full_az_deg) --
    # aggressive early in the turn, progressively relaxed as az closes so
    # the nose settles onto the target instead of sweeping through it.
    target_bank_max_deg: float = 70.0
    target_bank_min_deg: float = 20.0
    target_bank_full_az_deg: float = 60.0
    bank_hold_gain: float = 1.0
    bank_hold_norm_deg: float = 30.0
    # Excessive-climb pitch ceiling (added 2026-08-21 v14, user diagnosis
    # from run19/20: pitch_cmd was negative -- nose-up/pull -- 85% of the
    # time (mean -0.59) for the ENTIRE flight while the target's actual
    # world-frame altitude stayed BELOW ownship the whole time (mean
    # rel_alt -337m, worst -964m) -- own_pitch_deg (world nose angle)
    # climbed monotonically from ~6deg to ~73deg mean by the run's second
    # half and own_speed bled from ~208m/s down to ~67m/s, i.e. an unbounded
    # zoom climb with nothing in the pitch law that says "this isn't
    # converging, stop pulling". The existing altitude_safe_floor_m ceiling
    # only caps NOSE-DOWN (positive) pitch_cmd near the ground -- there was
    # no symmetric cap on sustained NOSE-UP (negative) pitch_cmd regardless
    # of how steep the aircraft's own pitch attitude already is. This adds
    # one: once own_pitch_deg (world attitude, not LOS elevation) exceeds
    # climb_ramp_start_deg, the most-negative pitch_cmd allowed is ramped
    # from -1.0 down toward max_climb_pitch_floor as pitch approaches
    # max_climb_pitch_deg -- caps how long a pure pull can keep steepening
    # the climb without forcing a real recovery, while never restricting
    # nose-down/descending commands.
    climb_ramp_start_deg: float = 30.0
    max_climb_pitch_deg: float = 60.0
    max_climb_pitch_floor: float = 0.0
    # Final EMA smoothing on the blended roll_cmd (2026-08-21) -- independent
    # of the commitment logic above (which prevents chattering DIRECTION),
    # this smooths chattering MAGNITUDE/high-frequency noise from the SAC
    # blend component and the fine-track proportional law.
    roll_cmd_smoothing_alpha: float = 0.4


class TacticalWrapperActionProvider(ActionProvider):
    """Wraps `inner` (typically an RLActionProvider) with a rule-based
    Neutral/Offensive/Overshoot state machine and lead-pursuit controller.
    See module docstring for the full design rationale."""

    def __init__(self, inner: ActionProvider, config: TacticalWrapperConfig | None = None):
        self.inner = inner
        self.cfg = config or TacticalWrapperConfig()
        self.geometry = GeometryInfo()
        self._state = "neutral"
        self._prev_distance: float | None = None
        self._prev_sim_time: float | None = None
        self._prev_target_position: np.ndarray | None = None
        self._prev_az: float | None = None
        self._prev_el: float | None = None
        self._smoothed_az_rate: float = 0.0
        self._smoothed_el_rate: float = 0.0
        self._high_cmd_start_time: float | None = None
        self._high_cmd_start_ata: float | None = None
        self._disengage_until: float | None = None
        self._roll_commit_sign: int = 0
        self._smoothed_roll_cmd: float = 0.0
        self.state_log: list[str] = []

    def reset(self, context: ActionContext | None = None) -> None:
        self.inner.reset(context)
        self._state = "neutral"
        self._prev_distance = None
        self._prev_sim_time = None
        self._prev_target_position = None
        self._prev_az = None
        self._prev_el = None
        self._smoothed_az_rate = 0.0
        self._smoothed_el_rate = 0.0
        self._high_cmd_start_time = None
        self._high_cmd_start_ata = None
        self._disengage_until = None
        self._roll_commit_sign = 0
        self._smoothed_roll_cmd = 0.0
        self.state_log = []

    def compute_action(self, context: ActionContext) -> ActionResult:
        ownship_state = context.ownship_state
        target_state = context.target_state
        cfg = self.cfg

        rl_result = self.inner.compute_action(context)
        rl_action = np.asarray(rl_result.action, dtype=np.float32)

        if ownship_state is None or target_state is None:
            return ActionResult(action=rl_action, source="tactical_wrapper_passthrough", confidence=rl_result.confidence)

        distance = float(self.geometry._get_distance(ownship_state, target_state))
        ata = abs(float(self.geometry._get_antenna_train_angle(ownship_state, target_state, False)))
        aa = abs(float(self.geometry._get_aspect_angle(ownship_state, target_state, False)))
        sim_time = float(ownship_state[StateIndex.SIM_TIME])

        closure_rate = 0.0
        if self._prev_distance is not None and self._prev_sim_time is not None:
            dt = sim_time - self._prev_sim_time
            if dt > 1e-3:
                closure_rate = (self._prev_distance - distance) / dt  # positive = closing

        self._state = self._classify_state(distance, ata, aa, closure_rate)

        # Lag pursuit is a TAIL-CHASE technique (2026-08-21 design-doc fix
        # for run13): it only makes sense when we're already roughly behind
        # the target (aa small) and overtaking too fast. run12/run13's
        # overshoot moments were both head-on/high-aspect (aa~115-178deg) --
        # applying lag there deliberately pulls the nose OFF an
        # already-converging solution right before the merge, exactly
        # matching run13's collapse (ata 5.45deg@Neutral -> 21.78deg@Overshoot).
        # Only engage lag aim when genuinely tail-chasing.
        is_tail_chase = aa < cfg.tail_chase_aa_deg
        rule_action = self._compute_lead_pursuit_action(
            ownship_state, target_state, sim_time, rl_action,
            lead_horizon_s=self._lead_horizon_for_distance(distance),
            lag_distance_m=(
                cfg.overshoot_lag_distance_m if self._state == "overshoot" and is_tail_chase else 0.0
            ),
        )

        if self._state == "offensive":
            w_rl = cfg.offensive_rl_weight
        elif self._state == "weapons_track":
            w_rl = cfg.weapons_track_rl_weight
            rule_action = rule_action.copy()
            rule_action[0] *= cfg.weapons_track_command_scale
            rule_action[1] *= cfg.weapons_track_command_scale
            rule_action[3] = float(np.clip(
                cfg.weapons_track_throttle_base
                - cfg.weapons_track_throttle_closure_gain
                * (distance - cfg.weapons_track_target_distance_m) / max(1.0, cfg.weapons_track_target_distance_m)
                - cfg.weapons_track_throttle_closure_gain * closure_rate / 100.0,
                cfg.weapons_track_throttle_min, cfg.weapons_track_throttle_max,
            ))
        elif self._state == "overshoot":
            w_rl = cfg.overshoot_rl_weight
            rule_action = rule_action.copy()
            rule_action[3] = cfg.overshoot_throttle
        else:
            w_rl = cfg.neutral_rl_weight
            rule_action = rule_action.copy()
            rule_action[3] = cfg.neutral_throttle
        w_rule = 1.0 - w_rl

        blended = w_rl * rl_action + w_rule * rule_action
        blended = np.clip(blended, [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]).astype(np.float32)

        altitude = float(ownship_state[StateIndex.ALT])
        ceiling = self._pitch_ceiling(altitude, self._state)
        blended[1] = min(blended[1], ceiling)
        own_pitch_deg = float(ownship_state[StateIndex.PITCH])
        climb_floor = self._climb_floor(own_pitch_deg)
        blended[1] = max(blended[1], climb_floor)

        # Roll stabilization once already close to a solution (2026-08-21,
        # user request) -- caps the roll swing so a correction can't kick a
        # near-converged aircraft back out of alignment.
        if ata <= cfg.stabilize_ata_deg:
            blended[0] = float(np.clip(blended[0], -cfg.stabilize_roll_limit, cfg.stabilize_roll_limit))
        # Low-altitude roll cap (2026-08-21, user request) -- wings-level
        # recovery takes priority over continuing to bank hard while
        # critically low, same threshold as the pitch ceiling.
        if altitude <= cfg.altitude_critical_floor_m:
            blended[0] = float(np.clip(blended[0], -cfg.low_altitude_roll_limit, cfg.low_altitude_roll_limit))

        state_label = self._state
        if self._update_stuck_detector(sim_time, ata, blended):
            blended = np.array(
                [blended[0] * cfg.disengage_roll_scale, cfg.disengage_pitch_cmd, 0.0, blended[3]],
                dtype=np.float32,
            )
            state_label = "disengage"

        current_target_position = np.asarray(position_ned(target_state), dtype=np.float64)
        self._prev_distance = distance
        self._prev_sim_time = sim_time
        self._prev_target_position = current_target_position
        self.state_log.append(state_label)

        return ActionResult(
            action=blended,
            source=f"tactical_wrapper[{state_label}]",
            confidence=rl_result.confidence,
            info={"state": state_label, "ata": ata, "distance": distance, "closure_rate": closure_rate},
        )

    def _lead_horizon_for_distance(self, distance: float) -> float:
        """Distance-scheduled lead horizon -- see config comment (fixes a
        2026-08-21 deadlock where a state-gated short horizon could never
        activate because the long horizon it replaced was itself preventing
        ATA from ever getting low enough to reach that state)."""
        cfg = self.cfg
        if distance <= cfg.weapons_track_distance_m:
            return cfg.lead_horizon_wez_s
        if distance <= cfg.lead_horizon_close_distance_m:
            return cfg.lead_horizon_close_s
        if distance <= cfg.lead_horizon_mid_distance_m:
            return cfg.lead_horizon_mid_s
        return cfg.lead_horizon_far_s

    def _pitch_ceiling(self, altitude: float, state: str) -> float:
        """Max allowed (nose-down, positive) pitch_cmd at this altitude --
        see TacticalWrapperConfig's altitude_safe_floor_m comment.
        Head-On Attack uses its own, much narrower band (see config
        comment) so it isn't fighting this ceiling during normal merges."""
        cfg = self.cfg
        if state == "head_on_attack":
            safe_floor = cfg.head_on_attack_altitude_safe_floor_m
            critical_floor = cfg.head_on_attack_altitude_critical_floor_m
        else:
            safe_floor = cfg.altitude_safe_floor_m
            critical_floor = cfg.altitude_critical_floor_m
        if altitude >= safe_floor:
            return 1.0
        if altitude <= critical_floor:
            return cfg.min_climb_pitch_ceiling
        span = max(1.0, safe_floor - critical_floor)
        frac = (altitude - critical_floor) / span
        return cfg.min_climb_pitch_ceiling + frac * (1.0 - cfg.min_climb_pitch_ceiling)

    def _climb_floor(self, own_pitch_deg: float) -> float:
        """Min allowed (nose-up, negative) pitch_cmd given the aircraft's
        OWN world-frame pitch attitude -- see TacticalWrapperConfig's
        climb_ramp_start_deg comment. Symmetric to _pitch_ceiling but keyed
        on own attitude, not altitude: caps sustained pull once already
        pitched steeply up, regardless of how high above the ground that
        is."""
        cfg = self.cfg
        pitch = abs(own_pitch_deg)
        if pitch <= cfg.climb_ramp_start_deg:
            return -1.0
        if pitch >= cfg.max_climb_pitch_deg:
            return cfg.max_climb_pitch_floor
        span = max(1.0, cfg.max_climb_pitch_deg - cfg.climb_ramp_start_deg)
        frac = (pitch - cfg.climb_ramp_start_deg) / span
        return -1.0 + frac * (cfg.max_climb_pitch_floor - (-1.0))

    def _update_stuck_detector(self, sim_time: float, ata: float, blended: np.ndarray) -> bool:
        """Returns True if currently in a disengage window (caller should
        override the action). See TacticalWrapperConfig's stuck_* comment."""
        cfg = self.cfg
        if self._disengage_until is not None:
            if sim_time < self._disengage_until:
                return True
            self._disengage_until = None
            self._high_cmd_start_time = None
            self._high_cmd_start_ata = None

        saturated = max(abs(float(blended[0])), abs(float(blended[1]))) >= cfg.stuck_saturation_threshold
        if not saturated:
            self._high_cmd_start_time = None
            self._high_cmd_start_ata = None
            return False

        if self._high_cmd_start_time is None:
            self._high_cmd_start_time = sim_time
            self._high_cmd_start_ata = ata
            return False

        stuck_elapsed = sim_time - self._high_cmd_start_time
        start_ata = self._high_cmd_start_ata if self._high_cmd_start_ata is not None else ata
        improved = start_ata - ata
        if stuck_elapsed >= cfg.stuck_duration_s and improved < cfg.stuck_ata_improvement_deg:
            self._disengage_until = sim_time + cfg.disengage_duration_s
            return True
        return False

    def _classify_state(self, distance: float, ata: float, aa: float, closure_rate: float) -> str:
        cfg = self.cfg
        # Weapons Track checked BEFORE Overshoot/Head-On (2026-08-21
        # design-doc fix -- a good firing solution during a fast approach
        # was previously getting grabbed by Overshoot before Weapons Track
        # ever got a chance to claim it). Weapons Track manages its own
        # throttle/closure once active, so it doesn't need their protection
        # layered on top.
        weapons_track_ata_threshold = (
            cfg.weapons_track_exit_ata_deg if self._state == "weapons_track" else cfg.weapons_track_enter_ata_deg
        )
        if ata <= weapons_track_ata_threshold and distance <= cfg.weapons_track_distance_m:
            return "weapons_track"
        # Overshoot only for a genuine tail-chase overtake (aa<head_on_attack_aa_deg).
        # REVISED 2026-08-21 (user diagnosis from run16): a dedicated
        # "head_on_attack" state with its own weaker, fixed control law
        # (short lead horizon, reduced RL weight) was over-applying tactical
        # intervention across ordinary ATA>15deg maneuvering -- a head-on
        # crossing at aa>=120deg is NOT an overtake, so it should just fall
        # through to normal Neutral/Offensive classification below (full
        # turn authority, same as before any of tonight's wrapper changes)
        # instead of getting its own subdued state at all.
        if closure_rate > cfg.overshoot_closure_mps and aa < cfg.head_on_attack_aa_deg:
            time_to_min_range = (distance - cfg.overshoot_min_range_m) / max(closure_rate, 1.0)
            if time_to_min_range < cfg.overshoot_time_to_min_range_s:
                return "overshoot"
        # neutral <-> offensive hysteresis. weapons_track counts as
        # "already engaged" for this purpose -- losing the weapons-track
        # lock (ata crept back up past 3deg, still under 10deg) should drop
        # back to offensive, not all the way to neutral.
        was_engaged = self._state in ("offensive", "weapons_track")
        if was_engaged:
            return "offensive" if ata <= cfg.exit_offensive_ata_deg else "neutral"
        return "offensive" if ata <= cfg.enter_offensive_ata_deg else "neutral"

    def _compute_lead_pursuit_action(
        self,
        ownship_state: np.ndarray,
        target_state: np.ndarray,
        sim_time: float,
        rl_action: np.ndarray,
        lead_horizon_s: float,
        lag_distance_m: float = 0.0,
    ) -> np.ndarray:
        cfg = self.cfg
        target_position = np.asarray(position_ned(target_state), dtype=np.float64)

        predicted_position = target_position
        if self._prev_target_position is not None and self._prev_sim_time is not None:
            dt = sim_time - self._prev_sim_time
            if dt > 1e-3:
                velocity = (target_position - self._prev_target_position) / dt
                speed = float(np.linalg.norm(velocity))
                if speed > cfg.lead_velocity_clip_mps:
                    velocity = velocity * (cfg.lead_velocity_clip_mps / speed)
                if lag_distance_m > 0.0 and speed > 1e-3:
                    # Overshoot: aim BEHIND the target's direction of travel
                    # instead of ahead of it -- see overshoot_lag_distance_m
                    # comment.
                    unit_velocity = velocity / speed
                    predicted_position = target_position - unit_velocity * lag_distance_m
                else:
                    predicted_position = target_position + velocity * lead_horizon_s

        synthetic_target = np.zeros_like(target_state)
        synthetic_target[:3] = predicted_position
        az, el = self.geometry._get_los_angle(ownship_state, synthetic_target)
        az = float(az)
        el = float(el)

        # Gain scheduling: tighter norm (more command per degree) once
        # already inside fine_track_threshold_deg -- see config comment.
        roll_norm = cfg.roll_norm_deg_fine if abs(az) < cfg.fine_track_threshold_deg else cfg.roll_norm_deg
        pitch_norm = cfg.pitch_norm_deg_fine if abs(el) < cfg.fine_track_threshold_deg else cfg.pitch_norm_deg

        # LOS-rate (PN-style) term -- see config comment. az wraps at
        # +-180deg so its finite difference must go through the shortest
        # angular path; el (-90..90, never wraps) is a plain difference.
        # Raw per-tick rate is noisy over live telemetry, so it's EMA-smoothed
        # before use, and its gain is SCHEDULED (weak far out, strong only in
        # fine-track mode) -- see config comment.
        raw_az_rate = 0.0
        raw_el_rate = 0.0
        if self._prev_az is not None and self._prev_sim_time is not None:
            dt = sim_time - self._prev_sim_time
            if dt > 1e-3:
                az_delta = ((az - self._prev_az + 180.0) % 360.0) - 180.0
                raw_az_rate = az_delta / dt
                raw_el_rate = (el - self._prev_el) / dt

        alpha = cfg.los_rate_smoothing_alpha
        self._smoothed_az_rate = alpha * raw_az_rate + (1.0 - alpha) * self._smoothed_az_rate
        self._smoothed_el_rate = alpha * raw_el_rate + (1.0 - alpha) * self._smoothed_el_rate

        in_fine_track = abs(az) < cfg.fine_track_threshold_deg or abs(el) < cfg.fine_track_threshold_deg
        los_rate_gain = cfg.los_rate_gain_fine if in_fine_track else cfg.los_rate_gain_coarse

        roll_cmd_raw = float(np.clip(
            cfg.roll_gain * az / roll_norm + los_rate_gain * self._smoothed_az_rate / cfg.los_rate_norm_degps,
            -1.0, 1.0,
        ))

        # Roll commitment / bank-angle hold (2026-08-21 v13/v14, see
        # TacticalWrapperConfig's target_bank_max_deg comment) -- outside
        # fine-track range, commit hard to one TURN DIRECTION (deadband +
        # hysteresis gate, no time-hold -- see roll_commit_hysteresis_deg
        # comment) but then hold a target BANK ANGLE (scaled by |az|) on
        # that side via proportional control, instead of forcing full roll
        # input for a fixed window. Inside fine-track range the precise law
        # above applies directly (precision holding needs small continuous
        # corrections, not commitment).
        if abs(az) < cfg.fine_track_threshold_deg:
            roll_cmd = roll_cmd_raw
            self._roll_commit_sign = 1 if az >= 0 else -1
        else:
            if az > cfg.roll_deadband_deg:
                desired_sign = 1
            elif az < -cfg.roll_deadband_deg:
                desired_sign = -1
            else:
                desired_sign = self._roll_commit_sign or (1 if roll_cmd_raw >= 0 else -1)
            if self._roll_commit_sign != 0 and desired_sign != self._roll_commit_sign:
                reversed_clearly = abs(az) > cfg.roll_deadband_deg + cfg.roll_commit_hysteresis_deg
                if not reversed_clearly:
                    desired_sign = self._roll_commit_sign
            self._roll_commit_sign = desired_sign
            az_frac = float(np.clip(
                (abs(az) - cfg.roll_deadband_deg)
                / max(1.0, cfg.target_bank_full_az_deg - cfg.roll_deadband_deg),
                0.0, 1.0,
            ))
            target_bank_mag = cfg.target_bank_min_deg + az_frac * (cfg.target_bank_max_deg - cfg.target_bank_min_deg)
            target_bank_deg = target_bank_mag * self._roll_commit_sign
            current_bank_deg = float(ownship_state[StateIndex.ROLL])
            bank_error_deg = ((target_bank_deg - current_bank_deg + 180.0) % 360.0) - 180.0
            roll_cmd = float(np.clip(
                cfg.bank_hold_gain * bank_error_deg / cfg.bank_hold_norm_deg, -1.0, 1.0,
            ))

        # Final EMA smoothing (separate from the commitment logic above,
        # which prevents DIRECTION chattering -- this smooths magnitude/
        # high-frequency noise, mainly relevant inside fine-track range
        # where roll_cmd is still the raw proportional+PN value).
        alpha_r = cfg.roll_cmd_smoothing_alpha
        self._smoothed_roll_cmd = alpha_r * roll_cmd + (1.0 - alpha_r) * self._smoothed_roll_cmd
        roll_cmd = float(np.clip(self._smoothed_roll_cmd, -1.0, 1.0))

        # pitch_cmd=+1.0 is nose-DOWN (verified convention, single_agent_env.py
        # _apply_safety_override docstring) -- positive el (target above) must
        # command negative pitch_cmd (pull up).
        pitch_cmd = float(np.clip(
            -cfg.pitch_gain * el / pitch_norm - los_rate_gain * self._smoothed_el_rate / cfg.los_rate_norm_degps,
            -1.0, 1.0,
        ))
        yaw_cmd = float(np.clip(cfg.yaw_gain * az / cfg.roll_norm_deg, -1.0, 1.0))
        throttle_cmd = float(rl_action[3]) if len(rl_action) > 3 else 0.7

        self._prev_az = az
        self._prev_el = el

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


__all__ = ["TacticalWrapperConfig", "TacticalWrapperActionProvider"]
