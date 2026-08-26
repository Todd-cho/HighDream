"""Measured-envelope axis arbitration for the W100 live baseline.

This controller deliberately does not add another guidance law.  The live
logs show that W100 produced the best repeatable nose alignment, while later
controllers lost that result when several guidance proposals fought over the
bank axis.  W114 therefore keeps W100 as the sole bank/rudder owner and only
adds two measured-plant safeguards:

* fill the weak elevator interval while the lift plane is first established;
* schedule one throttle owner around the live 180--220 m/s turn envelope and
  unload elevator below that envelope.

All changes are visible in telemetry and can be removed by selecting W100.
"""

from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.control_arbitration import diagnose_control_ownership
from dogfight.ai.phase_manager import classify_phase
from dogfight.sim.state_schema import StateIndex


class DistanceHandoffActionProvider(ActionProvider):
    """W119 acquisition with a hysteretic, continuous handoff to pure RL."""

    def __init__(
        self,
        rule: ActionProvider,
        rl: ActionProvider,
        enter_start_m: float = 1900.0,
        enter_full_m: float = 1500.0,
        exit_start_m: float = 1900.0,
        exit_full_m: float = 2200.0,
    ) -> None:
        self.rule = rule
        self.rl = rl
        self.enter_start_m = enter_start_m
        self.enter_full_m = enter_full_m
        self.exit_start_m = exit_start_m
        self.exit_full_m = exit_full_m
        self._near_latched = False

    def reset(self, context: ActionContext | None = None) -> None:
        self.rule.reset(context)
        self.rl.reset(context)
        self._near_latched = False

    @staticmethod
    def _smoothstep(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def compute_action(self, context: ActionContext) -> ActionResult:
        # Both policies run every tick. The SAC policy therefore never wakes
        # up cold at the 1500m boundary, and all internal rule estimators also
        # remain continuous while RL owns the aircraft.
        rule_result = self.rule.compute_action(context)
        rl_result = self.rl.compute_action(context)
        rule_action = np.asarray(rule_result.action, dtype=np.float32)
        rl_action = np.asarray(rl_result.action, dtype=np.float32)
        distance = float(rule_result.info.get("distance", 99999.0))

        if not self._near_latched and distance <= self.enter_full_m:
            self._near_latched = True
        elif self._near_latched and distance >= self.exit_full_m:
            self._near_latched = False

        if self._near_latched:
            # Full RL through 1900m on the outbound leg, then fade back to
            # W119 by 2200m. This prevents 1500m boundary chatter.
            rl_weight = 1.0 - self._smoothstep(
                (distance - self.exit_start_m)
                / max(self.exit_full_m - self.exit_start_m, 1.0)
            )
        else:
            # Inbound acquisition: W119 until 1900m, full RL by 1500m.
            rl_weight = self._smoothstep(
                (self.enter_start_m - distance)
                / max(self.enter_start_m - self.enter_full_m, 1.0)
            )

        action = ((1.0 - rl_weight) * rule_action + rl_weight * rl_action)
        action = np.clip(
            action, [-0.8, -1.0, -1.0, 0.0], [0.8, 1.0, 1.0, 1.0]
        ).astype(np.float32)
        info = dict(rule_result.info)
        info.update({
            "distance_handoff_active": 0.0 < rl_weight < 1.0,
            "distance_handoff_near_latched": self._near_latched,
            "distance_handoff_rl_weight": rl_weight,
            "distance_handoff_distance_m": distance,
            "distance_handoff_rule_roll": float(rule_action[0]),
            "distance_handoff_rule_pitch": float(rule_action[1]),
            "distance_handoff_rule_yaw": float(rule_action[2]),
            "distance_handoff_rule_throttle": float(rule_action[3]),
            "distance_handoff_rl_roll": float(rl_action[0]),
            "distance_handoff_rl_pitch": float(rl_action[1]),
            "distance_handoff_rl_yaw": float(rl_action[2]),
            "distance_handoff_rl_throttle": float(rl_action[3]),
            "bank_owner": "distance_blend" if rl_weight < 1.0 else "pure_v8",
            "pitch_owner": "distance_blend" if rl_weight < 1.0 else "pure_v8",
            "throttle_owner": "distance_blend" if rl_weight < 1.0 else "pure_v8",
        })
        return ActionResult(
            action=action,
            source=f"w119rl1500({rl_weight:.3f})",
            confidence=(1.0 - rl_weight) * rule_result.confidence
            + rl_weight * rl_result.confidence,
            info=info,
        )

    def close(self) -> None:
        self.rule.close()
        self.rl.close()


class MaxRateTurnActionProvider(ActionProvider):
    """Single-purpose sustained max-rate turn experiment for live plant data."""

    def __init__(self, base: ActionProvider) -> None:
        self.base = base
        self._commit_sign = 0.0
        self._target_turn_candidate = 0.0
        self._target_turn_since: float | None = None
        self._matched_target_turn = False

    def reset(self, context: ActionContext | None = None) -> None:
        self.base.reset(context)
        self._commit_sign = 0.0
        self._target_turn_candidate = 0.0
        self._target_turn_since = None
        self._matched_target_turn = False

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.base.compute_action(context)
        action = np.asarray(result.action, dtype=np.float32).copy()
        info = dict(result.info)
        own = context.ownship_state
        now = float(own[StateIndex.SIM_TIME]) if own is not None else 0.0
        speed = float(own[StateIndex.KCAS]) if own is not None else 0.0
        bank = float(info.get("current_bank", own[StateIndex.ROLL] if own is not None else 0.0))
        target_bank = float(info.get("target_bank", 0.0))
        target_yaw_rate = float(info.get("target_yaw_rate", 0.0))

        if self._commit_sign == 0.0:
            self._commit_sign = 1.0 if target_bank >= 0.0 else -1.0

        # After the merge, identify the opponent's actual sustained turn once
        # and match that direction. Then lock it: no LOS-wrap chatter and no
        # repeated reversal proposals are allowed in this experiment.
        if (
            not self._matched_target_turn
            and bool(info.get("first_merge_passed", False))
            and abs(target_yaw_rate) >= 5.0
        ):
            candidate = 1.0 if target_yaw_rate > 0.0 else -1.0
            if candidate != self._target_turn_candidate:
                self._target_turn_candidate = candidate
                self._target_turn_since = now
            elif self._target_turn_since is not None and now - self._target_turn_since >= 0.5:
                self._commit_sign = candidate
                self._matched_target_turn = True

        desired_bank = self._commit_sign * 82.0
        bank_error = (desired_bank - bank + 180.0) % 360.0 - 180.0
        roll_rate = float(info.get("measured_roll_rate", info.get("roll_rate", 0.0)))
        action[0] = float(np.clip(
            0.018 * bank_error - 0.005 * roll_rate, -0.80, 0.80
        ))

        # The live plant produced its best sustained rate at 240--300m/s.
        # Below 220m/s unload just enough to regain energy; otherwise use the
        # full measured pull instead of allowing tactical modes to release it.
        if speed < 200.0:
            action[1] = -0.55
        elif speed < 220.0:
            action[1] = -0.75
        elif abs(bank) >= 45.0:
            action[1] = -1.0
        else:
            action[1] = min(float(action[1]), -0.55)

        target_speed = 275.0
        action[3] = float(np.clip(0.72 + 0.014 * (target_speed - speed), 0.25, 1.0))
        action = np.clip(
            action, [-0.8, -1.0, -1.0, 0.0], [0.8, 1.0, 1.0, 1.0]
        ).astype(np.float32)
        info.update({
            "bank_owner": "max_rate_turn",
            "pitch_owner": "max_rate_turn",
            "throttle_owner": "max_rate_turn",
            "max_rate_turn_active": True,
            "max_rate_turn_commit_sign": self._commit_sign,
            "max_rate_turn_matched_target": self._matched_target_turn,
            "max_rate_turn_target_yaw_rate_degps": target_yaw_rate,
            "max_rate_turn_target_bank_deg": desired_bank,
            "max_rate_turn_target_speed_mps": target_speed,
        })
        return ActionResult(
            action=action,
            source=f"w124<{result.source}>",
            confidence=result.confidence,
            info=info,
        )

    def close(self) -> None:
        self.base.close()


class PostMergePullBoostActionProvider(ActionProvider):
    """Restore the measured successful W41 post-merge pull for 15 seconds."""

    def __init__(self, base: ActionProvider, duration_s: float = 15.0) -> None:
        self.base = base
        self.duration_s = duration_s
        self._merge_time: float | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self.base.reset(context)
        self._merge_time = None

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.base.compute_action(context)
        action = np.asarray(result.action, dtype=np.float32).copy()
        info = dict(result.info)
        own = context.ownship_state
        now = float(own[StateIndex.SIM_TIME]) if own is not None else 0.0
        speed = float(own[StateIndex.KCAS]) if own is not None else 0.0
        bank = abs(float(info.get(
            "current_bank", own[StateIndex.ROLL] if own is not None else 0.0
        )))
        if self._merge_time is None and bool(info.get("first_merge_passed", False)):
            self._merge_time = now
        age = now - self._merge_time if self._merge_time is not None else -1.0
        active = 0.0 <= age <= self.duration_s

        if active:
            # Do not invent another bank direction: W41 already established
            # 82deg correctly in the failed retest. Only restore the pull that
            # changed from -0.96 (old successful run) to -0.33 (new run).
            if bank < 45.0:
                boost_pitch = -0.55
            elif speed >= 190.0:
                boost_pitch = -0.96
            elif speed >= 170.0:
                boost_pitch = -0.78
            else:
                boost_pitch = -0.55
            action[1] = min(float(action[1]), boost_pitch)
            action[3] = 1.0
            info["pitch_owner"] = "postmerge_pull_boost"
            info["throttle_owner"] = "postmerge_pull_boost"
        else:
            boost_pitch = 0.0

        action = np.clip(
            action, [-0.8, -1.0, -1.0, 0.0], [0.8, 1.0, 1.0, 1.0]
        ).astype(np.float32)
        info.update({
            "postmerge_pull_boost_active": active,
            "postmerge_pull_boost_age_s": age,
            "postmerge_pull_boost_cmd": boost_pitch,
            "postmerge_pull_boost_duration_s": self.duration_s,
        })
        return ActionResult(
            action=action,
            source=f"w125<{result.source}>",
            confidence=result.confidence,
            info=info,
        )

    def close(self) -> None:
        self.base.close()


class MeasuredEnvelopeController(ActionProvider):
    """W100 geometry with exclusive per-axis measured-envelope authority."""

    def __init__(self, base: ActionProvider, name: str = "w114") -> None:
        self.base = base
        self.name = name
        self.optimized = name in ("w115", "w116", "w117", "w118", "w119", "w120", "w121")
        self.confirmed_overshoot_only = name in ("w116", "w117", "w118", "w119", "w120", "w121")
        self.contextual_turn_speed = name in ("w117", "w118", "w119", "w120", "w121")
        self.vertical_energy_loop = name in ("w118", "w119")
        self.safe_los_reacquire = name == "w119"
        self.terminal_lift_vector = name in ("w120", "w121")
        self.rear_quarter_capture = name == "w121"
        self._start_time: float | None = None
        self._last_time: float | None = None
        self._throttle_cmd: float | None = None
        self._speed_recovery_active = False
        self._rear_threat_since: float | None = None
        self._vertical_phase = "inactive"  # telemetry-compatible counterattack phase
        self._vertical_phase_since = 0.0
        self._vertical_pitch_cmd = -0.82
        self._vertical_cooldown_until = 0.0
        self._counter_turn_sign = 1.0
        self._counter_min_distance = 99999.0
        self._counter_overshoot_detected = False
        self._counter_used = False
        self._initial_altitude: float | None = None
        self._vertical_speed_ema = 0.0
        self._vertical_recovery_active = False
        self._vertical_recovery_since = 0.0
        self._los_disagree_since: float | None = None
        self._los_reacquire_active = False
        self._los_reacquire_since = 0.0
        self._los_reacquire_sign = 0.0
        self._los_reacquire_used = False
        self._terminal_lift_active = False
        self._terminal_lift_since = 0.0
        self._terminal_lift_cooldown_until = 0.0

    def reset(self, context: ActionContext | None = None) -> None:
        self.base.reset(context)
        self._start_time = None
        self._last_time = None
        self._throttle_cmd = None
        self._speed_recovery_active = False
        self._rear_threat_since = None
        self._vertical_phase = "inactive"
        self._vertical_phase_since = 0.0
        self._vertical_pitch_cmd = -0.82
        self._vertical_cooldown_until = 0.0
        self._counter_turn_sign = 1.0
        self._counter_min_distance = 99999.0
        self._counter_overshoot_detected = False
        self._counter_used = False
        self._initial_altitude = None
        self._vertical_speed_ema = 0.0
        self._vertical_recovery_active = False
        self._vertical_recovery_since = 0.0
        self._los_disagree_since = None
        self._los_reacquire_active = False
        self._los_reacquire_since = 0.0
        self._los_reacquire_sign = 0.0
        self._los_reacquire_used = False
        self._terminal_lift_active = False
        self._terminal_lift_since = 0.0
        self._terminal_lift_cooldown_until = 0.0

    @staticmethod
    def _number(value, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.base.compute_action(context)
        action = np.asarray(result.action, dtype=np.float32).copy()
        info = dict(result.info)

        own = context.ownship_state
        now = self._number(own[StateIndex.SIM_TIME]) if own is not None else 0.0
        speed = self._number(own[StateIndex.KCAS]) if own is not None else 0.0
        target = context.target_state
        target_speed = (
            self._number(target[StateIndex.KCAS]) if target is not None else 0.0
        )
        current_bank = self._number(info.get("current_bank"), self._number(
            own[StateIndex.ROLL] if own is not None else 0.0
        ))
        bank = abs(current_bank)
        ata = self._number(info.get("ata"), 180.0)
        threat_ata = self._number(info.get("threat_ata"), 0.0)
        distance = self._number(info.get("distance"), 99999.0)
        closure = self._number(info.get("closure_rate"), 0.0)
        phase = classify_phase(info).value

        if self._start_time is None:
            self._start_time = now
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.25))
        self._last_time = now
        elapsed = max(0.0, now - self._start_time)
        own_altitude = self._number(info.get("own_alt"), self._number(
            own[StateIndex.ALT] if own is not None else 0.0
        ))
        if self._initial_altitude is None and own_altitude > 0.0:
            self._initial_altitude = own_altitude
        altitude_loss = max(
            0.0, (self._initial_altitude or own_altitude) - own_altitude
        )
        measured_vertical_speed = self._number(info.get("vertical_speed"), 0.0)
        # 0.1s telemetry is noisy; a first-order filter prevents bank-owner
        # chatter around the recovery thresholds.
        alpha = 1.0 - math.exp(-dt / 0.8) if dt > 0.0 else 1.0
        self._vertical_speed_ema += alpha * (
            measured_vertical_speed - self._vertical_speed_ema
        )
        target_bank = self._number(info.get("target_bank"), 0.0)
        target_sign = (
            1.0 if target_bank > 0.0
            else -1.0 if target_bank < 0.0 else 0.0
        )
        los_az = self._number(info.get("los_az"), 0.0)
        aim_az = self._number(info.get("aim_az"), los_az)
        los_sign = 1.0 if los_az > 0.0 else -1.0 if los_az < 0.0 else 0.0
        aim_sign = 1.0 if aim_az > 0.0 else -1.0 if aim_az < 0.0 else 0.0
        wrapped_aim_los_delta = (aim_az - los_az + 180.0) % 360.0 - 180.0
        safe_disagreement = (
            self.safe_los_reacquire and not self._los_reacquire_used
            and bool(info.get("first_merge_passed", False))
            and los_sign != 0.0 and aim_sign != 0.0 and los_sign != aim_sign
            and abs(wrapped_aim_los_delta) >= 25.0
            and ata >= 100.0 and threat_ata >= 45.0 and distance >= 1200.0
        )
        if safe_disagreement:
            if self._los_disagree_since is None:
                self._los_disagree_since = now
        else:
            self._los_disagree_since = None
        los_disagree_age = (
            max(0.0, now - self._los_disagree_since)
            if self._los_disagree_since is not None else 0.0
        )
        if (
            not self._los_reacquire_active and safe_disagreement
            and los_disagree_age >= 0.3
        ):
            self._los_reacquire_active = True
            self._los_reacquire_since = now
            self._los_reacquire_sign = los_sign
            self._los_reacquire_used = True
        if self._los_reacquire_active and (
            now - self._los_reacquire_since >= 2.2 or ata <= 70.0
        ):
            self._los_reacquire_active = False

        # Detect a genuinely persistent rear-quarter trap. A transient nose
        # crossing must not start a scissors/counterattack sequence.
        if self.confirmed_overshoot_only:
            # W116: a brake is useful only when the attacker is close and
            # rapidly closing.  A generic rear-quarter condition caused W115
            # to throw away energy three times while the attacker merely
            # followed the reversal.
            rear_trap = (
                bool(info.get("first_merge_passed", False))
                and ata >= 100.0 and threat_ata <= 15.0
                and 400.0 <= distance <= 1100.0 and closure >= 45.0
                and not self._counter_used
            )
            required_rear_time = 0.5
        else:
            rear_trap = (
                bool(info.get("first_merge_passed", False))
                and ata >= 90.0 and threat_ata <= 30.0 and distance <= 2200.0
            )
            required_rear_time = 3.0
        if rear_trap:
            if self._rear_threat_since is None:
                self._rear_threat_since = now
        else:
            self._rear_threat_since = None
        rear_trap_age = (
            max(0.0, now - self._rear_threat_since)
            if self._rear_threat_since is not None else 0.0
        )
        if (
            self.optimized and self._vertical_phase == "inactive"
            and rear_trap_age >= required_rear_time
            and now >= self._vertical_cooldown_until
        ):
            self._vertical_phase = "brake_break"
            self._vertical_phase_since = now
            current_sign = 1.0 if current_bank >= 0.0 else -1.0
            self._counter_turn_sign = (
                target_sign if target_sign != 0.0 else current_sign
            )
            self._counter_min_distance = distance
            self._counter_overshoot_detected = False
            if self.confirmed_overshoot_only:
                self._counter_used = True
        phase_age = max(0.0, now - self._vertical_phase_since)
        if self._vertical_phase != "inactive":
            self._counter_min_distance = min(
                self._counter_min_distance, distance
            )
            if self.confirmed_overshoot_only:
                # Range opening or negative closure is not proof that the
                # attacker's gun solution is broken.  Run0233 had both while
                # threat ATA remained 1--12 degrees.  Require nose separation.
                self._counter_overshoot_detected = (
                    self._counter_overshoot_detected or threat_ata >= 35.0
                )
            else:
                self._counter_overshoot_detected = (
                    self._counter_overshoot_detected
                    or threat_ata >= 40.0
                    or closure <= -10.0
                    or distance >= self._counter_min_distance + 80.0
                )
        if self._vertical_phase == "brake_break":
            if self.confirmed_overshoot_only and self._counter_overshoot_detected:
                self._vertical_phase = "cross_reversal"
                self._vertical_phase_since = now
                phase_age = 0.0
            elif self.confirmed_overshoot_only and phase_age >= 1.2:
                # No overshoot: never cross wings-level in front of a bandit.
                self._vertical_phase = "inactive"
                self._vertical_phase_since = now
                self._vertical_cooldown_until = now + 9999.0
                self._rear_threat_since = None
                phase_age = 0.0
            elif not self.confirmed_overshoot_only and phase_age >= 1.8:
                self._vertical_phase = "cross_reversal"
                self._vertical_phase_since = now
                phase_age = 0.0
        elif (
            self._vertical_phase == "cross_reversal"
            and (
                (phase_age >= 0.6 and self._counter_overshoot_detected)
                or phase_age >= (1.4 if self.confirmed_overshoot_only else 2.4)
            )
        ):
            self._vertical_phase = "attack_conversion"
            self._vertical_phase_since = now
            phase_age = 0.0
        elif self._vertical_phase == "attack_conversion" and phase_age >= 3.0:
            self._vertical_phase = "inactive"
            self._vertical_phase_since = now
            self._vertical_cooldown_until = now + 8.0
            self._rear_threat_since = None
            phase_age = 0.0

        speed_deficit = target_speed + 2.0 - speed if target_speed >= 50.0 else 0.0
        immediate_threat = (
            threat_ata <= 25.0 and distance <= 1800.0
        )
        if self.optimized:
            if self._speed_recovery_active:
                self._speed_recovery_active = speed_deficit > 8.0
            else:
                self._speed_recovery_active = speed_deficit > 18.0
        else:
            self._speed_recovery_active = False

        # Bank and rudder remain byte-for-byte W100 commands.  W100 is the
        # only live controller that repeatedly reached ~2 degrees ATA, so no
        # new estimator or tactical proposal may take either axis here.
        info["bank_owner"] = "w100_course_geometry"
        info["bank_authority"] = 1.0

        # W115 roll plant correction.  Live W100 reduced roll command to
        # 0.22 at t=1 s while the opponent already held ~60 degrees bank.
        # Commit long enough to establish the lift plane, then retain a small
        # feed-forward command so the steady bank does not sag to 77 degrees.
        opening_roll_active = False
        bank_hold_active = False
        roll_schedule_cmd = float(action[0])
        if self.optimized and target_sign != 0.0:
            opening_roll_active = (
                not bool(info.get("first_merge_passed", False))
                and elapsed <= 4.0 and ata >= 75.0 and bank < 66.0
            )
            if opening_roll_active:
                # Pulse-derived roll gain is 150--170 deg/s/unit.  These
                # commands reproduce the opponent's measured 60deg@1s and
                # 77deg@2s without the old full-command overshoot.
                scheduled_mag = 0.38 if bank < 45.0 else 0.28
                roll_schedule_cmd = target_sign * scheduled_mag
                if abs(float(action[0])) < scheduled_mag:
                    action[0] = roll_schedule_cmd
            bank_hold_active = (
                not opening_roll_active
                and abs(target_bank) >= 78.0
                and target_bank * self._number(info.get("current_bank"), 0.0) > 0.0
                and 68.0 <= bank < 81.5
            )
            if bank_hold_active:
                scheduled_mag = 0.15 if bank < 78.0 else 0.12
                roll_schedule_cmd = target_sign * scheduled_mag
                if abs(float(action[0])) < scheduled_mag:
                    action[0] = roll_schedule_cmd

        # Energy-manoeuvrability schedule: throttle, bank and elevator must
        # agree.  Holding 82deg bank while asking for acceleration preserves
        # induced drag and cannot close a 30--50m/s speed deficit.  Roll out
        # only as much as needed to regain parity.  An immediate gun threat
        # bypasses this schedule and retains W100's maximum defensive turn.
        energy_bank_active = (
            self.optimized and self._speed_recovery_active
            and ata > 25.0 and not immediate_threat and target_sign != 0.0
            and target_bank * current_bank > 0.0 and bank >= 30.0
        )
        energy_bank_target = abs(target_bank)
        if energy_bank_active:
            # Continuous schedule avoids 64/72/78-degree chatter when the
            # measured speed deficit crosses a discrete boundary.
            energy_bank_target = float(np.interp(
                speed_deficit,
                [8.0, 18.0, 35.0, 60.0],
                [78.0, 72.0, 64.0, 60.0],
            ))
            desired_bank = target_sign * energy_bank_target
            bank_error = (desired_bank - current_bank + 180.0) % 360.0 - 180.0
            roll_rate = self._number(info.get("measured_roll_rate"), 0.0)
            roll_schedule_cmd = float(np.clip(
                0.012 * bank_error - 0.006 * roll_rate, -0.60, 0.60
            ))
            action[0] = roll_schedule_cmd
            info["bank_owner"] = "measured_energy_bank"
            # This command supersedes the normal hold feed-forward; telemetry
            # must not claim two simultaneous bank owners.
            bank_hold_active = False

        # The measured plant takes roughly 3.5 s to establish an 80-degree
        # bank.  W100's quadratic bank gate supplies too little elevator in
        # the middle of that roll.  Fill only the missing pull; never weaken a
        # stronger W100 command and never apply it before a usable lift plane.
        opening_fill_active = (
            not bool(info.get("first_merge_passed", False))
            and elapsed <= 4.0
            and ata >= 80.0
            and speed >= 185.0
            and 32.0 <= bank <= 62.0
        )
        opening_fill_cmd = 0.0
        if opening_fill_active:
            fraction = float(np.clip((bank - 32.0) / 30.0, 0.0, 1.0))
            opening_fill_cmd = -(0.34 + 0.24 * fraction)
            action[1] = min(float(action[1]), opening_fill_cmd)

        # Below 180 m/s, additional pull only makes the turn-rate deficit
        # worse.  Continuously reduce the maximum negative elevator demand;
        # full W100 pitch authority returns by 190 m/s.
        unload_release_speed = 200.0 if self.optimized else 190.0
        energy_unload_active = (
            speed > 1.0 and speed < unload_release_speed
            and not (self.optimized and immediate_threat)
        )
        pitch_floor = -1.0
        if energy_unload_active:
            if self.optimized:
                pitch_floor = float(np.interp(
                    speed, [150.0, 175.0, 190.0, 200.0],
                    [-0.22, -0.42, -0.72, -1.0],
                ))
            else:
                pitch_floor = float(np.interp(
                    speed, [145.0, 165.0, 180.0, 190.0],
                    [-0.25, -0.45, -0.72, -1.0],
                ))
            action[1] = max(float(action[1]), pitch_floor)
        # Full elevator at high bank can consume all available thrust as
        # induced drag.  When the bandit has opened a measured energy gap,
        # unload just enough to accelerate; hysteresis prevents mode chatter.
        if self._speed_recovery_active and ata > 25.0 and not immediate_threat:
            recovery_floor = float(np.interp(
                speed_deficit,
                [8.0, 18.0, 35.0, 60.0],
                [-0.75, -0.55, -0.38, -0.30],
            ))
            pitch_floor = max(pitch_floor, recovery_floor)
            action[1] = max(float(action[1]), pitch_floor)
            energy_unload_active = True

        if energy_unload_active:
            pitch_owner = "measured_energy_unload"
        elif opening_fill_active:
            pitch_owner = "measured_opening_fill"
        else:
            pitch_owner = "w100_gamma_pull"
        info["pitch_owner"] = pitch_owner
        info["pitch_authority"] = 1.0

        # A single phase-dependent speed owner.  Live W100/W89 data place the
        # useful sustained-turn band at 180--220 m/s.  Do not chase one fixed
        # corner speed in every geometry: retain energy in defence/reacquire,
        # then trade a little speed for radius only after a safe alignment.
        safe_terminal = (
            ata <= 20.0 and threat_ata >= 40.0 and distance <= 2000.0
        )
        if phase == "defensive":
            commanded_speed = 220.0 if self.optimized else 215.0
            throttle_base = 0.88
        elif phase in ("reacquire", "opening_merge", "headon_neutral"):
            commanded_speed = 215.0 if self.optimized else 205.0
            throttle_base = 0.72
        elif safe_terminal:
            commanded_speed = 205.0 if self.optimized else 190.0
            throttle_base = 0.58 if self.optimized else 0.48
        else:
            commanded_speed = 210.0 if self.optimized else 195.0
            throttle_base = 0.68 if self.optimized else 0.60

        # W115 live energy parity: the opponent speed is available every
        # synchronized telemetry pair.  W117 makes this contextual.  The
        # run0234 plant data showed that blindly matching a 330m/s bandit in
        # a sustained circle reduced own turn rate, whereas 254--280m/s
        # produced 16.5--17deg/s and could cut inside the target circle.
        sustained_turn_cut = (
            self.contextual_turn_speed
            and bool(info.get("first_merge_passed", False))
            and ata >= 50.0 and distance <= 4500.0 and bank >= 65.0
        )
        if sustained_turn_cut:
            # Use the lower end while directly threatened; retain a little
            # more energy in neutral/reacquire geometry.  This is one speed
            # owner, not a second bank proposal.
            commanded_speed = 260.0 if threat_ata <= 35.0 else 275.0
        elif self.optimized and target_speed >= 50.0:
            # Once aligned or outside the turning fight, regain the original
            # pursuit requirement. W120 requests a 10m/s closing margin;
            # earlier modes retain their measured 2m/s parity target.
            pursuit_margin = 10.0 if self.terminal_lift_vector else 2.0
            commanded_speed = float(np.clip(
                target_speed + pursuit_margin, 205.0, 380.0
            ))

        # At very high closure in a genuine terminal solution, modestly lower
        # the target speed.  Never do this in a mutual/head-on geometry.
        if safe_terminal and closure > 100.0 and not self.optimized:
            commanded_speed -= min(
                5.0 if self.optimized else 10.0,
                (0.025 if self.optimized else 0.05) * (closure - 100.0),
            )

        raw_throttle = float(np.clip(
            throttle_base + 0.018 * (commanded_speed - speed), 0.12, 1.0
        ))
        if self._speed_recovery_active:
            raw_throttle = 1.0
        if speed < (195.0 if self.optimized else 180.0):
            raw_throttle = max(raw_throttle, 0.95)
        if self._throttle_cmd is None:
            self._throttle_cmd = float(action[3])
        max_step = 1.20 * dt if dt > 0.0 else 1.0
        self._throttle_cmd += float(np.clip(
            raw_throttle - self._throttle_cmd, -max_step, max_step
        ))
        action[3] = float(np.clip(self._throttle_cmd, 0.0, 1.0))
        info["throttle_owner"] = "measured_phase_speed"
        info["throttle_authority"] = 1.0

        # W118 vertical-energy loop.  Run0235 lost 3.5km because an 82-degree
        # bank was held while elevator was already saturated.  More elevator
        # cannot create vertical lift in that geometry.  Temporarily reduce
        # bank, keep a bounded pull and full thrust, then return authority only
        # after the filtered descent has stopped.  Hysteresis and a minimum
        # hold time prevent rapid owner switching.
        if self.vertical_energy_loop:
            if self._vertical_recovery_active:
                held = now - self._vertical_recovery_since
                recovered = self._vertical_speed_ema >= -4.0
                self._vertical_recovery_active = not (held >= 2.0 and recovered)
            elif (
                bank >= 68.0
                and (
                    self._vertical_speed_ema <= -28.0
                    or (altitude_loss >= 500.0 and self._vertical_speed_ema <= -12.0)
                    or own_altitude <= 1200.0
                )
            ):
                self._vertical_recovery_active = True
                self._vertical_recovery_since = now
        else:
            self._vertical_recovery_active = False

        vertical_recovery_bank = 0.0
        if self._vertical_recovery_active and target_sign != 0.0:
            # Preserve more defensive turn authority while directly tracked;
            # otherwise use enough roll-out to rebuild vertical lift.
            recovery_bank_mag = 68.0 if immediate_threat else 62.0
            if own_altitude <= 900.0:
                recovery_bank_mag = 55.0
            desired_recovery_bank = target_sign * recovery_bank_mag
            recovery_error = (
                desired_recovery_bank - current_bank + 180.0
            ) % 360.0 - 180.0
            roll_rate = self._number(info.get("measured_roll_rate"), 0.0)
            action[0] = float(np.clip(
                0.014 * recovery_error - 0.005 * roll_rate, -0.65, 0.65
            ))
            action[1] = min(float(action[1]), -0.78)
            action[3] = 1.0
            vertical_recovery_bank = recovery_bank_mag
            info["bank_owner"] = "vertical_energy_recovery"
            info["pitch_owner"] = "vertical_energy_recovery"
            info["throttle_owner"] = "vertical_energy_recovery"
            energy_bank_active = False
            bank_hold_active = False

        # W119 one-shot orbit escape.  It is deliberately downstream of the
        # vertical-energy loop so there is exactly one bank owner, and is
        # allowed only while the opponent's nose is safely displaced.
        if self._los_reacquire_active and self._los_reacquire_sign != 0.0:
            desired_reacquire_bank = self._los_reacquire_sign * 78.0
            reacquire_error = (
                desired_reacquire_bank - current_bank + 180.0
            ) % 360.0 - 180.0
            roll_rate = self._number(info.get("measured_roll_rate"), 0.0)
            action[0] = float(np.clip(
                0.016 * reacquire_error - 0.004 * roll_rate, -0.75, 0.75
            ))
            action[3] = 1.0
            info["bank_owner"] = "safe_los_reacquire"
            info["throttle_owner"] = "safe_los_reacquire"
            self._vertical_recovery_active = False
            vertical_recovery_bank = 0.0
            energy_bank_active = False
            bank_hold_active = False

        # W121 rear-quarter capture.  With a measured speed advantage, aim at
        # a moving point behind the target instead of its current position.
        # This converts excess closure into an inside cut and avoids another
        # nose-to-nose pass.  It owns bank only in a safe, non-defensive
        # geometry; W120 terminal lift-vector control remains downstream.
        rear_capture_active = False
        rear_capture_az = 0.0
        rear_capture_offset = 0.0
        speed_advantage = speed - target_speed if target_speed >= 50.0 else 0.0
        if (
            self.rear_quarter_capture
            and own is not None and target is not None
            and bool(info.get("first_merge_passed", False))
            and speed_advantage >= 5.0
            and 900.0 <= distance <= 4500.0
            and 25.0 <= ata <= 140.0 and threat_ata >= 25.0
        ):
            own_n = self._number(own[StateIndex.N])
            own_e = self._number(own[StateIndex.E])
            target_n = self._number(target[StateIndex.N])
            target_e = self._number(target[StateIndex.E])
            target_yaw = math.radians(self._number(target[StateIndex.YAW]))
            own_yaw = self._number(own[StateIndex.YAW])
            rear_capture_offset = float(np.clip(
                350.0 + 0.18 * distance + 0.45 * max(closure, 0.0),
                450.0, 900.0,
            ))
            rear_n = target_n - rear_capture_offset * math.cos(target_yaw)
            rear_e = target_e - rear_capture_offset * math.sin(target_yaw)
            rear_bearing = math.degrees(math.atan2(
                rear_e - own_e, rear_n - own_n
            ))
            rear_capture_az = (
                rear_bearing - own_yaw + 180.0
            ) % 360.0 - 180.0
            capture_sign = (
                1.0 if rear_capture_az > 0.0
                else -1.0 if rear_capture_az < 0.0 else 0.0
            )
            if capture_sign != 0.0:
                capture_bank_mag = float(np.interp(
                    abs(rear_capture_az), [0.0, 12.0, 35.0, 90.0],
                    [20.0, 38.0, 68.0, 78.0],
                ))
                desired_capture_bank = capture_sign * capture_bank_mag
                capture_error = (
                    desired_capture_bank - current_bank + 180.0
                ) % 360.0 - 180.0
                roll_rate = self._number(info.get("measured_roll_rate"), 0.0)
                action[0] = float(np.clip(
                    0.014 * capture_error - 0.005 * roll_rate, -0.70, 0.70
                ))
                action[3] = 1.0
                rear_capture_active = True
                info["bank_owner"] = "rear_quarter_capture"
                info["throttle_owner"] = "rear_quarter_capture"
                energy_bank_active = False
                bank_hold_active = False

        # W120 terminal 3-D lift-vector conversion.  Run0235 reached 26.7deg
        # ATA with only 5.9deg horizontal LOS error but 26.1deg elevation
        # error while banked 70deg and elevator-saturated.  Temporarily roll
        # toward a shallow bank so elevator produces vertical nose motion.
        terminal_lift_candidate = (
            self.terminal_lift_vector
            and bool(info.get("first_merge_passed", False))
            and ata <= 45.0 and distance <= 2500.0
            and abs(los_az) <= 20.0 and abs(self._number(info.get("los_el"), 0.0)) >= 12.0
            and now >= self._terminal_lift_cooldown_until
        )
        if terminal_lift_candidate and not self._terminal_lift_active:
            self._terminal_lift_active = True
            self._terminal_lift_since = now
        terminal_lift_age = (
            max(0.0, now - self._terminal_lift_since)
            if self._terminal_lift_active else 0.0
        )
        los_el = self._number(info.get("los_el"), 0.0)
        terminal_lift_limit = 0.6 if threat_ata < 15.0 else 1.4
        if self._terminal_lift_active and (
            abs(los_el) <= 8.0 or ata > 52.0 or distance > 2700.0
            or terminal_lift_age >= terminal_lift_limit
        ):
            self._terminal_lift_active = False
            self._terminal_lift_cooldown_until = now + 0.8

        terminal_lift_bank = 0.0
        if self._terminal_lift_active:
            retained_sign = (
                1.0 if current_bank > 0.0 else -1.0 if current_bank < 0.0
                else target_sign if target_sign != 0.0 else 1.0
            )
            terminal_lift_bank = retained_sign * float(np.interp(
                abs(los_el), [12.0, 25.0, 45.0], [38.0, 24.0, 18.0]
            ))
            lift_bank_error = (
                terminal_lift_bank - current_bank + 180.0
            ) % 360.0 - 180.0
            roll_rate = self._number(info.get("measured_roll_rate"), 0.0)
            action[0] = float(np.clip(
                0.016 * lift_bank_error - 0.005 * roll_rate, -0.70, 0.70
            ))
            action[1] = -0.95 if los_el > 0.0 else 0.65
            if closure >= 300.0:
                action[3] = min(float(action[3]), 0.55)
            info["bank_owner"] = "terminal_lift_vector"
            info["pitch_owner"] = "terminal_lift_vector"
            info["throttle_owner"] = "terminal_lift_vector"
            energy_bank_active = False
            bank_hold_active = False

        # Counterattack owns all axes only during the energy-trade and crossing
        # phases.  Unlike the rejected wings-level climb, brake_break retains
        # a hard turn while shedding speed; cross_reversal then drives a
        # rolling-scissors crossing.  W100 regains geometry in conversion.
        vertical_escape_active = self._vertical_phase in (
            "brake_break", "cross_reversal"
        )
        if vertical_escape_active:
            roll_rate = self._number(info.get("measured_roll_rate"), 0.0)
            if self._vertical_phase == "brake_break":
                desired_counter_bank = self._counter_turn_sign * 78.0
                if self.confirmed_overshoot_only:
                    # Preserve the existing defensive pull; only bound the
                    # energy trade.  W115's fixed -0.82 drove pitch saturation.
                    action[1] = max(float(action[1]), -0.75)
                    action[3] = 0.35
                else:
                    action[1] = -0.82
                    action[3] = 0.20
            else:
                desired_counter_bank = -self._counter_turn_sign * 78.0
                action[1] = -0.65
                action[3] = 0.25
            counter_bank_error = (
                desired_counter_bank - current_bank + 180.0
            ) % 360.0 - 180.0
            counter_roll_cmd = float(np.clip(
                0.014 * counter_bank_error - 0.005 * roll_rate,
                -0.75, 0.75,
            ))
            if self._vertical_phase == "cross_reversal":
                # Preserve enough bang-bang authority to cross the old lift
                # plane; otherwise the rate damper stalls near wings-level.
                reverse_sign = -self._counter_turn_sign
                if abs(current_bank) >= 25.0 and current_bank * reverse_sign < 0.0:
                    counter_roll_cmd = reverse_sign * 0.65
            action[0] = counter_roll_cmd
            info["bank_owner"] = "counterattack_scissors"
            info["pitch_owner"] = "counterattack_scissors"
            info["throttle_owner"] = "counterattack_scissors"
            energy_bank_active = False
            bank_hold_active = False
        elif self._vertical_phase == "attack_conversion":
            # Restore energy immediately and let W100's proven attack geometry
            # exploit the crossing instead of imposing another fixed bank.
            action[3] = 1.0
            info["throttle_owner"] = "counterattack_conversion"

        action = np.clip(action, [-0.8, -1.0, -1.0, 0.0],
                         [0.8, 1.0, 1.0, 1.0]).astype(np.float32)
        diagnostics = diagnose_control_ownership(info, action)
        # Preserve the explicit exclusive owners after generic diagnostics.
        diagnostics.update({
            "bank_owner": info["bank_owner"],
            "pitch_owner": info["pitch_owner"],
            "throttle_owner": info["throttle_owner"],
            "bank_proposal_count": 1,
            "pitch_proposal_count": 1,
            "throttle_proposal_count": 1,
            "proposal_conflict": False,
        })
        info.update(diagnostics)
        info.update({
            "measured_envelope_active": True,
            "measured_target_speed_mps": commanded_speed,
            "measured_pursuit_speed_margin_mps": (
                10.0 if self.terminal_lift_vector and not sustained_turn_cut
                else 2.0 if not sustained_turn_cut else 0.0
            ),
            "measured_speed_error_mps": commanded_speed - speed,
            "measured_enemy_speed_mps": target_speed,
            "measured_sustained_turn_cut": sustained_turn_cut,
            "measured_vertical_speed_ema_mps": self._vertical_speed_ema,
            "measured_altitude_loss_m": altitude_loss,
            "measured_vertical_recovery_active": self._vertical_recovery_active,
            "measured_vertical_recovery_bank_deg": vertical_recovery_bank,
            "measured_los_aim_delta_deg": wrapped_aim_los_delta,
            "measured_los_disagree_age_s": los_disagree_age,
            "measured_los_reacquire_active": self._los_reacquire_active,
            "measured_los_reacquire_sign": self._los_reacquire_sign,
            "measured_los_reacquire_used": self._los_reacquire_used,
            "measured_terminal_lift_active": self._terminal_lift_active,
            "measured_terminal_lift_age_s": terminal_lift_age,
            "measured_terminal_lift_bank_deg": terminal_lift_bank,
            "measured_rear_capture_active": rear_capture_active,
            "measured_rear_capture_az_deg": rear_capture_az,
            "measured_rear_capture_offset_m": rear_capture_offset,
            "measured_speed_advantage_mps": speed_advantage,
            "measured_speed_recovery_active": self._speed_recovery_active,
            "measured_immediate_threat": immediate_threat,
            "measured_energy_bank_active": energy_bank_active,
            "measured_energy_bank_target_deg": energy_bank_target,
            "measured_rear_trap": rear_trap,
            "measured_rear_trap_age_s": rear_trap_age,
            "measured_vertical_phase": self._vertical_phase,
            "measured_vertical_phase_age_s": phase_age,
            "measured_vertical_pitch_cmd": self._vertical_pitch_cmd,
            "measured_vertical_escape_active": vertical_escape_active,
            "measured_counter_turn_sign": self._counter_turn_sign,
            "measured_counter_min_distance_m": self._counter_min_distance,
            "measured_counter_overshoot_detected": self._counter_overshoot_detected,
            "measured_counter_used": self._counter_used,
            "measured_opening_fill_active": opening_fill_active,
            "measured_opening_fill_cmd": opening_fill_cmd,
            "measured_energy_unload_active": energy_unload_active,
            "measured_pitch_floor": pitch_floor,
            "measured_safe_terminal": safe_terminal,
            "measured_raw_throttle": raw_throttle,
            "measured_opening_roll_active": opening_roll_active,
            "measured_bank_hold_active": bank_hold_active,
            "measured_roll_schedule_cmd": roll_schedule_cmd,
        })
        return ActionResult(
            action=action,
            source=f"{self.name}<{result.source}>",
            confidence=result.confidence,
            info=info,
        )

    def close(self) -> None:
        self.base.close()
