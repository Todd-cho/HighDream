"""RL-first terminal stabilizer for live Unreal inference.

The inner policy keeps full authority during acquisition.  Once it has already
created a close firing opportunity, this layer only removes axis saturation and
command chatter; it does not replace the RL aim point with another pursuit law.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


@dataclass
class TerminalStabilizerConfig:
    enter_ata_deg: float = 22.0
    exit_ata_deg: float = 35.0
    enter_distance_m: float = 2200.0
    exit_distance_m: float = 3000.0
    # Retain enough authority to follow a turning target, but prevent the
    # +-1 discontinuities seen in run0218 (65/83/80% axis saturation).
    roll_limit: float = 0.78
    pitch_limit: float = 0.68
    yaw_limit: float = 0.42
    smoothing_alpha: float = 0.38
    max_delta_roll: float = 0.30
    max_delta_pitch: float = 0.24
    max_delta_yaw: float = 0.20
    terminal_throttle_floor: float = 0.72
    precision_blend: float = 0.72
    precision_full_ata_deg: float = 8.0
    az_norm_deg: float = 7.0
    el_norm_deg: float = 6.0
    rate_norm_degps: float = 24.0
    az_rate_gain: float = 0.20
    el_rate_gain: float = 0.16
    los_rate_alpha: float = 0.32
    # 2026-08-25 (user log analysis: "가까이 붙긴 해도 맞추지를 못해" -- live run
    # showed ata bottom out at 5.08deg for exactly one frame then immediately
    # diverge again, while roll_rate kept ACCELERATING through that moment
    # (-12.7->-119.2 deg/s, never decelerating) -- the aircraft rolled clean
    # through the alignment instead of stopping there. Root cause: roll_cmd
    # here is driven directly off az (a doubly-integrated quantity relative
    # to roll_cmd: roll_cmd -> roll rate -> bank angle -> turn rate -> az),
    # with no feedback on the aircraft's OWN roll rate -- so residual roll
    # momentum from the approach carries straight through the zero-crossing.
    # Adds a proper rate-damping term on the ownship's OWN measured roll
    # rate (not az_rate, which is the LOS's rate, not the airframe's) so the
    # loop actively arrests roll motion as az converges instead of just
    # continuing to command roll off position error alone.
    own_roll_rate_damping_gain: float = 0.35
    own_roll_rate_norm_degps: float = 60.0

    # 2026-08-25 (user live report + log analysis): a live run at the widened
    # entry gate lost 100:70 and the user visually observed "couldn't commit
    # to a turn direction while fighting" -- confirmed in the log: roll_cmd
    # flipped sign 150 times over 200s (median 0.52s apart, 81% under 1s).
    # This is the SAME failure tactical_wrapper.py's own roll-commitment fix
    # targeted (there it took JSBSim win 60%->96.7%), but that fix works by
    # computing its OWN target bank from LOS az and driving roll_cmd toward
    # it -- i.e. the rule flies the roll axis outright, which is exactly the
    # "rule-dominant acquisition" pattern this file's own docstring says
    # underperforms raw RL (every such attempt this session scored far below
    # raw v8 in JSBSim). This is a much narrower port of the same idea: it
    # does NOT compute a target bank or drive roll_cmd itself. It only
    # freezes the PREVIOUS roll_cmd when the RL's raw output tries to flip
    # sign but LOS az hasn't itself crossed deadband+hysteresis -- i.e. it
    # blocks reversals the geometry doesn't support, but never overrides a
    # reversal az itself confirms, and never touches magnitude/pitch/yaw/
    # throttle. Off by default (opt-in) since it has not been live-verified
    # yet -- validate in JSBSim first per this project's own rule 15.
    roll_commit_enabled: bool = False
    roll_commit_deadband_deg: float = 3.0
    roll_commit_hysteresis_deg: float = 10.0
    # 2026-08-25 (live log diagnosis, user report of "wings rocking level->
    # vertical->level, turn feels slow"): desired_sign = 1 if az>=0 else -1
    # is unreliable near the +-180deg wrap -- az was measured oscillating
    # between +179deg and -179deg (a true ~2deg wobble) while the sign flip
    # made the commit logic treat every crossing as a confirmed hard
    # reversal (abs(az)~179 >> deadband+hysteresis=13, so it always passed
    # the "confirmed" gate), flipping roll_cmd +1.0<->-1.0 every ~0.5-1s the
    # entire time ata was actually converging smoothly 150->90deg. Same bug
    # shape exists in pursuit_controller.py's _turn_sign (identical
    # `1 if az >= 0.0 else -1"` pattern) -- not fixed there yet. Fix: skip
    # re-evaluating the commit sign entirely while az is within this many
    # degrees of the +-180 boundary (sign is inherently ambiguous there);
    # only the one-time initial assignment is exempt (need *some* starting
    # sign).
    roll_commit_wrap_guard_deg: float = 15.0

    # 2026-08-25 (user live observation, confirmed in log): opponent averaged
    # 291.7 m/s vs our 202.1 m/s over a full live run, above the 220 m/s
    # corner speed (w104/w105's live-measured value) 91% of the time vs our
    # 34% -- i.e. we spent most of the fight energy-starved while pulling at
    # 76-86% axis saturation, which likely costs sustained turn rate/ATA-hold
    # even when the instantaneous pull is aggressive. This does NOT touch
    # roll/pitch/yaw (RL's own aim/direction decisions stay untouched) -- it
    # only raises the RAW throttle command during acquisition when own speed
    # is under corner_speed_mps, exactly like a real pilot adding power in a
    # turn fight. Off by default -- validate in JSBSim first per rule 15.
    energy_throttle_floor_enabled: bool = False
    corner_speed_mps: float = 220.0
    energy_throttle_floor: float = 0.95

    # 2026-08-25 (throttle-floor-only fix insufficient): FlyONSPEED's BFM
    # energy-management doctrine -- "if you are slower than ONSPEED and at
    # full throttle, the only way to accelerate is to unload (reduce pitch/
    # G/AOA) and trade altitude for airspeed" (source: flyonspeed.org/
    # basic-energy-management). Confirmed relevant: the throttle floor alone
    # fired 3411/7929 frames in a live run but the below-corner-speed
    # fraction still ROSE (22.2%->46.9%) because pitch saturation stayed at
    # 83.4% the whole time -- max throttle can't out-accelerate max-AoA drag.
    # This adds a second, independent lever: once speed is meaningfully
    # under corner speed (not just at it), cap |pitch_cmd| so the aircraft
    # actually unloads instead of continuing to pull at whatever magnitude
    # RL commands. Direction (sign of pitch_cmd, and all of roll/yaw) is
    # untouched -- only magnitude is capped, and only far below corner speed.
    # 2026-08-25 v2: live run showed energy_pitch_unload_cap=0.5 was far too
    # permissive -- at the moment unload activated, altitude was still
    # climbing +34m in 0.23s (~150 m/s climb rate) and speed barely
    # decayed (179.9->179.2 m/s), i.e. still deep in the "unbounded zoom
    # climb" pattern this project first diagnosed in tactical_wrapper.py
    # 2026-08-21 ("own_pitch_deg climbed monotonically... nothing in the
    # pitch law says this isn't converging, stop pulling"). Cap dropped
    # much closer to 0 (barely any pull allowed) so unload actually stops
    # the climb instead of merely halving it.
    energy_pitch_unload_enabled: bool = False
    energy_pitch_unload_deficit_mps: float = 40.0
    energy_pitch_unload_cap: float = 0.15

    # 2026-08-25 v3: the hard on/off cap above (v1: 0.5, v2: 0.15) live-tested
    # worse both times than the plain roll_commit+energy_throttle_floor
    # baseline with NO pitch limiting at all (that baseline scored this
    # project's best live result, 100:99 near-win). Web research (F-16
    # sustained-turn-performance discussion, hushkit.net manoeuvre-
    # performance primer) says real pilots don't hard-clip either: they
    # throttle/unload just enough to sit near best-turn-rate speed and
    # target a high AVERAGE turn rate over time, not an all-or-nothing
    # switch. This replaces the step-function cap with a continuous scale:
    # pitch pull tapers smoothly from full authority (at/above corner
    # speed) down to energy_pitch_scale_floor (at corner_speed -
    # energy_pitch_scale_deficit_mps or below), instead of snapping
    # straight to one fixed small value the moment the deficit threshold is
    # crossed. Independent of, and meant to be tried INSTEAD of (not
    # together with), energy_pitch_unload_enabled -- both touch the same
    # signal.
    energy_pitch_scale_enabled: bool = False
    energy_pitch_scale_deficit_mps: float = 60.0
    energy_pitch_scale_floor: float = 0.4

    # 2026-08-25 (user report: shot down, 100:90->loss; log showed enemy held
    # threat_ata<=6deg for ~53.6s of a 183s live flight with zero reaction).
    # Nothing in this codebase (W-series, tactical_wrapper, pursuit_
    # controller, this file) has ever implemented defensive BFM -- every
    # rule built this session is offense-only (point OUR nose at THEM).
    # Web research (defensive BFM doctrine, e.g. fa714ladder.tripod.com/
    # bfm3.html via search) confirmed the standard break-turn technique:
    # "put your lift vector directly on the bandit and pull your tightest
    # corner-velocity max-G turn towards him" -- turning TOWARD the
    # attacker (not away) denies him turning room and forces an overshoot,
    # because his turn radius/rate requirement to stay on your tail grows
    # faster than yours when you turn into him. This is a hard override:
    # when active it takes priority over acquisition/terminal/roll-commit/
    # energy-management entirely (survival beats everything else), using
    # the same atan2(az,el) lift-vector-pointing law as lead_pull_handoff.py
    # but aimed at the THREAT's LOS (az/el from us to them) at max authority,
    # no energy conservation (bleeding energy is acceptable to survive).
    defensive_break_enabled: bool = False
    defensive_break_threat_ata_deg: float = 30.0
    defensive_break_range_m: float = 2000.0
    defensive_break_roll_authority: float = 1.0
    defensive_break_pitch_authority: float = 0.9
    defensive_break_bank_error_norm_deg: float = 20.0
    # 2026-08-25 v2 (first version regressed live -- shot down, user report:
    # "break가 안 된 것 같다"): the atan2(az,el)-every-frame law re-picked a
    # target bank continuously, so as az swept through 0 during a near-
    # head-on pass (this engagement was nose-on, not a classic 6-o'clock
    # chase) roll_cmd re-servoed continuously (-1.0 -> 0 -> +0.77 inside
    # 0.6s) instead of holding one committed hard turn, while threat_ata
    # kept improving FOR the enemy the whole time. Web research (proportional
    # navigation guidance + hawk pursuit study, PMC6560099) says the fix
    # doctrine actually wants: LOW gain, rate-informed control, not raw-
    # angle-reactive high gain -- hawks blend low-gain PN (LOS RATE, already
    # computed here as _az_rate) with low-gain pure pursuit specifically
    # because it resists being thrown off by erratic/fast-changing target
    # motion. Re-implemented as: pick a break DIRECTION once (sign of az,
    # same wrap-safe commit as roll_commit_enabled) and hold it for the
    # whole activation instead of re-solving every frame, then EMA-smooth
    # the roll command onto it -- the same "commit + EMA" pattern already
    # validated in this codebase (tactical_wrapper.py 2026-08-21, win
    # 60%->96.7% from exactly this fix).
    defensive_break_ema_alpha: float = 0.25


class TerminalStabilizerActionProvider(ActionProvider):
    def __init__(self, inner: ActionProvider, config: TerminalStabilizerConfig | None = None):
        self.inner = inner
        self.cfg = config or TerminalStabilizerConfig()
        self.geometry = GeometryInfo()
        self._active = False
        self._previous: np.ndarray | None = None
        self._previous_time: float | None = None
        self._previous_az: float | None = None
        self._previous_el: float | None = None
        self._previous_own_roll: float | None = None
        self._az_rate = 0.0
        self._el_rate = 0.0
        self._own_roll_rate = 0.0
        self._roll_commit_sign = 0
        self._break_roll_cmd = 0.0
        self._break_engaged = False
        self._break_sign = 0
        self._break_target_bank = 0.0

    def reset(self, context: ActionContext | None = None) -> None:
        self.inner.reset(context)
        self._active = False
        self._previous = None
        self._previous_time = None
        self._previous_az = None
        self._previous_el = None
        self._previous_own_roll = None
        self._az_rate = 0.0
        self._el_rate = 0.0
        self._own_roll_rate = 0.0
        self._roll_commit_sign = 0
        self._break_roll_cmd = 0.0
        self._break_engaged = False
        self._break_sign = 0
        self._break_target_bank = 0.0

    def close(self) -> None:
        self.inner.close()

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.inner.compute_action(context)
        raw = clip_action(result.action)
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            self._previous = raw.copy()
            return result

        distance = float(self.geometry._get_distance(own, target))
        ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
        az, el = self.geometry._get_los_angle(own, target)
        az, el = float(az), float(el)
        # 2026-08-25 (user request, after a live loss (100:24) that this log
        # schema couldn't explain without manually recomputing from raw
        # position/attitude columns): the enemy's angle-off to US -- i.e. how
        # good THEIR firing solution on us is -- logged under the same
        # "threat_ata" info key the W-controller uses, so policies.py's CSV
        # logger picks it up automatically (see its `threat_ata` column).
        threat_ata = abs(float(self.geometry._get_antenna_train_angle(target, own, False)))

        cfg = self.cfg
        if (
            cfg.defensive_break_enabled
            and threat_ata <= cfg.defensive_break_threat_ata_deg
            and distance <= cfg.defensive_break_range_m
        ):
            # Turn TOWARD the threat's LOS (az/el from us to them) -- see the
            # module docstring/config comment for why toward, not away, and
            # for why the target bank is now frozen at engagement instead of
            # re-solved every frame (v1 regressed live by continuously
            # re-servoing roll as az swept through a near-head-on pass).
            if not self._break_engaged:
                self._break_engaged = True
                self._break_target_bank = float(np.degrees(np.arctan2(az, el)))
            current_bank = _wrap180(float(own[StateIndex.ROLL]))
            bank_error = _wrap180(self._break_target_bank - current_bank)
            roll_raw = float(np.clip(
                bank_error / cfg.defensive_break_bank_error_norm_deg, -1.0, 1.0
            )) * cfg.defensive_break_roll_authority
            alpha = cfg.defensive_break_ema_alpha
            self._break_roll_cmd += alpha * (roll_raw - self._break_roll_cmd)
            # 2026-08-25 v3 (v2 regressed live: forced pitch=-0.9 unconditionally
            # "survival beats energy conservation" -- but a live run showed that
            # reasoning was backwards. Speed bled to 101 m/s while STILL
            # climbing (7762m->7780m) under that forced pull -- a near-stall,
            # not survival. Bleeding all your energy into a mushing climb does
            # not deny the bandit turning room; it just makes you slower and
            # easier to out-turn. Reuses the SAME energy law as
            # energy_pitch_unload instead of ignoring it: full break pull only
            # when there's speed to sustain it, otherwise capped exactly like
            # normal acquisition would cap it.
            own_speed = float(own[StateIndex.KCAS])
            if own_speed < cfg.corner_speed_mps - cfg.energy_pitch_unload_deficit_mps:
                break_pitch_cmd = -cfg.energy_pitch_unload_cap
            else:
                break_pitch_cmd = -cfg.defensive_break_pitch_authority
            action = np.array(
                [self._break_roll_cmd, break_pitch_cmd, 0.0, 1.0],
                dtype=np.float32,
            )
            action = clip_action(action)
            self._previous = action.copy()
            return ActionResult(
                action=action,
                source="terminal_stabilizer[defensive_break]",
                confidence=result.confidence,
                info={
                    **result.info,
                    "state": "defensive_break",
                    "ata": ata,
                    "distance": distance,
                    "threat_ata": threat_ata,
                    "los_az": az,
                    "los_el": el,
                    "break_target_bank": self._break_target_bank,
                    "own_speed": own_speed,
                },
            )
        self._break_engaged = False

        now = float(own[StateIndex.SIM_TIME])
        own_roll_now = _wrap180(float(own[StateIndex.ROLL]))
        if self._previous_time is not None and self._previous_az is not None:
            dt = now - self._previous_time
            if dt > 1e-3:
                az_delta = ((az - self._previous_az + 180.0) % 360.0) - 180.0
                raw_az_rate = az_delta / dt
                raw_el_rate = (el - self._previous_el) / dt
                alpha_rate = self.cfg.los_rate_alpha
                self._az_rate = alpha_rate * raw_az_rate + (1.0 - alpha_rate) * self._az_rate
                self._el_rate = alpha_rate * raw_el_rate + (1.0 - alpha_rate) * self._el_rate
                if self._previous_own_roll is not None:
                    roll_delta = _wrap180(own_roll_now - self._previous_own_roll)
                    raw_own_roll_rate = roll_delta / dt
                    self._own_roll_rate = (
                        alpha_rate * raw_own_roll_rate + (1.0 - alpha_rate) * self._own_roll_rate
                    )
        self._previous_time = now
        self._previous_az = az
        self._previous_el = el
        self._previous_own_roll = own_roll_now
        if self._active:
            self._active = ata < self.cfg.exit_ata_deg and distance < self.cfg.exit_distance_m
        else:
            self._active = ata <= self.cfg.enter_ata_deg and distance <= self.cfg.enter_distance_m

        if not self._active:
            # Do not smooth acquisition: run0218 proved raw v8 can reach the
            # firing cone while every rule-dominant acquisition candidate did not.
            out = raw
            roll_commit_active = False
            if self.cfg.roll_commit_enabled:
                cfg = self.cfg
                desired_sign = 1 if az >= 0.0 else -1
                wrap_ambiguous = abs(az) >= (180.0 - cfg.roll_commit_wrap_guard_deg)
                if self._roll_commit_sign == 0:
                    self._roll_commit_sign = desired_sign
                elif (
                    not wrap_ambiguous
                    and desired_sign != self._roll_commit_sign
                    and abs(az) >= cfg.roll_commit_deadband_deg + cfg.roll_commit_hysteresis_deg
                ):
                    self._roll_commit_sign = desired_sign
                raw_roll_sign = 1 if raw[0] >= 0.0 else -1
                if (
                    abs(az) >= cfg.roll_commit_deadband_deg
                    and raw_roll_sign != self._roll_commit_sign
                    and self._previous is not None
                ):
                    out = raw.copy()
                    out[0] = self._previous[0]
                    roll_commit_active = True
            own_speed = float(own[StateIndex.KCAS])
            energy_floor_active = False
            if self.cfg.energy_throttle_floor_enabled and own_speed < self.cfg.corner_speed_mps:
                if out is raw:
                    out = raw.copy()
                out[3] = max(float(out[3]), self.cfg.energy_throttle_floor)
                energy_floor_active = True
            energy_unload_active = False
            if (
                self.cfg.energy_pitch_unload_enabled
                and own_speed < self.cfg.corner_speed_mps - self.cfg.energy_pitch_unload_deficit_mps
                and out[1] < -self.cfg.energy_pitch_unload_cap
            ):
                # Negative pitch_cmd is nose-up/pull in this protocol -- only
                # the PULL direction bleeds energy via induced drag, so only
                # clamp toward less-negative (unload), never touch a
                # nose-down/push command (that already regains speed).
                if out is raw:
                    out = raw.copy()
                out[1] = -self.cfg.energy_pitch_unload_cap
                energy_unload_active = True
            energy_pitch_scale = 1.0
            if self.cfg.energy_pitch_scale_enabled and out[1] < 0.0:
                deficit = self.cfg.corner_speed_mps - own_speed
                energy_pitch_scale = 1.0 - float(np.clip(
                    deficit / max(1e-3, self.cfg.energy_pitch_scale_deficit_mps), 0.0, 1.0
                )) * (1.0 - self.cfg.energy_pitch_scale_floor)
                if energy_pitch_scale < 1.0:
                    if out is raw:
                        out = raw.copy()
                    out[1] = out[1] * energy_pitch_scale
            self._previous = out.copy()
            return ActionResult(
                action=out,
                source="terminal_stabilizer[rl_acquisition]",
                confidence=result.confidence,
                info={
                    **result.info,
                    "terminal_stabilizer_active": False,
                    "state": "rl_acquisition",
                    "own_speed": own_speed,
                    "energy_floor_active": energy_floor_active,
                    "energy_unload_active": energy_unload_active,
                    "energy_pitch_scale": energy_pitch_scale,
                    "ata": ata,
                    "distance": distance,
                    "los_az": az,
                    "los_el": el,
                    "threat_ata": threat_ata,
                    "roll_commit_active": roll_commit_active,
                    "roll_commit_sign": self._roll_commit_sign,
                },
            )

        # Actual LOS PD loop.  Unlike the old tactical wrapper this does not
        # invent a lead/lag aim point or select a manoeuvre.  It only drives
        # the measured body-frame LOS error and its rate toward zero after RL
        # has acquired the target.
        precision = raw.copy()
        precision[0] = float(np.clip(
            az / self.cfg.az_norm_deg
            + self.cfg.az_rate_gain * self._az_rate / self.cfg.rate_norm_degps
            - self.cfg.own_roll_rate_damping_gain * self._own_roll_rate / self.cfg.own_roll_rate_norm_degps,
            -self.cfg.roll_limit, self.cfg.roll_limit,
        ))
        # Live convention: negative pitch command pulls the nose up.
        precision[1] = float(np.clip(
            -el / self.cfg.el_norm_deg
            - self.cfg.el_rate_gain * self._el_rate / self.cfg.rate_norm_degps,
            -self.cfg.pitch_limit, self.cfg.pitch_limit,
        ))
        precision[2] = float(np.clip(
            0.55 * az / self.cfg.az_norm_deg
            + 0.10 * self._az_rate / self.cfg.rate_norm_degps,
            -self.cfg.yaw_limit, self.cfg.yaw_limit,
        ))
        # Gradually give the precision loop authority, reaching the configured
        # blend by 8deg.  This avoids a discontinuity at the 22deg entry gate.
        convergence = float(np.clip(
            (self.cfg.enter_ata_deg - ata)
            / max(1e-3, self.cfg.enter_ata_deg - self.cfg.precision_full_ata_deg),
            0.0, 1.0,
        ))
        precision_weight = self.cfg.precision_blend * convergence
        limited = (1.0 - precision_weight) * raw + precision_weight * precision
        limits = np.asarray(
            [self.cfg.roll_limit, self.cfg.pitch_limit, self.cfg.yaw_limit], dtype=np.float32
        )
        limited[:3] = np.clip(limited[:3], -limits, limits)
        limited[3] = max(float(limited[3]), self.cfg.terminal_throttle_floor)

        if self._previous is not None:
            alpha = self.cfg.smoothing_alpha
            filtered = self._previous + alpha * (limited - self._previous)
            max_delta = np.asarray(
                [self.cfg.max_delta_roll, self.cfg.max_delta_pitch, self.cfg.max_delta_yaw, 0.25],
                dtype=np.float32,
            )
            limited = self._previous + np.clip(filtered - self._previous, -max_delta, max_delta)
        limited = clip_action(limited)
        self._previous = limited.copy()
        return ActionResult(
            action=limited,
            source="terminal_stabilizer[terminal]",
            confidence=result.confidence,
            info={
                **result.info,
                "terminal_stabilizer_active": True,
                "state": "terminal",
                "terminal_precision_weight": precision_weight,
                "los_az": az,
                "los_el": el,
                "terminal_los_az_deg": az,
                "terminal_los_el_deg": el,
                "terminal_los_az_rate_degps": self._az_rate,
                "terminal_los_el_rate_degps": self._el_rate,
                "ata": ata,
                "distance": distance,
                "threat_ata": threat_ata,
            },
        )


__all__ = ["TerminalStabilizerActionProvider", "TerminalStabilizerConfig"]
