"""Integrated 1v1 BFM acquisition controller for live Unreal telemetry.

Unlike the W1 family, this controller does not collapse every large LOS angle
to a fixed bank.  It estimates engagement rates, classifies tactical state,
computes a bounded horizontal intercept point, commands a desired course rate,
and converts that rate into speed-aware bank/pull/throttle commands.

This is an acquisition/track controller, not a learned weapons policy.  A
learned fine-track provider can be handed control inside weapons_track later.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.vpp_guidance import compute_vpp_guidance
from dogfight.sim.state_schema import StateIndex


G = 9.80665


def wrap180(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def signed_unit(value: float, fallback: int = 1) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return fallback


@dataclass
class IntegratedBFMConfig:
    controller_name: str = "ibfm1"

    # Tactical gates.
    merge_initial_s: float = 2.0
    break_hold_s: float = 12.0
    defensive_range_m: float = 2200.0
    defensive_threat_ata_deg: float = 35.0
    defensive_closure_mps: float = 20.0
    defensive_min_hold_s: float = 0.0
    track_ata_deg: float = 35.0
    weapons_ata_deg: float = 10.0
    weapons_range_m: float = 1400.0

    # Relative-advantage manager.  Fixed threat cones classify every mutual
    # pass as defensive, even when ownship has the better nose position.  The
    # score compares both gun geometries, energy, and time-to-collision so an
    # actual attack window may be held without ignoring an imminent threat.
    advantage_manager_enabled: bool = False
    advantage_attack_ata_deg: float = 30.0
    advantage_attack_range_m: float = 1800.0
    advantage_attack_score: float = 0.20
    advantage_angle_scale_deg: float = 30.0
    advantage_energy_scale_mps: float = 80.0
    advantage_energy_weight: float = 0.20
    advantage_collision_ttc_s: float = 2.2
    advantage_collision_width_s: float = 1.2
    advantage_collision_weight: float = 0.60
    advantage_attack_hold_s: float = 0.8
    advantage_hard_threat_ata_deg: float = 8.0
    advantage_min_attack_ttc_s: float = 0.0

    # Estimator smoothing.
    rate_alpha: float = 0.25
    target_velocity_alpha: float = 0.20
    target_velocity_clip_mps: float = 450.0

    # Horizontal intercept horizons by manager state.
    horizon_merge_s: float = 0.3
    horizon_break_s: float = 0.8
    horizon_defensive_s: float = 0.0
    horizon_reacquire_max_s: float = 6.0
    horizon_track_max_s: float = 1.0
    horizon_weapons_s: float = 0.15

    # Guidance: desired course rate = az gain + LOS-rate feed-forward.
    az_to_turn_rate_gain: float = 0.060
    los_rate_gain: float = 0.35
    turn_rate_merge_degps: float = 4.0
    turn_rate_break_degps: float = 6.0
    turn_rate_reacquire_degps: float = 5.5
    turn_rate_track_degps: float = 3.5
    turn_rate_weapons_degps: float = 2.0
    turn_rate_defensive_degps: float = 6.0
    acquisition_min_turn_rate_degps: float = 0.0
    acquisition_min_turn_rate_ata_deg: float = 0.0
    max_bank_deg: float = 70.0
    break_bank_deg: float = 67.0
    sign_min_hold_s: float = 3.0
    # Values below 180 enable a rear-hemisphere commitment zone.  W15 uses
    # 120deg so +180/-180 representation wraps cannot reverse guidance; the
    # predicted aimpoint must first return to a physically unambiguous side.
    guidance_rear_commit_deg: float = 180.0
    use_horizontal_course_guidance: bool = False
    use_constant_turn_prediction: bool = False
    adaptive_turn_prediction: bool = False
    turn_prediction_max_arc_deg: float = 180.0
    turn_prediction_unstable_horizon_s: float = 6.0
    turn_prediction_min_stable_s: float = 0.0
    turn_prediction_yaw_accel_limit_degps2: float = 1.0e9
    lag_pursuit_ata_deg: float = 0.0
    lag_pursuit_range_m: float = 0.0
    lag_pursuit_closure_mps: float = 1.0e9
    lag_pursuit_offset_min_m: float = 0.0
    lag_pursuit_offset_max_m: float = 0.0
    lag_pursuit_offset_gain_s: float = 0.0
    lag_pursuit_terminal_taper_start_deg: float = 0.0
    lag_pursuit_terminal_taper_end_deg: float = 0.0
    lag_pursuit_mutual_lateral_offset_m: float = 0.0
    lag_pursuit_mutual_threat_ata_deg: float = 0.0
    # A low own ATA is not an attack when aspect is near 180 degrees and the
    # bandit also has a low ATA: that is a mutual nose-on pass.  Move the
    # commanded point off the bandit's nose line and split vertically instead
    # of presenting a stable head-on gun solution.  Disabled by default so
    # historical W profiles remain byte-for-byte behaviour compatible.
    headon_deconflict_enabled: bool = False
    headon_deconflict_max_ata_deg: float = 20.0
    headon_deconflict_max_threat_ata_deg: float = 20.0
    headon_deconflict_min_aspect_deg: float = 140.0
    headon_deconflict_range_m: float = 2200.0
    headon_deconflict_min_closure_mps: float = 80.0
    headon_deconflict_lateral_offset_m: float = 1000.0
    headon_deconflict_gamma_deg: float = 10.0
    # Coupled 3-D LOS/lift-vector outer loop.  This computes one normal
    # acceleration vector from the newest relative position and velocity,
    # then derives bank and flight-path targets from that same vector.  It is
    # an opt-in overlay so no historical controller changes behaviour.
    lift_vector_guidance_enabled: bool = False
    lift_vector_min_range_m: float = 150.0
    lift_vector_max_range_m: float = 3000.0
    lift_vector_max_ata_deg: float = 180.0
    lift_vector_los_kp_g: float = 3.0
    lift_vector_los_rate_gain: float = 1.2
    lift_vector_max_accel_mps2: float = 35.0
    lift_vector_bank_limit_deg: float = 72.0
    lift_vector_gamma_limit_deg: float = 20.0
    lift_vector_gamma_horizon_s: float = 0.7
    lift_vector_blend: float = 0.75
    lift_vector_defensive_blend: float = 1.0
    lift_vector_activation_delay_s: float = 0.0
    lift_vector_bank_tau_s: float = 0.0
    lift_vector_gamma_tau_s: float = 0.0
    lift_vector_bank_slew_degps: float = 1.0e9
    lift_vector_sign_hold_s: float = 0.0
    lift_vector_sign_min_bank_deg: float = 10.0
    # Preserve lag while radial closure is still high.  ATA-only tapering can
    # collapse the offset exactly when geometry needs a longer flight path to
    # avoid an overshoot.  Optional throttle support preserves airspeed while
    # lag pursuit reduces displacement/closure geometrically.
    lag_pursuit_taper_max_closure_mps: float = 1.0e9
    lag_pursuit_energy_target_speed_mps: float = 0.0
    lag_pursuit_energy_throttle_base: float = 0.55
    lag_pursuit_energy_throttle_gain: float = 0.015
    # Probability-like VPP blending.  Lag is an offensive control-zone tool,
    # not a defensive escape.  Scale it continuously from relative angular
    # advantage and low-pass the result to avoid lag/pure command jumps.
    lag_pursuit_disable_defensive: bool = False
    lag_pursuit_advantage_full_deg: float = 0.0
    lag_pursuit_scale_tau_s: float = 0.0
    vertical_alignment_elevation_deg: float = 0.0
    vertical_alignment_range_m: float = 0.0
    vertical_alignment_closure_mps: float = 1.0e9
    vertical_alignment_lag_offset_m: float = 0.0
    vertical_alignment_lag_gain_m_per_deg: float = 0.0
    vertical_alignment_lag_max_m: float = 0.0
    vertical_alignment_target_closure_mps: float = 0.0

    # Stable gun-track mode: after acquisition, match the bandit's measured
    # turn rate instead of repeatedly cutting across its circle.  Hysteresis
    # prevents rapid switching around the entry ATA.
    turn_match_enter_ata_deg: float = 0.0
    turn_match_exit_ata_deg: float = 0.0
    turn_match_enter_range_m: float = 0.0
    turn_match_exit_range_m: float = 0.0
    turn_match_min_threat_ata_deg: float = 0.0
    turn_match_enter_course_error_deg: float = 180.0
    turn_match_exit_course_error_deg: float = 180.0
    turn_match_enter_max_closure_mps: float = 1.0e9
    turn_match_entry_hold_s: float = 0.0
    turn_match_require_stable_prediction: bool = False
    turn_match_yaw_rate_gain: float = 1.0
    turn_match_course_gain: float = 0.0
    turn_match_los_rate_gain: float = 0.0
    turn_match_rate_limit_degps: float = 14.0
    turn_match_sign_guard_error_deg: float = 0.0
    turn_match_sign_guard_rate_degps: float = 3.0
    terminal_track_enter_ata_deg: float = 0.0
    terminal_track_exit_ata_deg: float = 0.0
    terminal_track_enter_range_m: float = 0.0
    terminal_track_exit_range_m: float = 0.0
    terminal_track_prelock_ata_deg: float = 0.0
    terminal_track_prelock_range_m: float = 0.0
    terminal_track_min_threat_ata_deg: float = 0.0
    terminal_track_max_aspect_deg: float = 180.0
    terminal_track_yaw_rate_gain: float = 0.7
    terminal_track_course_gain: float = 1.2
    terminal_track_los_rate_gain: float = 0.4
    terminal_track_rate_limit_degps: float = 14.0
    # In the gun cone, point the nose at the target's current world-frame
    # elevation.  Long-horizon vertical extrapolation is useful for pursuit,
    # but a diving target can otherwise command nose-down while it is still
    # visibly above the nose and spoil the final few degrees of ATA.
    terminal_pitch_attitude_kp: float = 0.0
    terminal_pitch_los_rate_gain: float = 0.0
    terminal_vertical_unload_el_deg: float = 0.0
    terminal_vertical_unload_ratio: float = 0.7
    terminal_vertical_unload_bank_scale: float = 0.15
    direct_course_bank_full_error_deg: float = 0.0
    direct_course_bank_exponent: float = 1.0
    direct_course_rear_ambiguity_deg: float = 180.0
    direct_course_roll_bias: float = 0.0
    direct_course_min_bank_deg: float = 0.0
    direct_course_min_bank_ata_deg: float = 0.0
    bank_reversal_error_deg: float = 0.0
    bank_reversal_full_cmd: float = 0.0
    bank_reversal_release_bank_deg: float = 0.0
    # Post-merge rear-hemisphere rate matching. When the target's measured
    # turn is stable, choose its world-yaw direction and demand a bounded
    # rate advantage instead of holding a stale +/-180deg turn sign.
    rear_rate_match_enabled: bool = False
    rear_rate_match_ata_deg: float = 90.0
    rear_rate_match_min_target_rate_degps: float = 2.5
    rear_rate_match_target_gain: float = 1.0
    rear_rate_match_error_gain: float = 0.04
    rear_rate_match_max_error_deg: float = 90.0
    rear_rate_match_limit_degps: float = 15.0

    # Integrated engagement manager (W64+).  The estimator already observes
    # all of these signals; this layer resolves their priorities so terminal
    # tracking, post-merge rate matching, overshoot control, vertical pursuit,
    # and energy recovery cannot issue contradictory commands.
    integrated_manager_enabled: bool = False
    integrated_postmerge_ata_deg: float = 70.0
    integrated_overshoot_ata_deg: float = 35.0
    integrated_overshoot_range_m: float = 2200.0
    integrated_overshoot_closure_mps: float = 60.0
    integrated_overshoot_course_gain: float = 0.18
    integrated_overshoot_rate_gain: float = 0.85
    integrated_overshoot_rate_limit_degps: float = 14.0
    integrated_energy_speed_margin_mps: float = 25.0
    integrated_energy_pull_scale: float = 0.65
    integrated_vertical_elevation_deg: float = 12.0
    integrated_vertical_range_m: float = 3200.0
    integrated_rate_bank_authority: bool = False
    integrated_rate_bank_max_ata_deg: float = 180.0
    integrated_energy_gamma_guard_speed_mps: float = 0.0
    integrated_vertical_follow_ata_deg: float = 180.0
    integrated_mutual_commit_enabled: bool = False
    integrated_mutual_commit_ata_deg: float = 35.0
    integrated_mutual_commit_range_m: float = 1500.0
    integrated_mutual_commit_min_closure_mps: float = 150.0
    integrated_overshoot_allow_defensive: bool = False
    # Preserve a favorable nose position while closing.  This narrow mode
    # prevents the lag aimpoint from commanding the observed opposite-bank
    # reversal after ATA has already reached the single digits.
    attack_conversion_enabled: bool = False
    attack_conversion_max_ata_deg: float = 22.0
    attack_conversion_min_threat_ata_deg: float = 35.0
    attack_conversion_max_aspect_deg: float = 180.0
    attack_conversion_min_range_m: float = 1800.0
    attack_conversion_max_range_m: float = 4000.0
    attack_conversion_target_rate_gain: float = 1.0
    attack_conversion_course_gain: float = 0.25
    attack_conversion_los_rate_gain: float = 0.20
    attack_conversion_rate_limit_degps: float = 16.0
    formula_vpp_enabled: bool = False
    formula_vpp_transition_rate_per_s: float = 0.8
    formula_vpp_lag_distance_m: float = 800.0
    formula_vpp_mutual_lateral_m: float = 500.0
    formula_vpp_projectile_speed_mps: float = 650.0
    formula_vpp_pursuit_gain: float = 0.5
    formula_vpp_navigation_constant: float = 1.0
    formula_vpp_turn_rate_limit_degps: float = 16.0
    formula_vpp_gamma_limit_deg: float = 24.0
    formula_vpp_turn_circle_enabled: bool = False
    formula_vpp_turn_circle_horizon_s: float = 1.0
    formula_vpp_vertical_enabled: bool = True
    formula_vpp_recommit_enabled: bool = False
    formula_vpp_recommit_hold_s: float = 1.8
    formula_vpp_recommit_arm_s: float = 8.0
    formula_vpp_recommit_pass_range_m: float = 1000.0
    formula_vpp_recommit_entry_closure_mps: float = 80.0
    formula_vpp_recommit_exit_closure_mps: float = -20.0
    formula_ballistic_tof_enabled: bool = False
    formula_apg_enabled: bool = False
    formula_apg_gain: float = 0.7
    formula_energy_vertical_enabled: bool = False
    formula_energy_gamma_gain_deg_per_m: float = 0.004
    formula_energy_gamma_limit_deg: float = 8.0
    formula_zem_defense_enabled: bool = False
    formula_zem_horizon_s: float = 1.2
    formula_zem_lateral_accel_mps2: float = 22.0
    formula_zem_turn_rate_degps: float = 16.0
    formula_zem_sign_hold_s: float = 0.6
    # ZEM is an emergency break, not a continuously running pursuit law.
    # Limit its bank authority to a short committed pulse, then force the
    # normal manager to extend/reacquire before another pulse is permitted.
    formula_zem_max_burst_s: float = 0.0
    formula_zem_cooldown_s: float = 0.0
    formula_zem_exit_closure_mps: float = -10.0
    formula_rate_bank_authority: bool = False

    # Receding-horizon horizontal guidance (W74+). Rather than selecting a
    # pursuit mode from fixed gates, score reachable ownship turns against a
    # constant-turn target prediction every synchronized telemetry update.
    predictive_guidance_enabled: bool = False
    predictive_guidance_min_ata_deg: float = 20.0
    predictive_guidance_max_ata_deg: float = 180.0
    predictive_guidance_max_range_m: float = float("inf")
    predictive_guidance_min_threat_ata_deg: float = 0.0
    predictive_horizon_min_s: float = 0.8
    predictive_horizon_max_s: float = 3.0
    predictive_candidate_count: int = 15
    predictive_turn_limit_degps: float = 15.0
    predictive_min_separation_m: float = 650.0
    predictive_max_separation_m: float = 2600.0
    predictive_too_close_weight: float = 0.025
    predictive_far_weight: float = 0.0015
    predictive_rate_change_weight: float = 0.04
    predictive_low_energy_speed_mps: float = 180.0
    predictive_low_energy_turn_weight: float = 0.06
    predictive_turn_slew_degps2: float = 12.0
    predictive_override_min_score_gain: float = 0.0
    predictive_rule_sign_guard_enabled: bool = False
    predictive_max_rule_delta_degps: float = 0.0
    predictive_override_min_rate_delta_degps: float = 0.0
    predictive_response_delay_s: float = 0.0
    predictive_live_turn_envelope: bool = False
    # Short-horizon security strategy (W90+). Score each reachable own turn
    # against several plausible target turns and keep the best worst case.
    predictive_adversarial_enabled: bool = False
    predictive_target_turn_limit_degps: float = 15.0
    predictive_threat_cone_deg: float = 30.0
    predictive_threat_weight: float = 1.5
    predictive_midpoint_weight: float = 0.65
    predictive_sign_guard_error_deg: float = 0.0
    predictive_rear_geometry_weight: float = 0.0
    predictive_rear_geometry_range_m: float = 3000.0

    # Coupled 3-D receding-horizon manoeuvre planner (W95+).  Unlike the
    # horizontal yaw-rate search above, every candidate carries a coordinated
    # turn rate, flight-path angle and energy-consistent speed evolution.  It
    # is evaluated against an adversarial envelope of target turn/climb
    # responses using both aircraft's future antenna angles, range, specific
    # energy and terrain clearance.
    planner3d_enabled: bool = False
    planner3d_horizon_s: float = 2.5
    planner3d_midpoint_weight: float = 0.65
    planner3d_turn_limit_degps: float = 16.0
    planner3d_target_turn_limit_degps: float = 18.0
    planner3d_gamma_deg: float = 24.0
    planner3d_target_gamma_deg: float = 22.0
    planner3d_min_speed_mps: float = 170.0
    planner3d_corner_speed_mps: float = 205.0
    planner3d_chase_speed_mps: float = 230.0
    planner3d_floor_m: float = 1200.0
    planner3d_min_separation_m: float = 500.0
    planner3d_max_separation_m: float = 3200.0
    planner3d_threat_cone_deg: float = 35.0
    planner3d_manoeuvre_hold_s: float = 0.65
    planner3d_rate_change_weight: float = 0.10
    planner3d_gamma_change_weight: float = 0.08
    planner3d_energy_weight: float = 0.003
    planner3d_adaptive_horizon: bool = False
    planner3d_horizon_min_s: float = 0.6
    planner3d_reachable_target_envelope: bool = False
    planner3d_target_turn_delta_degps: float = 6.0
    planner3d_target_gamma_delta_deg: float = 8.0
    planner3d_emergency_replan: bool = False
    planner3d_emergency_hold_s: float = 0.20
    planner3d_research_scoring: bool = False
    planner3d_time_weight_beta: float = 0.55
    planner3d_turn_drag_coeff: float = 0.12
    planner3d_role_tau_s: float = 0.50
    planner3d_attack_range_m: float = 1100.0
    planner3d_attack_range_width_m: float = 900.0
    planner3d_turn_center_weight: float = 0.006

    # Conditional spiral recovery. Preserve the proven coarse-turn pull and
    # intervene only when the nose is no longer gaining ATA while a sustained
    # climb is consuming horizontal turn performance.
    spiral_recovery_enabled: bool = False
    spiral_recovery_min_ata_deg: float = 35.0
    spiral_recovery_vertical_speed_mps: float = 42.0
    spiral_recovery_stall_ata_rate_degps: float = -0.5
    spiral_recovery_pitch_gain: float = 0.010
    spiral_recovery_max_pitch_relief: float = 0.25

    # Rate-deficit 3-D reposition. If a bandit is sustaining a higher turn
    # rate, more pure-pursuit time cannot close ATA. Briefly permit overbank
    # above 90deg to rotate the lift vector through the vertical plane, then
    # return to the normal bank envelope once angular parity is recovered.
    rate_deficit_overbank_enabled: bool = False
    rate_deficit_margin_degps: float = 2.0
    rate_deficit_min_target_rate_degps: float = 8.0
    rate_deficit_min_ata_deg: float = 45.0
    rate_deficit_min_altitude_m: float = 3000.0
    rate_deficit_overbank_deg: float = 100.0

    # Coordinated rudder assist, live pulse identified. On the Unreal plant a
    # positive yaw command produced negative yaw rate; therefore the assist
    # sign is opposite the commanded bank. Disabled by default.
    turn_rudder_assist: float = 0.0
    turn_rudder_min_bank_deg: float = 55.0
    turn_rudder_min_ata_deg: float = 25.0
    high_bank_target_speed_mps: float = 0.0
    high_bank_speed_min_bank_deg: float = 70.0
    high_bank_speed_min_ata_deg: float = 25.0
    high_bank_speed_throttle_base: float = 0.55
    high_bank_speed_throttle_gain: float = 0.015
    high_bank_dynamic_speed_enabled: bool = False
    high_bank_dynamic_near_range_m: float = 1500.0
    high_bank_dynamic_far_range_m: float = 3000.0
    high_bank_dynamic_target_margin_mps: float = 10.0
    high_bank_dynamic_max_speed_mps: float = 265.0

    # Exclusive sequential manoeuvre scheduler. This deliberately avoids the
    # old failure mode where max bank and vertical escape were blended into a
    # predictable spiral. Phases are wings-level setup, vertical displacement,
    # then horizontal recommit.
    sequential_maneuver_enabled: bool = False
    sequential_threat_ata_deg: float = 25.0
    sequential_threat_range_m: float = 2200.0
    sequential_threat_closure_mps: float = 40.0
    sequential_stall_ata_deg: float = 60.0
    sequential_stall_ata_rate_degps: float = -0.25
    sequential_rate_deficit_degps: float = 2.0
    sequential_setup_timeout_s: float = 1.0
    sequential_level_bank_deg: float = 25.0
    sequential_vertical_duration_s: float = 1.1
    sequential_vertical_gamma_deg: float = 28.0
    sequential_recommit_duration_s: float = 2.2
    sequential_cooldown_s: float = 5.0
    sequential_force_climb_below_m: float = 2800.0

    # Roll rate cascade, live pulse-derived.
    bank_kp: float = 1.0
    roll_rate_limit_degps: float = 35.0
    roll_rate_gain_degps_per_unit: float = 140.0
    roll_cmd_limit: float = 0.30

    # Optional initial acquisition commit.  Disabled by default so W14-W16
    # remain bit-for-bit on their original control path.  Live W16 data
    # showed that the normal bank-error cascade reduced roll_cmd too early:
    # it needed about 7s to reach 65deg bank while ATA escaped behind us.
    initial_commit_duration_s: float = 0.0
    initial_commit_ata_deg: float = 60.0
    initial_commit_bank_deg: float = 75.0
    initial_commit_release_bank_deg: float = 65.0
    initial_commit_min_roll_cmd: float = 0.0

    # Optional coarse max-rate turn.  Unlike the short initial commit this
    # holds a measured high-bank state until the target reaches the forward
    # tracking cone.  Disabled by default for W14-W18 compatibility.
    coarse_turn_ata_deg: float = 0.0
    coarse_turn_reentry_ata_deg: float = 0.0
    coarse_turn_bank_deg: float = 0.0
    coarse_turn_roll_bias: float = 0.0

    # Optional short max-authority identification schedule.  These values are
    # zero by default; W20 enables them to find the live plant's upper bound
    # before tuning downward.
    max_test_duration_s: float = 0.0
    max_test_requires_coarse_turn: bool = True
    max_test_roll_full_until_deg: float = 0.0
    max_test_roll_full_cmd: float = 0.0
    max_test_roll_taper_until_deg: float = 0.0
    max_test_roll_taper_cmd: float = 0.0
    max_test_roll_brake_above_deg: float = 0.0
    max_test_roll_brake_cmd: float = 0.0
    max_test_pitch_bank_gate_deg: float = 0.0
    max_test_pitch_ata_gate_deg: float = 0.0
    max_test_pitch_cmd: float = 0.0
    max_test_pitch_relaxed_cmd: float = 0.0
    max_test_pitch_vertical_gain: float = 0.0
    max_test_target_vertical_speed_mps: float = 0.0
    max_test_pitch_bank_relax_gain: float = 0.0
    max_test_min_speed_mps: float = 0.0
    max_test_min_altitude_m: float = 0.0

    # Vertical flight-path and banked-turn pull.
    gamma_limit_deg: float = 15.0
    gamma_kp: float = 1.0
    pitch_rate_limit_degps: float = 10.0
    pitch_rate_gain_degps_per_unit: float = 25.0
    pitch_trim: float = -0.05
    turn_pull_at_max_bank: float = -0.22
    pitch_cmd_limit: float = 0.40
    fine_vertical_los_blend: float = 0.25
    vertical_prediction_horizon_s: float = 0.0
    use_altitude_rate_vertical_prediction: bool = False
    vertical_prediction_stable_horizon_s: float = 1.0
    vertical_prediction_unstable_horizon_s: float = 0.4
    vertical_prediction_min_stable_s: float = 0.7
    vertical_prediction_accel_limit_mps2: float = 35.0
    vertical_maneuver_gamma_limit_deg: float = 0.0
    vertical_maneuver_ata_gate_deg: float = 0.0
    vertical_bank_relief_elevation_deg: float = 0.0
    vertical_bank_relief_min_scale: float = 1.0
    vertical_speed_damping_deadband_mps: float = 0.0
    vertical_speed_damping_gain: float = 0.0
    vertical_speed_damping_track_desired_gamma: bool = False

    # Reactive defensive vertical separation.  The original defensive state
    # only changed gains/throttle and therefore flew an easy, persistent arc.
    defensive_vertical_escape: bool = False
    defensive_escape_threat_ata_deg: float = 20.0
    defensive_escape_own_ata_deg: float = 60.0
    defensive_escape_range_m: float = 1800.0
    defensive_escape_gamma_deg: float = 30.0
    defensive_escape_deadband_m: float = 120.0
    defensive_escape_switch_s: float = 2.3
    defensive_escape_floor_m: float = 0.0

    # After a nose-on threat has been defeated, avoid falling straight back
    # into pure pursuit.  Briefly target a point behind the bandit's velocity
    # vector and outside its measured turn circle, creating a side/rear entry
    # opportunity before terminal tracking takes over.
    post_defense_conversion_duration_s: float = 0.0
    post_defense_conversion_arm_window_s: float = 3.0
    post_defense_conversion_min_threat_ata_deg: float = 40.0
    post_defense_conversion_max_range_m: float = 3000.0
    post_defense_conversion_rear_offset_m: float = 1500.0
    post_defense_conversion_lateral_offset_m: float = 800.0
    post_defense_conversion_min_target_speed_mps: float = 40.0

    # Optional bang-bang sustained pull for coarse acquisition.  Negative
    # pitch command is nose-up on the live plant.  Defaults disable it so
    # earlier controller modes are unchanged.
    sustained_pull_rear_cmd: float = 0.0
    sustained_pull_rear_ata_deg: float = 90.0
    sustained_pull_mid_cmd: float = 0.0
    sustained_pull_mid_ata_deg: float = 60.0
    sustained_pull_min_bank_deg: float = 55.0
    sustained_pull_min_speed_mps: float = 0.0
    sustained_pull_min_altitude_m: float = 0.0

    # State-specific energy management.
    corner_speed_mps: float = 285.0
    corner_throttle_base: float = 0.48
    corner_speed_gain: float = 0.004
    throttle_min: float = 0.20
    throttle_max: float = 0.95
    throttle_slew_per_s: float = 0.45
    throttle_merge: float = 0.90
    throttle_defensive: float = 0.95
    throttle_weapons: float = 0.48

    # Optional closure-aware approach energy control.  Scalar speed matching
    # cannot detect a 300m/s head-on closure when both aircraft have the same
    # airspeed, so use radial closure directly before entering the WEZ.
    closure_throttle_ata_deg: float = 0.0
    closure_throttle_range_m: float = 0.0
    closure_throttle_far_range_m: float = 3000.0
    closure_throttle_near_range_m: float = 1500.0
    closure_target_far_mps: float = 120.0
    closure_target_mid_mps: float = 70.0
    closure_target_near_mps: float = 20.0
    closure_throttle_base: float = 0.50
    closure_throttle_gain: float = 0.0
    # Disabled by default to preserve every historical controller. A finite
    # value is an experiment-specific guard on (closure-desired_closure).
    closure_throttle_error_limit_mps: float = float("inf")
    minimum_energy_speed_mps: float = 0.0
    minimum_energy_throttle: float = 0.0


class IntegratedBFMController(ActionProvider):
    def __init__(self, config: IntegratedBFMConfig | None = None):
        self.cfg = config or IntegratedBFMConfig()
        self.geometry = GeometryInfo()
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._start_time: float | None = None
        self._prev_time: float | None = None
        self._prev_distance: float | None = None
        self._prev_ata: float | None = None
        self._prev_los_az: float | None = None
        self._prev_horizontal_los_error: float | None = None
        self._prev_world_elevation: float | None = None
        self._prev_own_yaw: float | None = None
        self._prev_target_yaw: float | None = None
        self._prev_bank: float | None = None
        self._prev_pitch: float | None = None
        self._prev_alt: float | None = None
        self._prev_target_alt: float | None = None
        self._prev_target_position: np.ndarray | None = None
        self._target_velocity = np.zeros(3, dtype=np.float64)
        self._los_rate = 0.0
        self._horizontal_los_rate = 0.0
        self._world_elevation_rate = 0.0
        self._ata_rate = 0.0
        self._own_yaw_rate = 0.0
        self._target_yaw_rate = 0.0
        self._target_yaw_accel = 0.0
        self._target_turn_stable_s = 0.0
        self._roll_rate = 0.0
        self._pitch_rate = 0.0
        self._target_climb_rate = 0.0
        self._target_vertical_accel = 0.0
        self._target_vertical_stable_s = 0.0
        self._formula_vpp_blend = 0.5
        self._prev_closure = 0.0
        self._formula_vpp_recommit_armed_until = 0.0
        self._formula_vpp_pass_entry_until = 0.0
        self._formula_vpp_recommit_until = 0.0
        self._formula_zem_escape_sign = 1
        self._formula_zem_escape_until = 0.0
        self._formula_zem_burst_until = 0.0
        self._formula_zem_cooldown_until = 0.0
        self._state = "merge"
        self._state_since = 0.0
        self._break_until = 0.0
        self._turn_sign = 1
        self._last_sign_change = -1e9
        self._coarse_turn_initialized = False
        self._coarse_turn_latched = False
        self._turn_match_latched = False
        self._turn_match_candidate_s = 0.0
        self._terminal_track_latched = False
        self._post_defense_conversion_until = 0.0
        self._post_defense_conversion_armed_until = 0.0
        self._predictive_turn_rate = 0.0
        self._predictive_was_active = False
        self._planner3d_until = 0.0
        self._planner3d_turn_rate = 0.0
        self._planner3d_gamma = 0.0
        self._planner3d_target_speed = self.cfg.planner3d_corner_speed_mps
        self._planner3d_name = "inactive"
        self._planner3d_score = 0.0
        self._planner3d_future_ata = 180.0
        self._planner3d_future_threat_ata = 180.0
        self._planner3d_future_range = 0.0
        self._planner3d_worst_target_turn = 0.0
        self._planner3d_worst_target_gamma = 0.0
        self._planner3d_threat_pressure = 0.0
        self._sequential_phase = "none"
        self._sequential_until = 0.0
        self._sequential_cooldown_until = 0.0
        self._sequential_vertical_sign = 1
        self._advantage_score = 0.0
        self._advantage_ttc = 99.0
        self._advantage_manager_reason = "disabled"
        self._advantage_attack_until = 0.0
        self._lag_pursuit_scale = 1.0
        self._lift_vector_bank = 0.0
        self._lift_vector_gamma = 0.0
        self._lift_vector_sign = 1
        self._lift_vector_sign_until = 0.0
        self._lift_vector_was_active = False
        self._throttle = self.cfg.throttle_merge
        self.state_log: list[str] = []
        self.info_log: list[dict] = []

    def _ema(self, previous: float, current: float) -> float:
        alpha = self.cfg.rate_alpha
        return alpha * current + (1.0 - alpha) * previous

    def _set_state(self, state: str, now: float) -> None:
        if state != self._state:
            self._state = state
            self._state_since = now

    def _intercept_horizon(
        self,
        state: str,
        relative_position: np.ndarray,
        target_velocity: np.ndarray,
        own_speed: float,
    ) -> float:
        cfg = self.cfg
        if state == "merge":
            return cfg.horizon_merge_s
        if state == "break_turn":
            return cfg.horizon_break_s
        if state == "defensive":
            return cfg.horizon_defensive_s
        if state == "weapons_track":
            return cfg.horizon_weapons_s

        max_horizon = (
            cfg.horizon_track_max_s if state == "offensive_track"
            else cfg.horizon_reacquire_max_s
        )

        # Constant-velocity intercept: |r + vt| = own_speed * t.
        r2 = float(np.dot(relative_position, relative_position))
        a = float(np.dot(target_velocity, target_velocity) - own_speed * own_speed)
        b = 2.0 * float(np.dot(relative_position, target_velocity))
        roots: list[float] = []
        if abs(a) < 1e-6:
            if abs(b) > 1e-6:
                roots.append(-r2 / b)
        else:
            discriminant = b * b - 4.0 * a * r2
            if discriminant >= 0.0:
                root = math.sqrt(discriminant)
                roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        positive = [value for value in roots if value > 0.0 and math.isfinite(value)]
        if positive:
            return float(np.clip(min(positive), 0.1, max_horizon))
        time_to_range = math.sqrt(r2) / max(own_speed, 1.0)
        return float(np.clip(time_to_range, 0.1, max_horizon))

    @staticmethod
    def _predict_3d_state(
        position: np.ndarray,
        heading_deg: float,
        speed_mps: float,
        turn_rate_degps: float,
        gamma_deg: float,
        horizon_s: float,
        engine_accel_mps2: float,
        turn_drag_coeff: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """Propagate a constant-command 3-D coordinated manoeuvre.

        Altitude is positive-up in this local planner.  Speed changes include
        the gravity component along the flight path, preventing a climb from
        receiving free potential energy in the trajectory score.
        """
        gamma = math.radians(gamma_deg)
        omega = math.radians(turn_rate_degps)
        lateral_accel = abs(speed_mps * math.cos(gamma) * omega)
        load_factor_sq = 1.0 + (lateral_accel / G) ** 2
        induced_turn_decel = turn_drag_coeff * max(0.0, load_factor_sq - 1.0)
        accel = engine_accel_mps2 - G * math.sin(gamma) - induced_turn_decel
        final_speed = max(60.0, speed_mps + accel * horizon_s)
        mean_speed = max(60.0, 0.5 * (speed_mps + final_speed))
        horizontal_speed = mean_speed * math.cos(gamma)
        horizontal, final_heading = IntegratedBFMController._predict_horizontal_state(
            np.asarray(position[:2], dtype=np.float64),
            heading_deg,
            horizontal_speed,
            turn_rate_degps,
            horizon_s,
        )
        final_position = np.array(
            [horizontal[0], horizontal[1], position[2] + mean_speed * math.sin(gamma) * horizon_s],
            dtype=np.float64,
        )
        return final_position, final_heading, final_speed

    @staticmethod
    def _velocity_direction(heading_deg: float, gamma_deg: float) -> np.ndarray:
        heading = math.radians(heading_deg)
        gamma = math.radians(gamma_deg)
        return np.array([
            math.cos(gamma) * math.cos(heading),
            math.cos(gamma) * math.sin(heading),
            math.sin(gamma),
        ], dtype=np.float64)

    @staticmethod
    def _angle_to_los(direction: np.ndarray, los: np.ndarray) -> float:
        norm = float(np.linalg.norm(los))
        if norm < 1e-6:
            return 180.0
        cosine = float(np.clip(np.dot(direction, los / norm), -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def _manager(
        self,
        now: float,
        elapsed: float,
        ata: float,
        threat_ata: float,
        distance: float,
        closure: float,
        own_speed: float,
        target_speed: float,
    ) -> str:
        cfg = self.cfg
        if cfg.advantage_manager_enabled:
            angle_term = float(np.clip(
                (threat_ata - ata) / max(cfg.advantage_angle_scale_deg, 1.0),
                -1.0,
                1.0,
            ))
            energy_term = float(np.clip(
                (own_speed - target_speed) / max(cfg.advantage_energy_scale_mps, 1.0),
                -1.0,
                1.0,
            ))
            ttc = distance / max(closure, 1.0) if closure > 0.0 else 99.0
            collision_penalty = float(np.clip(
                (cfg.advantage_collision_ttc_s - ttc)
                / max(cfg.advantage_collision_width_s, 0.1),
                0.0,
                1.0,
            ))
            self._advantage_score = (
                angle_term
                + cfg.advantage_energy_weight * energy_term
                - cfg.advantage_collision_weight * collision_penalty
            )
            self._advantage_ttc = ttc
            hard_threat = (
                threat_ata <= cfg.advantage_hard_threat_ata_deg
                and ata > threat_ata
            )
            attack_window = (
                ata <= cfg.advantage_attack_ata_deg
                and distance <= cfg.advantage_attack_range_m
                and self._advantage_score >= cfg.advantage_attack_score
                and ttc >= cfg.advantage_min_attack_ttc_s
                and not hard_threat
            )
            if attack_window:
                self._advantage_attack_until = now + cfg.advantage_attack_hold_s
            if hard_threat:
                self._advantage_attack_until = 0.0
            attack_hold = now < self._advantage_attack_until and not hard_threat
            if attack_window or attack_hold:
                self._advantage_manager_reason = (
                    "attack_window" if attack_window else "attack_hold"
                )
                if ata <= cfg.weapons_ata_deg and distance <= cfg.weapons_range_m:
                    return "weapons_track"
                return "offensive_track"
            self._advantage_manager_reason = (
                "hard_threat" if hard_threat else "no_attack_window"
            )
        else:
            self._advantage_score = 0.0
            self._advantage_ttc = 99.0
            self._advantage_manager_reason = "disabled"
        if (
            self._state == "defensive"
            and cfg.defensive_min_hold_s > 0.0
            and now - self._state_since < cfg.defensive_min_hold_s
        ):
            return "defensive"
        immediate_threat = (
            threat_ata <= cfg.defensive_threat_ata_deg
            and distance <= cfg.defensive_range_m
            and closure >= cfg.defensive_closure_mps
        )
        if immediate_threat:
            return "defensive"
        if ata <= cfg.weapons_ata_deg and distance <= cfg.weapons_range_m:
            return "weapons_track"
        if ata <= cfg.track_ata_deg:
            return "offensive_track"
        if elapsed <= cfg.merge_initial_s:
            self._break_until = max(self._break_until, now + cfg.break_hold_s)
            return "merge"
        if now <= self._break_until:
            return "break_turn"
        return "reacquire"

    def _turn_rate_limit(self, state: str) -> float:
        cfg = self.cfg
        return {
            "merge": cfg.turn_rate_merge_degps,
            "break_turn": cfg.turn_rate_break_degps,
            "reacquire": cfg.turn_rate_reacquire_degps,
            "offensive_track": cfg.turn_rate_track_degps,
            "weapons_track": cfg.turn_rate_weapons_degps,
            "defensive": cfg.turn_rate_defensive_degps,
        }[state]

    @staticmethod
    def _predict_horizontal_state(
        position_ne: np.ndarray,
        heading_deg: float,
        speed_mps: float,
        turn_rate_degps: float,
        horizon_s: float,
    ) -> tuple[np.ndarray, float]:
        """Propagate a coordinated constant-speed horizontal turn."""
        omega = math.radians(turn_rate_degps)
        heading = math.radians(heading_deg)
        if abs(omega) < 1e-6:
            delta = np.array([
                speed_mps * math.cos(heading) * horizon_s,
                speed_mps * math.sin(heading) * horizon_s,
            ])
        else:
            final_heading = heading + omega * horizon_s
            delta = np.array([
                speed_mps * (math.sin(final_heading) - math.sin(heading)) / omega,
                speed_mps * (-math.cos(final_heading) + math.cos(heading)) / omega,
            ])
        return position_ne + delta, wrap180(
            heading_deg + turn_rate_degps * horizon_s
        )

    def compute_action(self, context: ActionContext) -> ActionResult:
        cfg = self.cfg
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            action = np.array([0.0, cfg.pitch_trim, 0.0, cfg.throttle_merge], dtype=np.float32)
            return ActionResult(action, f"{cfg.controller_name}[no_target]", info={"state": "no_target"})

        now = float(own[StateIndex.SIM_TIME])
        if self._start_time is None:
            self._start_time = now
            self._state_since = now
        elapsed = now - self._start_time
        dt = None
        if self._prev_time is not None and now > self._prev_time:
            dt = now - self._prev_time

        own_position = np.asarray(own[StateIndex.N : StateIndex.D + 1], dtype=np.float64)
        target_position = np.asarray(target[StateIndex.N : StateIndex.D + 1], dtype=np.float64)
        relative_position = target_position - own_position
        own_speed = max(1.0, float(own[StateIndex.KCAS]))
        target_speed = max(0.0, float(target[StateIndex.KCAS]))
        altitude = float(own[StateIndex.ALT])
        target_altitude = float(target[StateIndex.ALT])
        bank = wrap180(float(own[StateIndex.ROLL]))
        pitch = float(own[StateIndex.PITCH])
        own_yaw = float(own[StateIndex.YAW])
        target_yaw = float(target[StateIndex.YAW])

        distance = float(self.geometry._get_distance(own, target))
        los_az, los_el = self.geometry._get_los_angle(own, target)
        los_az, los_el = float(los_az), float(los_el)
        horizontal_bearing = math.degrees(math.atan2(
            relative_position[1], relative_position[0]
        ))
        horizontal_los_error = wrap180(horizontal_bearing - own_yaw)
        ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
        aa = abs(float(self.geometry._get_aspect_angle(own, target, False)))
        threat_ata = abs(float(self.geometry._get_antenna_train_angle(target, own, False)))

        closure = ata_rate = los_rate = 0.0
        horizontal_los_rate = 0.0
        raw_roll_rate = raw_pitch_rate = own_yaw_rate = target_yaw_rate = 0.0
        vertical_speed = 0.0
        target_climb_rate = 0.0
        if dt:
            if self._prev_distance is not None:
                closure = (self._prev_distance - distance) / dt
            if self._prev_ata is not None:
                ata_rate = (ata - self._prev_ata) / dt
            if self._prev_los_az is not None:
                los_rate = wrap180(los_az - self._prev_los_az) / dt
            if self._prev_horizontal_los_error is not None:
                horizontal_los_rate = wrap180(
                    horizontal_los_error - self._prev_horizontal_los_error
                ) / dt
            if self._prev_own_yaw is not None:
                own_yaw_rate = wrap180(own_yaw - self._prev_own_yaw) / dt
            if self._prev_target_yaw is not None:
                target_yaw_rate = wrap180(target_yaw - self._prev_target_yaw) / dt
            if self._prev_bank is not None:
                raw_roll_rate = wrap180(bank - self._prev_bank) / dt
            if self._prev_pitch is not None:
                raw_pitch_rate = wrap180(pitch - self._prev_pitch) / dt
            if self._prev_alt is not None:
                vertical_speed = (altitude - self._prev_alt) / dt
            if self._prev_target_alt is not None:
                target_climb_rate = (target_altitude - self._prev_target_alt) / dt
            if self._prev_target_position is not None:
                measured_velocity = (target_position - self._prev_target_position) / dt
                velocity_norm = float(np.linalg.norm(measured_velocity))
                if velocity_norm > cfg.target_velocity_clip_mps:
                    measured_velocity *= cfg.target_velocity_clip_mps / velocity_norm
                alpha = cfg.target_velocity_alpha
                self._target_velocity = alpha * measured_velocity + (1.0 - alpha) * self._target_velocity

        self._los_rate = self._ema(self._los_rate, los_rate)
        self._horizontal_los_rate = self._ema(
            self._horizontal_los_rate, horizontal_los_rate
        )
        self._ata_rate = self._ema(self._ata_rate, ata_rate)
        self._own_yaw_rate = self._ema(self._own_yaw_rate, own_yaw_rate)
        previous_filtered_target_yaw_rate = self._target_yaw_rate
        self._target_yaw_rate = self._ema(self._target_yaw_rate, target_yaw_rate)
        if dt:
            raw_target_yaw_accel = (
                self._target_yaw_rate - previous_filtered_target_yaw_rate
            ) / dt
            self._target_yaw_accel = self._ema(
                self._target_yaw_accel, raw_target_yaw_accel
            )
            same_turn = (
                abs(self._target_yaw_rate) >= 0.5
                and abs(previous_filtered_target_yaw_rate) >= 0.5
                and self._target_yaw_rate * previous_filtered_target_yaw_rate > 0.0
            )
            if same_turn:
                self._target_turn_stable_s += dt
            else:
                self._target_turn_stable_s = 0.0
        self._roll_rate = self._ema(self._roll_rate, raw_roll_rate)
        self._pitch_rate = self._ema(self._pitch_rate, raw_pitch_rate)
        previous_target_climb_rate = self._target_climb_rate
        self._target_climb_rate = self._ema(
            self._target_climb_rate, target_climb_rate
        )
        if dt:
            raw_vertical_accel = (
                self._target_climb_rate - previous_target_climb_rate
            ) / dt
            self._target_vertical_accel = self._ema(
                self._target_vertical_accel, raw_vertical_accel
            )
            same_vertical_direction = (
                abs(self._target_climb_rate) >= 3.0
                and abs(previous_target_climb_rate) >= 3.0
                and self._target_climb_rate * previous_target_climb_rate > 0.0
            )
            if same_vertical_direction:
                self._target_vertical_stable_s += dt
            else:
                self._target_vertical_stable_s = 0.0

        previous_state = self._state
        state = self._manager(
            now, elapsed, ata, threat_ata, distance, closure,
            own_speed, target_speed,
        )
        if cfg.sequential_maneuver_enabled:
            if self._sequential_phase == "setup" and (
                abs(bank) <= cfg.sequential_level_bank_deg
                or now >= self._sequential_until
            ):
                self._sequential_phase = "vertical"
                self._sequential_until = now + cfg.sequential_vertical_duration_s
            elif self._sequential_phase == "vertical" and now >= self._sequential_until:
                self._sequential_phase = "recommit"
                self._sequential_until = now + cfg.sequential_recommit_duration_s
            elif self._sequential_phase == "recommit" and now >= self._sequential_until:
                self._sequential_phase = "none"
                self._sequential_cooldown_until = now + cfg.sequential_cooldown_s

            immediate_sequence_threat = (
                threat_ata <= cfg.sequential_threat_ata_deg
                and distance <= cfg.sequential_threat_range_m
                and closure >= cfg.sequential_threat_closure_mps
                and ata >= cfg.weapons_ata_deg
            )
            stalled_sequence_attack = (
                ata >= cfg.sequential_stall_ata_deg
                and self._ata_rate >= cfg.sequential_stall_ata_rate_degps
                and abs(self._target_yaw_rate)
                >= abs(self._own_yaw_rate) + cfg.sequential_rate_deficit_degps
            )
            if (
                self._sequential_phase == "none"
                and now >= self._sequential_cooldown_until
                and (immediate_sequence_threat or stalled_sequence_attack)
            ):
                self._sequential_phase = "setup"
                self._sequential_until = now + cfg.sequential_setup_timeout_s
                if altitude <= cfg.sequential_force_climb_below_m:
                    self._sequential_vertical_sign = 1
                elif target_altitude > altitude:
                    self._sequential_vertical_sign = -1
                else:
                    self._sequential_vertical_sign = 1
        if (
            cfg.post_defense_conversion_duration_s > 0.0
            and previous_state == "defensive"
            and state != "defensive"
        ):
            if cfg.post_defense_conversion_arm_window_s > 0.0:
                self._post_defense_conversion_armed_until = (
                    now + cfg.post_defense_conversion_arm_window_s
                )
            elif (
                threat_ata >= cfg.post_defense_conversion_min_threat_ata_deg
                and distance <= cfg.post_defense_conversion_max_range_m
            ):
                self._post_defense_conversion_until = (
                    now + cfg.post_defense_conversion_duration_s
                )
        if state == "defensive":
            self._post_defense_conversion_until = 0.0
            self._post_defense_conversion_armed_until = 0.0
        elif (
            now < self._post_defense_conversion_armed_until
            and threat_ata >= cfg.post_defense_conversion_min_threat_ata_deg
            and distance <= cfg.post_defense_conversion_max_range_m
        ):
            self._post_defense_conversion_until = (
                now + cfg.post_defense_conversion_duration_s
            )
            self._post_defense_conversion_armed_until = 0.0
        if (
            now < self._post_defense_conversion_until
            and threat_ata < cfg.post_defense_conversion_min_threat_ata_deg
        ):
            self._post_defense_conversion_until = 0.0
        if distance > cfg.post_defense_conversion_max_range_m:
            self._post_defense_conversion_until = 0.0
        self._set_state(state, now)

        # Predict horizontal target position only.  Vertical extrapolation was
        # explicitly rejected by W12 live data.
        horizontal_relative = relative_position.copy()
        horizontal_relative[2] = 0.0
        horizontal_range_now = max(
            1.0, float(np.hypot(relative_position[0], relative_position[1]))
        )
        world_elevation = math.degrees(math.atan2(
            target_altitude - altitude, horizontal_range_now
        ))
        if (
            self._prev_world_elevation is not None
            and dt is not None
            and dt > 1e-6
        ):
            raw_world_elevation_rate = (
                world_elevation - self._prev_world_elevation
            ) / dt
            self._world_elevation_rate = self._ema(
                self._world_elevation_rate, raw_world_elevation_rate
            )
        self._prev_world_elevation = world_elevation
        horizontal_velocity = self._target_velocity.copy()
        horizontal_velocity[2] = 0.0
        horizon = self._intercept_horizon(state, horizontal_relative, horizontal_velocity, own_speed)
        prediction_stable = True
        if cfg.adaptive_turn_prediction:
            prediction_stable = (
                self._target_turn_stable_s >= cfg.turn_prediction_min_stable_s
                and abs(self._target_yaw_accel)
                <= cfg.turn_prediction_yaw_accel_limit_degps2
            )
            if not prediction_stable:
                horizon = min(horizon, cfg.turn_prediction_unstable_horizon_s)
            if abs(self._target_yaw_rate) >= 0.5:
                arc_limited_horizon = (
                    cfg.turn_prediction_max_arc_deg / abs(self._target_yaw_rate)
                )
                horizon = min(horizon, arc_limited_horizon)
        aim_target = np.array(target, copy=True)
        if (
            cfg.use_constant_turn_prediction
            and abs(self._target_yaw_rate) >= 0.5
            and horizon > 0.0
        ):
            omega = math.radians(self._target_yaw_rate)
            angle = omega * horizon
            sin_a, cos_a = math.sin(angle), math.cos(angle)
            vn, ve = horizontal_velocity[0], horizontal_velocity[1]
            delta_n = (vn * sin_a + ve * (cos_a - 1.0)) / omega
            delta_e = (vn * (1.0 - cos_a) + ve * sin_a) / omega
            aim_target[StateIndex.N] = target_position[0] + delta_n
            aim_target[StateIndex.E] = target_position[1] + delta_e
        else:
            aim_target[StateIndex.N] = target_position[0] + horizontal_velocity[0] * horizon
            aim_target[StateIndex.E] = target_position[1] + horizontal_velocity[1] * horizon

        post_defense_conversion_active = (
            cfg.post_defense_conversion_duration_s > 0.0
            and now < self._post_defense_conversion_until
            and state != "defensive"
            and threat_ata >= cfg.post_defense_conversion_min_threat_ata_deg
            and distance <= cfg.post_defense_conversion_max_range_m
        )
        conversion_rear_offset_m = 0.0
        conversion_lateral_offset_m = 0.0
        horizontal_target_speed = float(np.linalg.norm(horizontal_velocity[:2]))
        if (
            post_defense_conversion_active
            and horizontal_target_speed
            >= cfg.post_defense_conversion_min_target_speed_mps
        ):
            track_unit = horizontal_velocity[:2] / horizontal_target_speed
            left_normal = np.array([-track_unit[1], track_unit[0]])
            turn_sign = signed_unit(self._target_yaw_rate, self._turn_sign)
            outside_normal = -turn_sign * left_normal
            conversion_rear_offset_m = cfg.post_defense_conversion_rear_offset_m
            conversion_lateral_offset_m = cfg.post_defense_conversion_lateral_offset_m
            aim_target[StateIndex.N] -= track_unit[0] * conversion_rear_offset_m
            aim_target[StateIndex.E] -= track_unit[1] * conversion_rear_offset_m
            aim_target[StateIndex.N] += outside_normal[0] * conversion_lateral_offset_m
            aim_target[StateIndex.E] += outside_normal[1] * conversion_lateral_offset_m

        mutual_commit_candidate = (
            cfg.integrated_mutual_commit_enabled
            and state == "defensive"
            and ata <= cfg.integrated_mutual_commit_ata_deg
            and distance <= cfg.integrated_mutual_commit_range_m
            and closure >= cfg.integrated_mutual_commit_min_closure_mps
        )
        terminal_track_enabled = cfg.terminal_track_enter_ata_deg > 0.0
        if terminal_track_enabled:
            if self._terminal_track_latched:
                if (
                    ata >= cfg.terminal_track_exit_ata_deg
                    or distance >= cfg.terminal_track_exit_range_m
                    or aa > cfg.terminal_track_max_aspect_deg
                    or (
                        threat_ata < 0.5 * cfg.terminal_track_min_threat_ata_deg
                        and not mutual_commit_candidate
                    )
                ):
                    self._terminal_track_latched = False
            elif (
                (
                    ata <= cfg.terminal_track_enter_ata_deg
                    and distance <= cfg.terminal_track_enter_range_m
                )
                or (
                    cfg.terminal_track_prelock_ata_deg > 0.0
                    and ata <= cfg.terminal_track_prelock_ata_deg
                    and distance <= cfg.terminal_track_prelock_range_m
                )
            ) and (
                threat_ata >= cfg.terminal_track_min_threat_ata_deg
                or mutual_commit_candidate
            ) and (
                aa <= cfg.terminal_track_max_aspect_deg
            ) and (
                not cfg.advantage_manager_enabled
                or self._advantage_ttc >= cfg.advantage_min_attack_ttc_s
            ):
                self._terminal_track_latched = True
        else:
            self._terminal_track_latched = False
        terminal_track_active = self._terminal_track_latched
        if (
            cfg.integrated_manager_enabled
            and state == "defensive"
            and not mutual_commit_candidate
        ):
            # Threat survival has strict priority over a stale fine-track
            # latch inherited from the previous frame.
            self._terminal_track_latched = False
            terminal_track_active = False
        if terminal_track_active:
            # Inside the gun setup, stop aiming at a lag/intercept proxy.  The
            # live damage cone is only 1/2/3 degrees, so point at the actual
            # aircraft and use rate feed-forward below to keep it there.
            aim_target[StateIndex.N] = target_position[0]
            aim_target[StateIndex.E] = target_position[1]

        attack_conversion_active = (
            cfg.attack_conversion_enabled
            and state != "defensive"
            and ata <= cfg.attack_conversion_max_ata_deg
            and threat_ata >= cfg.attack_conversion_min_threat_ata_deg
            and aa <= cfg.attack_conversion_max_aspect_deg
            and cfg.attack_conversion_min_range_m <= distance
            <= cfg.attack_conversion_max_range_m
        )

        headon_deconflict_active = (
            cfg.headon_deconflict_enabled
            and ata <= cfg.headon_deconflict_max_ata_deg
            and threat_ata <= cfg.headon_deconflict_max_threat_ata_deg
            and aa >= cfg.headon_deconflict_min_aspect_deg
            and distance <= cfg.headon_deconflict_range_m
            and closure >= cfg.headon_deconflict_min_closure_mps
        )

        # Once nearly aligned, high radial closure calls for lag pursuit rather
        # than an even farther lead point.  Move the aim point behind the
        # target along its measured ground track; disengage immediately if ATA
        # opens so acquisition authority is never sacrificed.
        lag_allowed_state = not (
            cfg.lag_pursuit_disable_defensive and state == "defensive"
        )
        if cfg.lag_pursuit_advantage_full_deg > 0.0:
            desired_lag_scale = float(np.clip(
                (threat_ata - ata) / cfg.lag_pursuit_advantage_full_deg,
                0.0,
                1.0,
            ))
            if dt and cfg.lag_pursuit_scale_tau_s > 0.0:
                blend = float(np.clip(
                    dt / cfg.lag_pursuit_scale_tau_s, 0.0, 1.0
                ))
                self._lag_pursuit_scale += blend * (
                    desired_lag_scale - self._lag_pursuit_scale
                )
            else:
                self._lag_pursuit_scale = desired_lag_scale
        else:
            self._lag_pursuit_scale = 1.0
        normal_lag_active = (
            cfg.lag_pursuit_offset_max_m > 0.0
            and lag_allowed_state
            and self._lag_pursuit_scale > 0.05
            and ata <= cfg.lag_pursuit_ata_deg
            and distance <= cfg.lag_pursuit_range_m
            and closure >= cfg.lag_pursuit_closure_mps
        )
        vertical_alignment_active = (
            cfg.vertical_alignment_elevation_deg > 0.0
            and abs(world_elevation) >= cfg.vertical_alignment_elevation_deg
            and distance <= cfg.vertical_alignment_range_m
            and closure >= cfg.vertical_alignment_closure_mps
        )
        lag_pursuit_active = (
            (normal_lag_active or vertical_alignment_active)
            and not terminal_track_active
            and not attack_conversion_active
        )
        lag_offset_m = 0.0
        horizontal_target_speed = float(np.linalg.norm(horizontal_velocity[:2]))
        if lag_pursuit_active and horizontal_target_speed > 1.0:
            if normal_lag_active:
                lag_offset_m = float(np.clip(
                    cfg.lag_pursuit_offset_min_m
                    + cfg.lag_pursuit_offset_gain_s
                    * (closure - cfg.lag_pursuit_closure_mps),
                    cfg.lag_pursuit_offset_min_m,
                    cfg.lag_pursuit_offset_max_m,
                ))
                lag_offset_m *= self._lag_pursuit_scale
            if vertical_alignment_active:
                vertical_lag_offset = cfg.vertical_alignment_lag_offset_m
                if cfg.vertical_alignment_lag_gain_m_per_deg > 0.0:
                    vertical_lag_offset += (
                        cfg.vertical_alignment_lag_gain_m_per_deg
                        * max(
                            0.0,
                            abs(world_elevation)
                            - cfg.vertical_alignment_elevation_deg,
                        )
                    )
                    if cfg.vertical_alignment_lag_max_m > 0.0:
                        vertical_lag_offset = min(
                            vertical_lag_offset,
                            cfg.vertical_alignment_lag_max_m,
                        )
                lag_offset_m = max(lag_offset_m, vertical_lag_offset)
            taper_span = (
                cfg.lag_pursuit_terminal_taper_start_deg
                - cfg.lag_pursuit_terminal_taper_end_deg
            )
            if (
                taper_span > 0.0
                and ata < cfg.lag_pursuit_terminal_taper_start_deg
                and closure <= cfg.lag_pursuit_taper_max_closure_mps
            ):
                lag_offset_m *= float(np.clip(
                    (ata - cfg.lag_pursuit_terminal_taper_end_deg) / taper_span,
                    0.0,
                    1.0,
                ))
            track_unit = horizontal_velocity[:2] / horizontal_target_speed
            aim_target[StateIndex.N] -= track_unit[0] * lag_offset_m
            aim_target[StateIndex.E] -= track_unit[1] * lag_offset_m
            if (
                cfg.lag_pursuit_mutual_lateral_offset_m > 0.0
                and cfg.lag_pursuit_mutual_threat_ata_deg > 0.0
                and threat_ata < cfg.lag_pursuit_mutual_threat_ata_deg
            ):
                # A rear offset alone preserves a symmetric nose-on pass.
                # Move toward the outside of the target's measured turn
                # circle so the next merge develops lateral separation and a
                # side/rear-quarter opportunity. Fade continuously as the
                # target loses its firing aspect.
                lateral_scale = float(np.clip(
                    1.0 - threat_ata / cfg.lag_pursuit_mutual_threat_ata_deg,
                    0.0,
                    1.0,
                ))
                left_normal = np.array([-track_unit[1], track_unit[0]])
                target_turn_sign = (
                    signed_unit(self._target_yaw_rate)
                    if abs(self._target_yaw_rate) >= 2.5
                    else self._turn_sign
                )
                outside_normal = -target_turn_sign * left_normal
                lateral_offset = (
                    cfg.lag_pursuit_mutual_lateral_offset_m * lateral_scale
                )
                aim_target[StateIndex.N] += outside_normal[0] * lateral_offset
                aim_target[StateIndex.E] += outside_normal[1] * lateral_offset
        if headon_deconflict_active:
            # This path deliberately sits outside lag-pursuit state gating:
            # W100 classifies the dangerous mutual pass as defensive, where
            # lag pursuit is disabled.  Use the target ground track (or its
            # reported yaw before the velocity estimator settles) to create a
            # stable one-side crossing point.  _turn_sign supplies hysteresis
            # and prevents a 60 Hz left/right command chatter.
            if horizontal_target_speed >= 5.0:
                headon_track = horizontal_velocity[:2] / horizontal_target_speed
            else:
                target_yaw_rad = math.radians(target_yaw)
                headon_track = np.array([
                    math.cos(target_yaw_rad), math.sin(target_yaw_rad)
                ])
            headon_normal = np.array([-headon_track[1], headon_track[0]])
            headon_side = (
                signed_unit(self._target_yaw_rate, self._turn_sign)
                if abs(self._target_yaw_rate) >= 2.5
                else self._turn_sign
            )
            headon_offset = cfg.headon_deconflict_lateral_offset_m
            aim_target[StateIndex.N] += headon_side * headon_normal[0] * headon_offset
            aim_target[StateIndex.E] += headon_side * headon_normal[1] * headon_offset
        # W102 is an overlay on the proven W100 controller, not a replacement
        # pilot.  Only take horizontal guidance during W100's established
        # attack-conversion window, or briefly to deconflict an imminent
        # mutual nose-on pass.  W101/W102-live initially enabled this in every
        # non-defensive state and overwrote W100 acquisition/reacquire turns.
        formula_mutual_deconflict = (
            cfg.formula_vpp_turn_circle_enabled
            and distance <= 1800.0
            and ata <= 35.0
            and threat_ata < 30.0
            and closure > 80.0
        )
        if cfg.formula_vpp_recommit_enabled and ata <= 20.0:
            self._formula_vpp_recommit_armed_until = max(
                self._formula_vpp_recommit_armed_until,
                now + cfg.formula_vpp_recommit_arm_s,
            )
        if (
            cfg.formula_vpp_recommit_enabled
            and now <= self._formula_vpp_recommit_armed_until
            and distance <= 1500.0
            and closure >= cfg.formula_vpp_recommit_entry_closure_mps
        ):
            # Closure decelerates continuously through zero; remember that a
            # fast, aligned merge was entered instead of requiring an
            # impossible +80 -> -20 m/s jump in one telemetry frame.
            self._formula_vpp_pass_entry_until = max(
                self._formula_vpp_pass_entry_until,
                now + 4.0,
            )
        formula_vpp_pass_detected = (
            cfg.formula_vpp_recommit_enabled
            and now <= self._formula_vpp_recommit_armed_until
            and now <= self._formula_vpp_pass_entry_until
            and distance <= cfg.formula_vpp_recommit_pass_range_m
            and closure <= cfg.formula_vpp_recommit_exit_closure_mps
        )
        if formula_vpp_pass_detected:
            self._formula_vpp_recommit_until = max(
                self._formula_vpp_recommit_until,
                now + cfg.formula_vpp_recommit_hold_s,
            )
            self._formula_vpp_pass_entry_until = 0.0
        formula_vpp_recommit_active = (
            cfg.formula_vpp_recommit_enabled
            and now < self._formula_vpp_recommit_until
        )
        formula_vpp_active = (
            cfg.formula_vpp_enabled
            and state != "defensive"
            and dt is not None
            and (
                not cfg.formula_vpp_turn_circle_enabled
                or attack_conversion_active
                or formula_mutual_deconflict
                or formula_vpp_recommit_active
            )
        )
        formula_vpp_result = None
        formula_zem_defense_active = False
        formula_zem_escape_sign = 0
        formula_zem_eligible = (
            cfg.formula_zem_defense_enabled
            and state == "defensive"
            and dt is not None
        )
        # A zero burst limit preserves the legacy continuous-ZEM behaviour for
        # every existing mode. W104 opts into the pulse/cooldown manager.
        formula_zem_pulsed = cfg.formula_zem_max_burst_s > 0.0
        if formula_zem_eligible:
            own_yaw_rad = math.radians(own_yaw)
            own_hspeed = math.sqrt(max(own_speed * own_speed - vertical_speed * vertical_speed, 1.0))
            own_vel_xy = np.array([
                own_hspeed * math.cos(own_yaw_rad),
                own_hspeed * math.sin(own_yaw_rad),
            ])
            rel_xy = target_position[:2] - own_position[:2]
            rel_vel_xy = self._target_velocity[:2] - own_vel_xy
            rel_speed_sq = float(np.dot(rel_vel_xy, rel_vel_xy))
            t_zem = float(np.clip(
                -float(np.dot(rel_xy, rel_vel_xy)) / max(rel_speed_sq, 1.0),
                0.2,
                cfg.formula_zem_horizon_s,
            ))
            own_track_xy = own_vel_xy / max(float(np.linalg.norm(own_vel_xy)), 1.0)
            left_xy = np.array([-own_track_xy[1], own_track_xy[0]])
            scores = {}
            for escape_sign in (-1, 1):
                own_accel = escape_sign * cfg.formula_zem_lateral_accel_mps2 * left_xy
                zem = rel_xy + rel_vel_xy * t_zem - 0.5 * own_accel * t_zem * t_zem
                scores[escape_sign] = float(np.linalg.norm(zem))
            start_zem_pulse = (
                formula_zem_pulsed
                and now >= self._formula_zem_burst_until
                and now >= self._formula_zem_cooldown_until
            )
            if start_zem_pulse:
                self._formula_zem_escape_sign = max(scores, key=scores.get)
                self._formula_zem_burst_until = now + cfg.formula_zem_max_burst_s
                self._formula_zem_cooldown_until = (
                    self._formula_zem_burst_until + cfg.formula_zem_cooldown_s
                )
            elif not formula_zem_pulsed and now >= self._formula_zem_escape_until:
                self._formula_zem_escape_sign = max(scores, key=scores.get)
                self._formula_zem_escape_until = now + cfg.formula_zem_sign_hold_s

            formula_zem_escape_sign = self._formula_zem_escape_sign
            if formula_zem_pulsed:
                formula_zem_defense_active = (
                    now < self._formula_zem_burst_until
                    and closure > cfg.formula_zem_exit_closure_mps
                )
                # Once closest approach has passed and range is opening, end
                # the emergency break immediately but retain the cooldown.
                if not formula_zem_defense_active:
                    self._formula_zem_burst_until = min(
                        self._formula_zem_burst_until,
                        now,
                    )
            else:
                formula_zem_defense_active = True
        if formula_vpp_active:
            yaw_rad = math.radians(own_yaw)
            own_horizontal_speed = math.sqrt(max(own_speed * own_speed - vertical_speed * vertical_speed, 1.0))
            own_velocity_3d = np.array([
                own_horizontal_speed * math.cos(yaw_rad),
                own_horizontal_speed * math.sin(yaw_rad),
                -vertical_speed,
            ])
            target_velocity_3d = self._target_velocity.copy()
            if float(np.linalg.norm(target_velocity_3d)) < 20.0:
                target_yaw_rad = math.radians(target_yaw)
                target_velocity_3d = np.array([
                    target_speed * math.cos(target_yaw_rad),
                    target_speed * math.sin(target_yaw_rad),
                    -self._target_climb_rate,
                ])
            target_acceleration_3d = np.array([
                -math.radians(self._target_yaw_rate) * target_velocity_3d[1],
                math.radians(self._target_yaw_rate) * target_velocity_3d[0],
                -self._target_vertical_accel,
            ])
            pitch_rad = math.radians(pitch)
            own_forward_3d = np.array([
                math.cos(pitch_rad) * math.cos(yaw_rad),
                math.cos(pitch_rad) * math.sin(yaw_rad),
                -math.sin(pitch_rad),
            ])
            formula_vpp_result = compute_vpp_guidance(
                own_position, target_position, own_velocity_3d, target_velocity_3d,
                ata_deg=ata, threat_ata_deg=threat_ata, distance_m=distance,
                previous_blend=self._formula_vpp_blend, dt=dt,
                transition_rate_per_s=cfg.formula_vpp_transition_rate_per_s,
                lag_distance_m=cfg.formula_vpp_lag_distance_m,
                mutual_lateral_m=cfg.formula_vpp_mutual_lateral_m,
                projectile_speed_mps=cfg.formula_vpp_projectile_speed_mps,
                pursuit_gain=cfg.formula_vpp_pursuit_gain,
                navigation_constant=cfg.formula_vpp_navigation_constant,
                turn_rate_limit_degps=cfg.formula_vpp_turn_rate_limit_degps,
                gamma_limit_deg=cfg.formula_vpp_gamma_limit_deg,
                target_yaw_rate_degps=self._target_yaw_rate,
                turn_circle_enabled=cfg.formula_vpp_turn_circle_enabled,
                turn_circle_horizon_s=cfg.formula_vpp_turn_circle_horizon_s,
                mode_override=("lead" if formula_vpp_recommit_active else None),
                target_acceleration=target_acceleration_3d,
                own_forward=own_forward_3d,
                ballistic_tof_enabled=cfg.formula_ballistic_tof_enabled,
                apg_enabled=cfg.formula_apg_enabled,
                apg_gain=cfg.formula_apg_gain,
            )
            self._formula_vpp_blend = formula_vpp_result.blend
            aim_target[StateIndex.N:StateIndex.D + 1] = formula_vpp_result.point
        aim_target[StateIndex.D] = (
            aim_target[StateIndex.D] if formula_vpp_active else target[StateIndex.D]
        )
        aim_target[StateIndex.ALT] = target[StateIndex.ALT]
        aim_az, aim_el = self.geometry._get_los_angle(own, aim_target)
        aim_az, aim_el = float(aim_az), float(aim_el)

        aim_delta_n = float(aim_target[StateIndex.N] - own_position[0])
        aim_delta_e = float(aim_target[StateIndex.E] - own_position[1])
        aim_world_bearing = math.degrees(math.atan2(aim_delta_e, aim_delta_n))
        aim_course_error = wrap180(aim_world_bearing - own_yaw)
        guidance_az = aim_course_error if cfg.use_horizontal_course_guidance else aim_az
        guidance_los_rate = (
            self._horizontal_los_rate
            if cfg.use_horizontal_course_guidance
            else self._los_rate
        )

        turn_match_enabled = cfg.turn_match_enter_ata_deg > 0.0
        if turn_match_enabled:
            if self._turn_match_latched:
                if (
                    ata >= cfg.turn_match_exit_ata_deg
                    or distance >= cfg.turn_match_exit_range_m
                    or threat_ata < 0.5 * cfg.turn_match_min_threat_ata_deg
                    or abs(guidance_az) >= cfg.turn_match_exit_course_error_deg
                ):
                    self._turn_match_latched = False
                    self._turn_match_candidate_s = 0.0
            entry_candidate = (
                ata <= cfg.turn_match_enter_ata_deg
                and distance <= cfg.turn_match_enter_range_m
                and threat_ata >= cfg.turn_match_min_threat_ata_deg
                and abs(guidance_az) <= cfg.turn_match_enter_course_error_deg
                and closure <= cfg.turn_match_enter_max_closure_mps
                and (
                    not cfg.turn_match_require_stable_prediction
                    or prediction_stable
                )
            )
            if not self._turn_match_latched:
                if entry_candidate:
                    self._turn_match_candidate_s += dt or 0.0
                    if self._turn_match_candidate_s >= cfg.turn_match_entry_hold_s:
                        self._turn_match_latched = True
                else:
                    self._turn_match_candidate_s = 0.0
        else:
            self._turn_match_latched = False
            self._turn_match_candidate_s = 0.0
        turn_match_active = self._turn_match_latched
        if cfg.integrated_manager_enabled and state == "defensive":
            self._turn_match_latched = False
            self._turn_match_candidate_s = 0.0
            turn_match_active = False

        energy_deficit = (
            own_speed + cfg.integrated_energy_speed_margin_mps < target_speed
            or own_speed < cfg.minimum_energy_speed_mps
        )
        overshoot_control_active = (
            cfg.integrated_manager_enabled
            and (
                state != "defensive"
                or cfg.integrated_overshoot_allow_defensive
            )
            and ata <= cfg.integrated_overshoot_ata_deg
            and distance <= cfg.integrated_overshoot_range_m
            and closure >= cfg.integrated_overshoot_closure_mps
        )
        vertical_follow_active = (
            cfg.integrated_manager_enabled
            and state != "defensive"
            and abs(world_elevation) >= cfg.integrated_vertical_elevation_deg
            and distance <= cfg.integrated_vertical_range_m
        )
        if state == "defensive":
            engagement_mode = "defensive"
        elif terminal_track_active:
            engagement_mode = "weapons_track"
        elif overshoot_control_active:
            engagement_mode = "overshoot_control"
        elif (
            cfg.rear_rate_match_enabled
            and state == "reacquire"
            and ata >= cfg.integrated_postmerge_ata_deg
            and prediction_stable
            and abs(self._target_yaw_rate)
            >= cfg.rear_rate_match_min_target_rate_degps
        ):
            engagement_mode = "postmerge_rate"
        elif vertical_follow_active:
            engagement_mode = "vertical_follow"
        elif energy_deficit:
            engagement_mode = "energy_recovery"
        elif turn_match_active:
            engagement_mode = "turn_match"
        else:
            engagement_mode = state

        desired_sign = signed_unit(guidance_az, self._turn_sign)
        rear_rate_match_candidate = (
            cfg.rear_rate_match_enabled
            and state == "reacquire"
            and ata >= cfg.rear_rate_match_ata_deg
            and prediction_stable
            and abs(self._target_yaw_rate)
            >= cfg.rear_rate_match_min_target_rate_degps
        )
        if rear_rate_match_candidate:
            desired_sign = signed_unit(self._target_yaw_rate, self._turn_sign)
        if state == "break_turn":
            if elapsed <= cfg.merge_initial_s + 0.25:
                self._turn_sign = desired_sign
        elif (
            abs(guidance_az) < cfg.guidance_rear_commit_deg
            and desired_sign != self._turn_sign
            and now - self._last_sign_change >= cfg.sign_min_hold_s
        ):
            # Outside break-turn, predicted aimpoint is allowed to recommit.
            self._turn_sign = desired_sign
            self._last_sign_change = now

        if rear_rate_match_candidate and desired_sign != self._turn_sign:
            if now - self._last_sign_change >= cfg.sign_min_hold_s:
                self._turn_sign = desired_sign
                self._last_sign_change = now

        rate_limit = self._turn_rate_limit(state)
        rear_committed = abs(guidance_az) >= cfg.guidance_rear_commit_deg
        if (state in ("break_turn", "defensive") and ata > 60.0) or rear_committed:
            desired_turn_rate = self._turn_sign * rate_limit
        else:
            desired_turn_rate = (
                cfg.az_to_turn_rate_gain * guidance_az
                + cfg.los_rate_gain * guidance_los_rate
            )
            desired_turn_rate = float(np.clip(desired_turn_rate, -rate_limit, rate_limit))
        rear_rate_match_active = rear_rate_match_candidate
        if rear_rate_match_active:
            match_sign = signed_unit(self._target_yaw_rate, self._turn_sign)
            course_margin = (
                cfg.rear_rate_match_error_gain
                * min(abs(ata), cfg.rear_rate_match_max_error_deg)
            )
            desired_turn_rate = float(np.clip(
                cfg.rear_rate_match_target_gain * self._target_yaw_rate
                + match_sign * course_margin,
                -cfg.rear_rate_match_limit_degps,
                cfg.rear_rate_match_limit_degps,
            ))
        if overshoot_control_active:
            # Stop cutting across the target's circle.  Match most of its
            # measured rate and retain only a small course-error correction;
            # the independent closure-throttle loop simultaneously manages
            # range without undoing the horizontal solution.
            desired_turn_rate = float(np.clip(
                cfg.integrated_overshoot_rate_gain * self._target_yaw_rate
                + cfg.integrated_overshoot_course_gain * guidance_az,
                -cfg.integrated_overshoot_rate_limit_degps,
                cfg.integrated_overshoot_rate_limit_degps,
            ))
        if attack_conversion_active:
            # Once our nose is aligned and the bandit's nose is safely away,
            # preserve its turn circle instead of cutting toward a lag proxy.
            # Live W89 reversed bank near R=3.1km/ATA=5deg and surrendered
            # the solution before entering gun range.
            desired_turn_rate = float(np.clip(
                cfg.attack_conversion_target_rate_gain * self._target_yaw_rate
                + cfg.attack_conversion_course_gain * guidance_az
                + cfg.attack_conversion_los_rate_gain * guidance_los_rate,
                -cfg.attack_conversion_rate_limit_degps,
                cfg.attack_conversion_rate_limit_degps,
            ))
        if formula_vpp_result is not None and not attack_conversion_active:
            desired_turn_rate = formula_vpp_result.turn_rate_degps
        if formula_zem_defense_active:
            desired_turn_rate = (
                formula_zem_escape_sign * cfg.formula_zem_turn_rate_degps
            )
        if (
            cfg.acquisition_min_turn_rate_degps > 0.0
            and ata >= cfg.acquisition_min_turn_rate_ata_deg
            and abs(desired_turn_rate) < cfg.acquisition_min_turn_rate_degps
        ):
            desired_turn_rate = (
                signed_unit(desired_turn_rate, self._turn_sign)
                * cfg.acquisition_min_turn_rate_degps
            )
        if turn_match_active:
            desired_turn_rate = float(np.clip(
                cfg.turn_match_yaw_rate_gain * self._target_yaw_rate
                + cfg.turn_match_course_gain * guidance_az,
                -cfg.turn_match_rate_limit_degps,
                cfg.turn_match_rate_limit_degps,
            ))
            desired_turn_rate += cfg.turn_match_los_rate_gain * guidance_los_rate
            desired_turn_rate = float(np.clip(
                desired_turn_rate,
                -cfg.turn_match_rate_limit_degps,
                cfg.turn_match_rate_limit_degps,
            ))
            # Do not roll away from a still-visible horizontal error merely
            # because target yaw-rate feed-forward has the opposite sign.
            # Once the error is nearly zero, release the guard so matching the
            # target circle can maintain the gun solution.
            if (
                abs(guidance_az) >= cfg.turn_match_sign_guard_error_deg
                and desired_turn_rate * guidance_az < 0.0
            ):
                desired_turn_rate = (
                    signed_unit(guidance_az)
                    * min(
                        cfg.turn_match_sign_guard_rate_degps,
                        max(0.5, cfg.turn_match_course_gain * abs(guidance_az)),
                    )
                )
        if terminal_track_active:
            desired_turn_rate = float(np.clip(
                cfg.terminal_track_yaw_rate_gain * self._target_yaw_rate
                + cfg.terminal_track_course_gain * guidance_az
                + cfg.terminal_track_los_rate_gain * guidance_los_rate,
                -cfg.terminal_track_rate_limit_degps,
                cfg.terminal_track_rate_limit_degps,
            ))

        predictive_guidance_active = (
            cfg.predictive_guidance_enabled
            and ata >= cfg.predictive_guidance_min_ata_deg
            and ata <= cfg.predictive_guidance_max_ata_deg
            and distance <= cfg.predictive_guidance_max_range_m
            and threat_ata >= cfg.predictive_guidance_min_threat_ata_deg
            and state not in ("merge", "weapons_track")
            and not formula_vpp_active
            and not formula_zem_defense_active
            and dt is not None
        )
        predictive_horizon = 0.0
        predictive_score = 0.0
        predictive_rule_score = 0.0
        predictive_score_gain = 0.0
        predictive_future_error = 0.0
        predictive_future_threat_error = 180.0
        predictive_worst_target_rate = 0.0
        predictive_sign_guard_active = False
        predictive_override_active = False
        rule_desired_turn_rate = desired_turn_rate
        if predictive_guidance_active:
            # Time-to-contact drives the horizon but is bounded so noisy
            # instantaneous closure cannot cause a discontinuous far-future
            # plan. Target and ownship are then propagated under reachable
            # constant-turn hypotheses.
            closing_scale = max(abs(closure), 0.35 * own_speed, 1.0)
            predictive_horizon = float(np.clip(
                distance / closing_scale,
                cfg.predictive_horizon_min_s,
                cfg.predictive_horizon_max_s,
            ))
            target_horizontal_speed = float(np.linalg.norm(horizontal_velocity[:2]))
            if target_horizontal_speed < 20.0:
                target_horizontal_speed = target_speed
            target_future, _ = self._predict_horizontal_state(
                target_position[:2],
                target_yaw,
                target_horizontal_speed,
                self._target_yaw_rate,
                predictive_horizon,
            )
            limit = cfg.predictive_turn_limit_degps
            if cfg.predictive_live_turn_envelope:
                # Conservative p75/p90 envelope identified from 70 live 91deg
                # runs. Unlike W74's fixed 15deg/s assumption, this remains
                # reachable after the aircraft has spent energy in the turn.
                if own_speed < 180.0:
                    limit = min(limit, 10.5)
                elif own_speed < 210.0:
                    limit = min(limit, 13.0)
                elif own_speed < 240.0:
                    limit = min(limit, 14.5)
                else:
                    limit = min(limit, 17.0)
            candidates = list(np.linspace(
                -limit,
                limit,
                max(3, int(cfg.predictive_candidate_count)),
            ))
            candidates.extend((
                float(np.clip(rule_desired_turn_rate, -limit, limit)),
                float(np.clip(self._own_yaw_rate, -limit, limit)),
                float(np.clip(self._target_yaw_rate, -limit, limit)),
            ))
            best: tuple[float, float, float, float, float] | None = None
            rule_score: float | None = None
            for candidate in candidates:
                delay = min(cfg.predictive_response_delay_s, predictive_horizon)
                if cfg.predictive_adversarial_enabled:
                    # Treat this as a reachable envelope, not a lower bound.
                    # Live differentiation occasionally produces >100deg/s
                    # spikes although the target's p95 sustained rate is
                    # about 18deg/s.
                    target_limit = cfg.predictive_target_turn_limit_degps
                    target_candidates = (
                        -target_limit,
                        0.0,
                        target_limit,
                        float(np.clip(
                            self._target_yaw_rate, -target_limit, target_limit
                        )),
                    )
                else:
                    target_candidates = (self._target_yaw_rate,)

                worst: tuple[float, float, float, float] | None = None
                for target_candidate in target_candidates:
                    trajectory_score = 0.0
                    final_error = 0.0
                    final_threat_error = 180.0
                    weight_sum = 0.0
                    # Include the midpoint so a manoeuvre cannot look good
                    # only at its endpoint while first surrendering geometry.
                    for fraction, sample_weight in (
                        (0.5, cfg.predictive_midpoint_weight),
                        (1.0, 1.0),
                    ):
                        sample_horizon = predictive_horizon * fraction
                        sample_delay = min(delay, sample_horizon)
                        own_mid, own_mid_heading = self._predict_horizontal_state(
                            own_position[:2], own_yaw, own_speed,
                            self._own_yaw_rate, sample_delay,
                        )
                        own_future, own_future_heading = self._predict_horizontal_state(
                            own_mid, own_mid_heading, own_speed, float(candidate),
                            sample_horizon - sample_delay,
                        )
                        target_future, target_future_heading = self._predict_horizontal_state(
                            target_position[:2], target_yaw,
                            target_horizontal_speed, float(target_candidate),
                            sample_horizon,
                        )
                        future_relative = target_future - own_future
                        future_range = float(np.linalg.norm(future_relative))
                        future_bearing = math.degrees(math.atan2(
                            future_relative[1], future_relative[0]
                        ))
                        future_error = wrap180(
                            future_bearing - own_future_heading
                        )
                        threat_bearing = wrap180(future_bearing + 180.0)
                        threat_error = wrap180(
                            threat_bearing - target_future_heading
                        )
                        # Coupled geometry: favorable range cannot compensate
                        # for a poor nose position. Penalize threat only inside
                        # its predicted gun cone to avoid permanent evasion.
                        sample_score = abs(future_error)
                        sample_score += cfg.predictive_threat_weight * max(
                            0.0,
                            cfg.predictive_threat_cone_deg - abs(threat_error),
                        )
                        if cfg.predictive_rear_geometry_weight > 0.0:
                            rear_range_scale = float(np.clip(
                                (cfg.predictive_rear_geometry_range_m - future_range)
                                / max(cfg.predictive_rear_geometry_range_m, 1.0),
                                0.0,
                                1.0,
                            ))
                            sample_score += (
                                cfg.predictive_rear_geometry_weight
                                * rear_range_scale
                                * (180.0 - abs(threat_error))
                            )
                        sample_score += cfg.predictive_too_close_weight * max(
                            0.0,
                            cfg.predictive_min_separation_m - future_range,
                        )
                        sample_score += cfg.predictive_far_weight * max(
                            0.0,
                            future_range - cfg.predictive_max_separation_m,
                        )
                        trajectory_score += sample_weight * sample_score
                        weight_sum += sample_weight
                        if fraction == 1.0:
                            final_error = future_error
                            final_threat_error = threat_error
                    score = trajectory_score / max(weight_sum, 1e-6)
                    score += cfg.predictive_rate_change_weight * abs(
                        float(candidate) - self._own_yaw_rate
                    )
                    if own_speed < cfg.predictive_low_energy_speed_mps:
                        score += cfg.predictive_low_energy_turn_weight * abs(
                            float(candidate)
                        )
                    if worst is None or score > worst[0]:
                        worst = (
                            score,
                            final_error,
                            final_threat_error,
                            float(target_candidate),
                        )
                if worst is None:
                    continue
                score, future_error, future_threat_error, worst_target_rate = worst
                if best is None or score < best[0]:
                    best = (
                        score,
                        float(candidate),
                        future_error,
                        future_threat_error,
                        worst_target_rate,
                    )
                if abs(float(candidate) - float(np.clip(
                    rule_desired_turn_rate, -limit, limit
                ))) < 1e-6:
                    rule_score = score
            if best is not None:
                (
                    predictive_score,
                    selected_turn_rate,
                    predictive_future_error,
                    predictive_future_threat_error,
                    predictive_worst_target_rate,
                ) = best
                predictive_rule_score = (
                    predictive_score if rule_score is None else rule_score
                )
                predictive_score_gain = predictive_rule_score - predictive_score
                predictive_override_active = (
                    predictive_score_gain
                    >= cfg.predictive_override_min_score_gain
                    and abs(selected_turn_rate - rule_desired_turn_rate)
                    >= cfg.predictive_override_min_rate_delta_degps
                )
                predictive_sign_guard_active = (
                    cfg.predictive_sign_guard_error_deg > 0.0
                    and abs(guidance_az)
                    >= cfg.predictive_sign_guard_error_deg
                    and selected_turn_rate * guidance_az < 0.0
                )
                if predictive_sign_guard_active:
                    # A worst-case game score may prefer continued evasion.
                    # It must not override the direct geometric turn direction
                    # while the target is well away from the nose.
                    predictive_override_active = False
                if (
                    cfg.predictive_rule_sign_guard_enabled
                    and abs(rule_desired_turn_rate) >= 2.0
                    and selected_turn_rate * rule_desired_turn_rate < 0.0
                ):
                    # The horizon model is an advisory correction. It may not
                    # reverse a direct geometric pursuit command, which was
                    # the W104-live constant-right-turn failure.
                    predictive_override_active = False
                if predictive_override_active:
                    if cfg.predictive_max_rule_delta_degps > 0.0:
                        selected_turn_rate = float(np.clip(
                            selected_turn_rate,
                            rule_desired_turn_rate - cfg.predictive_max_rule_delta_degps,
                            rule_desired_turn_rate + cfg.predictive_max_rule_delta_degps,
                        ))
                    if self._predictive_was_active:
                        max_rate_step = cfg.predictive_turn_slew_degps2 * dt
                        self._predictive_turn_rate += float(np.clip(
                            selected_turn_rate - self._predictive_turn_rate,
                            -max_rate_step,
                            max_rate_step,
                        ))
                    else:
                        self._predictive_turn_rate = selected_turn_rate
                    desired_turn_rate = self._predictive_turn_rate
                else:
                    desired_turn_rate = rule_desired_turn_rate
                    self._predictive_turn_rate = rule_desired_turn_rate
            self._predictive_was_active = predictive_override_active
        else:
            self._predictive_was_active = False
            self._predictive_turn_rate = desired_turn_rate

        planner3d_active = (
            cfg.planner3d_enabled
            and dt is not None
            and not terminal_track_active
            and state != "merge"
        )
        planner3d_replanned = False
        planner_horizon = 0.0
        if planner3d_active:
            own_gamma_now = math.degrees(math.asin(np.clip(
                vertical_speed / max(own_speed, 1.0), -1.0, 1.0
            )))
            target_gamma_now = math.degrees(math.asin(np.clip(
                self._target_climb_rate / max(target_speed, 1.0), -1.0, 1.0
            )))
            own_position_up = np.array(
                [own_position[0], own_position[1], altitude], dtype=np.float64
            )
            target_position_up = np.array(
                [target_position[0], target_position[1], target_altitude],
                dtype=np.float64,
            )
            live_limit = cfg.planner3d_turn_limit_degps
            if own_speed < 180.0:
                live_limit = min(live_limit, 10.5)
            elif own_speed < 210.0:
                live_limit = min(live_limit, 13.0)
            elif own_speed < 240.0:
                live_limit = min(live_limit, 14.5)
            turn_candidates = list((
                -live_limit,
                -0.55 * live_limit,
                0.0,
                0.55 * live_limit,
                live_limit,
            ))
            turn_candidates.extend((
                float(np.clip(rule_desired_turn_rate, -live_limit, live_limit)),
                float(np.clip(self._own_yaw_rate, -live_limit, live_limit)),
            ))
            gamma_limit_3d = cfg.planner3d_gamma_deg
            if altitude <= cfg.planner3d_floor_m + 700.0:
                gamma_candidates = (0.0, gamma_limit_3d)
            else:
                gamma_candidates = (
                    -gamma_limit_3d,
                    0.0,
                    gamma_limit_3d,
                    float(np.clip(own_gamma_now, -gamma_limit_3d, gamma_limit_3d)),
                )
            target_turn_limit = cfg.planner3d_target_turn_limit_degps
            if cfg.planner3d_reachable_target_envelope:
                target_turn_candidates = tuple(float(np.clip(
                    self._target_yaw_rate + delta,
                    -target_turn_limit,
                    target_turn_limit,
                )) for delta in (
                    -cfg.planner3d_target_turn_delta_degps,
                    0.0,
                    cfg.planner3d_target_turn_delta_degps,
                ))
            else:
                target_turn_candidates = (
                    -target_turn_limit,
                    0.0,
                    target_turn_limit,
                    float(np.clip(
                        self._target_yaw_rate, -target_turn_limit, target_turn_limit
                    )),
                )
            target_gamma_limit = cfg.planner3d_target_gamma_deg
            if cfg.planner3d_reachable_target_envelope:
                target_gamma_candidates = tuple(float(np.clip(
                    target_gamma_now + delta,
                    -target_gamma_limit,
                    target_gamma_limit,
                )) for delta in (
                    -cfg.planner3d_target_gamma_delta_deg,
                    0.0,
                    cfg.planner3d_target_gamma_delta_deg,
                ))
            else:
                target_gamma_candidates = (
                    -target_gamma_limit,
                    0.0,
                    target_gamma_limit,
                    float(np.clip(
                        target_gamma_now, -target_gamma_limit, target_gamma_limit
                    )),
                )
            initial_specific_energy = 0.5 * own_speed * own_speed + G * altitude
            hard_threat_now = (
                threat_ata <= cfg.defensive_threat_ata_deg
                and distance <= cfg.defensive_range_m
            )
            if cfg.planner3d_research_scoring:
                angle_pressure = 1.0 / (
                    1.0 + math.exp((threat_ata - 25.0) / 7.0)
                )
                range_pressure = 1.0 / (
                    1.0 + math.exp((distance - 1900.0) / 300.0)
                )
                geometry_pressure = float(np.clip(
                    angle_pressure * range_pressure, 0.0, 1.0
                ))
                role_alpha = 1.0 - math.exp(
                    -(dt or 0.0) / max(cfg.planner3d_role_tau_s, 0.05)
                )
                self._planner3d_threat_pressure += role_alpha * (
                    geometry_pressure - self._planner3d_threat_pressure
                )
            else:
                self._planner3d_threat_pressure = 1.0 if hard_threat_now else 0.0
            planner_horizon = cfg.planner3d_horizon_s
            if cfg.planner3d_adaptive_horizon:
                closing_scale_3d = max(closure, 0.35 * own_speed, 1.0)
                time_to_contact_3d = distance / closing_scale_3d
                planner_horizon = float(np.clip(
                    1.20 * time_to_contact_3d,
                    cfg.planner3d_horizon_min_s,
                    cfg.planner3d_horizon_s,
                ))
            best_3d: tuple[float, float, float, float, float, float, float, float] | None = None
            for candidate_turn in turn_candidates:
                for candidate_gamma in gamma_candidates:
                    worst_3d: tuple[float, float, float, float, float, float] | None = None
                    for target_turn in target_turn_candidates:
                        for target_gamma in target_gamma_candidates:
                            trajectory_score = 0.0
                            final_ata = 180.0
                            final_threat = 180.0
                            final_range = distance
                            weight_sum = 0.0
                            if cfg.planner3d_research_scoring:
                                sigma = max(
                                    0.05,
                                    cfg.planner3d_time_weight_beta * planner_horizon,
                                )
                                trajectory_samples = tuple(
                                    (
                                        fraction,
                                        math.exp(
                                            -((fraction * planner_horizon) ** 2)
                                            / (2.0 * sigma * sigma)
                                        ),
                                    )
                                    for fraction in (0.20, 0.55, 1.0)
                                )
                            else:
                                trajectory_samples = (
                                    (
                                        (0.25, 0.35),
                                        (0.55, cfg.planner3d_midpoint_weight),
                                        (1.0, 1.0),
                                    )
                                    if cfg.planner3d_adaptive_horizon
                                    else (
                                        (0.5, cfg.planner3d_midpoint_weight),
                                        (1.0, 1.0),
                                    )
                                )
                            for fraction, sample_weight in trajectory_samples:
                                sample_horizon = planner_horizon * fraction
                                own_future, own_heading_future, own_speed_future = (
                                    self._predict_3d_state(
                                        own_position_up, own_yaw, own_speed,
                                        candidate_turn, candidate_gamma,
                                        sample_horizon, 2.5,
                                        cfg.planner3d_turn_drag_coeff
                                        if cfg.planner3d_research_scoring else 0.0,
                                    )
                                )
                                target_future_3d, target_heading_future, target_speed_future = (
                                    self._predict_3d_state(
                                        target_position_up, target_yaw, target_speed,
                                        target_turn, target_gamma,
                                        sample_horizon, 1.5,
                                        cfg.planner3d_turn_drag_coeff
                                        if cfg.planner3d_research_scoring else 0.0,
                                    )
                                )
                                future_relative_3d = target_future_3d - own_future
                                future_range_3d = float(np.linalg.norm(future_relative_3d))
                                own_direction = self._velocity_direction(
                                    own_heading_future, candidate_gamma
                                )
                                target_direction = self._velocity_direction(
                                    target_heading_future, target_gamma
                                )
                                future_ata_3d = self._angle_to_los(
                                    own_direction, future_relative_3d
                                )
                                future_threat_3d = self._angle_to_los(
                                    target_direction, -future_relative_3d
                                )
                                range_scale = float(np.clip(
                                    (cfg.planner3d_max_separation_m - future_range_3d)
                                    / max(cfg.planner3d_max_separation_m, 1.0),
                                    0.0, 1.0,
                                ))
                                if cfg.planner3d_research_scoring:
                                    # Coupled distance-angle quality.  Range is
                                    # multiplicative, so a good angle far
                                    # outside the WEZ cannot cancel a bad
                                    # distance (and vice versa), matching the
                                    # coupled-reward principle rather than an
                                    # arbitrary weighted sum of ATA and range.
                                    range_quality = math.exp(
                                        -0.5 * (
                                            (future_range_3d - cfg.planner3d_attack_range_m)
                                            / max(cfg.planner3d_attack_range_width_m, 1.0)
                                        ) ** 2
                                    )
                                    own_nose = max(
                                        0.0,
                                        math.cos(math.radians(future_ata_3d) / 2.0),
                                    ) ** 4
                                    target_nose = max(
                                        0.0,
                                        math.cos(math.radians(future_threat_3d) / 2.0),
                                    ) ** 4
                                    own_rear_control = math.sin(
                                        math.radians(future_threat_3d) / 2.0
                                    ) ** 2
                                    target_rear_control = math.sin(
                                        math.radians(future_ata_3d) / 2.0
                                    ) ** 2
                                    own_attack_quality = (
                                        range_quality * own_nose * own_rear_control
                                    )
                                    enemy_attack_quality = (
                                        range_quality * target_nose * target_rear_control
                                    )
                                    pressure = self._planner3d_threat_pressure
                                    sample_score = 120.0 * (
                                        (1.0 - pressure) * (1.0 - own_attack_quality)
                                        + pressure * enemy_attack_quality
                                    )
                                    # Retain a security term in offensive
                                    # states; otherwise a tiny filtered threat
                                    # probability can permit a mutual kill.
                                    sample_score += 35.0 * enemy_attack_quality

                                    # Horizontal VPP equivalent: when both
                                    # aircraft are turning the same way, align
                                    # their instantaneous turn-circle centres
                                    # instead of merely chasing target position.
                                    if (
                                        abs(candidate_turn) >= 1.0
                                        and abs(target_turn) >= 1.0
                                        and candidate_turn * target_turn > 0.0
                                    ):
                                        own_left = np.array([
                                            -math.sin(math.radians(own_heading_future)),
                                            math.cos(math.radians(own_heading_future)),
                                        ])
                                        target_left = np.array([
                                            -math.sin(math.radians(target_heading_future)),
                                            math.cos(math.radians(target_heading_future)),
                                        ])
                                        own_radius = (
                                            own_speed_future
                                            * math.cos(math.radians(candidate_gamma))
                                            / max(abs(math.radians(candidate_turn)), 1e-3)
                                        )
                                        target_radius = (
                                            target_speed_future
                                            * math.cos(math.radians(target_gamma))
                                            / max(abs(math.radians(target_turn)), 1e-3)
                                        )
                                        own_center = (
                                            own_future[:2]
                                            + signed_unit(candidate_turn) * own_left * own_radius
                                        )
                                        target_center = (
                                            target_future_3d[:2]
                                            + signed_unit(target_turn) * target_left * target_radius
                                        )
                                        center_error = min(
                                            3000.0,
                                            float(np.linalg.norm(target_center - own_center)),
                                        )
                                        sample_score += (
                                            cfg.planner3d_turn_center_weight
                                            * (1.0 - pressure)
                                            * center_error
                                        )
                                else:
                                    threat_weight = 4.0 if hard_threat_now else 1.8
                                    attack_weight = 0.70 if hard_threat_now else 1.0
                                    sample_score = attack_weight * future_ata_3d
                                    sample_score += threat_weight * max(
                                        0.0,
                                        cfg.planner3d_threat_cone_deg - future_threat_3d,
                                    )
                                    sample_score += (
                                        0.14 * range_scale * (180.0 - future_threat_3d)
                                    )
                                sample_score += 0.10 * max(
                                    0.0,
                                    cfg.planner3d_min_separation_m - future_range_3d,
                                )
                                sample_score += 0.008 * max(
                                    0.0,
                                    future_range_3d - cfg.planner3d_max_separation_m,
                                )
                                sample_score += 5.0 * max(
                                    0.0,
                                    cfg.planner3d_min_speed_mps - own_speed_future,
                                )
                                if own_future[2] < cfg.planner3d_floor_m:
                                    sample_score += (
                                        20.0 * (cfg.planner3d_floor_m - own_future[2])
                                        + 5000.0
                                    )
                                future_specific_energy = (
                                    0.5 * own_speed_future * own_speed_future
                                    + G * own_future[2]
                                )
                                sample_score += cfg.planner3d_energy_weight * max(
                                    0.0,
                                    initial_specific_energy - future_specific_energy,
                                )
                                trajectory_score += sample_weight * sample_score
                                weight_sum += sample_weight
                                if fraction == 1.0:
                                    final_ata = future_ata_3d
                                    final_threat = future_threat_3d
                                    final_range = future_range_3d
                            score_3d = trajectory_score / max(weight_sum, 1e-6)
                            score_3d += cfg.planner3d_rate_change_weight * abs(
                                candidate_turn - self._own_yaw_rate
                            )
                            score_3d += cfg.planner3d_gamma_change_weight * abs(
                                candidate_gamma - own_gamma_now
                            )
                            if worst_3d is None or score_3d > worst_3d[0]:
                                worst_3d = (
                                    score_3d, final_ata, final_threat,
                                    final_range, target_turn, target_gamma,
                                )
                    if worst_3d is None:
                        continue
                    if best_3d is None or worst_3d[0] < best_3d[0]:
                        best_3d = (
                            worst_3d[0], candidate_turn, candidate_gamma,
                            worst_3d[1], worst_3d[2], worst_3d[3],
                            worst_3d[4], worst_3d[5],
                        )
            emergency_replan_ready = (
                cfg.planner3d_emergency_replan
                and hard_threat_now
                and now >= (
                    self._planner3d_until
                    - cfg.planner3d_manoeuvre_hold_s
                    + cfg.planner3d_emergency_hold_s
                )
            )
            if best_3d is not None and (
                now >= self._planner3d_until or emergency_replan_ready
            ):
                (
                    self._planner3d_score,
                    self._planner3d_turn_rate,
                    self._planner3d_gamma,
                    self._planner3d_future_ata,
                    self._planner3d_future_threat_ata,
                    self._planner3d_future_range,
                    self._planner3d_worst_target_turn,
                    self._planner3d_worst_target_gamma,
                ) = best_3d
                turn_label = (
                    "left" if self._planner3d_turn_rate < -0.5
                    else "right" if self._planner3d_turn_rate > 0.5
                    else "straight"
                )
                gamma_label = (
                    "climb" if self._planner3d_gamma > 1.0
                    else "dive" if self._planner3d_gamma < -1.0
                    else "level"
                )
                self._planner3d_name = f"{gamma_label}_{turn_label}"
                self._planner3d_until = now + cfg.planner3d_manoeuvre_hold_s
                planner3d_replanned = True
            desired_turn_rate = self._planner3d_turn_rate
            range_blend_3d = float(np.clip(
                (distance - 1200.0) / 1800.0, 0.0, 1.0
            ))
            geometry_target_speed = (
                cfg.planner3d_corner_speed_mps
                + range_blend_3d
                * (cfg.planner3d_chase_speed_mps - cfg.planner3d_corner_speed_mps)
            )
            if ata > 35.0 or distance > 1800.0:
                # Do not command a fixed corner speed while the target is
                # escaping faster.  Match the observed bandit energy with a
                # bounded margin; once aligned, return to the smaller-radius
                # corner-speed schedule to avoid another overshoot.
                geometry_target_speed = max(
                    geometry_target_speed,
                    min(target_speed + 10.0, 265.0),
                )
            self._planner3d_target_speed = geometry_target_speed
        else:
            self._planner3d_name = "inactive"

        rate_deficit_overbank_active = (
            cfg.rate_deficit_overbank_enabled
            and state != "defensive"
            and ata >= cfg.rate_deficit_min_ata_deg
            and altitude >= cfg.rate_deficit_min_altitude_m
            and abs(self._target_yaw_rate)
            >= cfg.rate_deficit_min_target_rate_degps
            and abs(self._target_yaw_rate)
            >= abs(self._own_yaw_rate) + cfg.rate_deficit_margin_degps
        )
        effective_max_bank = (
            max(cfg.max_bank_deg, cfg.rate_deficit_overbank_deg)
            if rate_deficit_overbank_active
            else cfg.max_bank_deg
        )

        # W108 3-D outer loop. Positions/target-velocity are NED (D positive
        # downward); vertical_speed is positive upward, hence own D-rate is
        # -vertical_speed.  normal_error is the LOS component perpendicular to
        # current velocity, while los_dot supplies damping/lead from the
        # measured relative motion.  Both axes therefore react to the same
        # current target motion instead of independent azimuth/altitude rules.
        lift_vector_active = (
            cfg.lift_vector_guidance_enabled
            and elapsed >= cfg.lift_vector_activation_delay_s
            and cfg.lift_vector_min_range_m <= distance
            <= cfg.lift_vector_max_range_m
            and ata <= cfg.lift_vector_max_ata_deg
            and dt is not None
        )
        lift_vector_target_bank = 0.0
        lift_vector_target_gamma = 0.0
        lift_vector_los_rate = 0.0
        lift_vector_accel = 0.0
        if lift_vector_active:
            own_yaw_rad_lv = math.radians(own_yaw)
            own_hspeed_lv = math.sqrt(max(
                own_speed * own_speed - vertical_speed * vertical_speed,
                1.0,
            ))
            own_velocity_lv = np.array([
                own_hspeed_lv * math.cos(own_yaw_rad_lv),
                own_hspeed_lv * math.sin(own_yaw_rad_lv),
                -vertical_speed,
            ])
            target_velocity_lv = self._target_velocity.copy()
            if float(np.linalg.norm(target_velocity_lv)) < 20.0:
                target_yaw_rad_lv = math.radians(target_yaw)
                target_velocity_lv = np.array([
                    target_speed * math.cos(target_yaw_rad_lv),
                    target_speed * math.sin(target_yaw_rad_lv),
                    -self._target_climb_rate,
                ])
            range_lv = max(float(np.linalg.norm(relative_position)), 1.0)
            los_unit_lv = relative_position / range_lv
            relative_velocity_lv = target_velocity_lv - own_velocity_lv
            los_dot_lv = (
                relative_velocity_lv
                - los_unit_lv * float(np.dot(los_unit_lv, relative_velocity_lv))
            ) / range_lv
            own_velocity_norm_lv = max(
                float(np.linalg.norm(own_velocity_lv)), 1.0
            )
            own_forward_lv = own_velocity_lv / own_velocity_norm_lv
            normal_error_lv = (
                los_unit_lv
                - own_forward_lv * float(np.dot(los_unit_lv, own_forward_lv))
            )
            los_rate_normal_lv = (
                los_dot_lv
                - own_forward_lv * float(np.dot(los_dot_lv, own_forward_lv))
            )
            acceleration_lv = (
                cfg.lift_vector_los_kp_g * G * normal_error_lv
                + cfg.lift_vector_los_rate_gain
                * own_speed * los_rate_normal_lv
            )
            accel_norm_lv = float(np.linalg.norm(acceleration_lv))
            if accel_norm_lv > cfg.lift_vector_max_accel_mps2:
                acceleration_lv *= (
                    cfg.lift_vector_max_accel_mps2 / accel_norm_lv
                )
                accel_norm_lv = cfg.lift_vector_max_accel_mps2
            # Heading-frame right vector in N/E. Positive target bank in the
            # live plant accelerates toward this side.
            right_lv = np.array([
                -math.sin(own_yaw_rad_lv),
                math.cos(own_yaw_rad_lv),
                0.0,
            ])
            lateral_accel_lv = float(np.dot(acceleration_lv, right_lv))
            up_accel_lv = -float(acceleration_lv[2])
            lift_vector_target_bank = float(np.clip(
                math.degrees(math.atan2(lateral_accel_lv, G)),
                -cfg.lift_vector_bank_limit_deg,
                cfg.lift_vector_bank_limit_deg,
            ))
            current_gamma_lv = math.degrees(math.asin(np.clip(
                vertical_speed / own_speed, -1.0, 1.0
            )))
            gamma_delta_lv = math.degrees(
                up_accel_lv * cfg.lift_vector_gamma_horizon_s
                / max(own_speed, 1.0)
            )
            lift_vector_target_gamma = float(np.clip(
                current_gamma_lv + gamma_delta_lv,
                -cfg.lift_vector_gamma_limit_deg,
                cfg.lift_vector_gamma_limit_deg,
            ))
            lift_vector_los_rate = math.degrees(
                float(np.linalg.norm(los_dot_lv))
            )
            lift_vector_accel = accel_norm_lv
            raw_lift_bank = lift_vector_target_bank
            raw_lift_gamma = lift_vector_target_gamma
            if not self._lift_vector_was_active:
                self._lift_vector_bank = bank
                self._lift_vector_gamma = current_gamma_lv
                if abs(raw_lift_bank) >= cfg.lift_vector_sign_min_bank_deg:
                    self._lift_vector_sign = signed_unit(raw_lift_bank)
                self._lift_vector_sign_until = (
                    now + cfg.lift_vector_sign_hold_s
                )
            requested_sign = signed_unit(
                raw_lift_bank, self._lift_vector_sign
            )
            if (
                abs(raw_lift_bank) >= cfg.lift_vector_sign_min_bank_deg
                and requested_sign != self._lift_vector_sign
            ):
                if now >= self._lift_vector_sign_until:
                    self._lift_vector_sign = requested_sign
                    self._lift_vector_sign_until = (
                        now + cfg.lift_vector_sign_hold_s
                    )
                else:
                    raw_lift_bank = (
                        self._lift_vector_sign * abs(raw_lift_bank)
                    )
            if dt and cfg.lift_vector_bank_tau_s > 0.0:
                bank_alpha_lv = 1.0 - math.exp(
                    -dt / cfg.lift_vector_bank_tau_s
                )
            else:
                bank_alpha_lv = 1.0
            filtered_bank_lv = self._lift_vector_bank + bank_alpha_lv * (
                raw_lift_bank - self._lift_vector_bank
            )
            if dt and cfg.lift_vector_bank_slew_degps < 1.0e8:
                bank_step_lv = cfg.lift_vector_bank_slew_degps * dt
                filtered_bank_lv = self._lift_vector_bank + float(np.clip(
                    filtered_bank_lv - self._lift_vector_bank,
                    -bank_step_lv,
                    bank_step_lv,
                ))
            if dt and cfg.lift_vector_gamma_tau_s > 0.0:
                gamma_alpha_lv = 1.0 - math.exp(
                    -dt / cfg.lift_vector_gamma_tau_s
                )
            else:
                gamma_alpha_lv = 1.0
            self._lift_vector_bank = float(np.clip(
                filtered_bank_lv,
                -cfg.lift_vector_bank_limit_deg,
                cfg.lift_vector_bank_limit_deg,
            ))
            self._lift_vector_gamma += gamma_alpha_lv * (
                raw_lift_gamma - self._lift_vector_gamma
            )
            self._lift_vector_gamma = float(np.clip(
                self._lift_vector_gamma,
                -cfg.lift_vector_gamma_limit_deg,
                cfg.lift_vector_gamma_limit_deg,
            ))
            lift_vector_target_bank = self._lift_vector_bank
            lift_vector_target_gamma = self._lift_vector_gamma
            self._lift_vector_was_active = True
        else:
            self._lift_vector_was_active = False

        # Convert desired horizontal course rate to coordinated bank demand.
        bank_from_rate = math.degrees(math.atan2(
            own_speed * math.radians(desired_turn_rate), G
        ))
        initial_commit = (
            cfg.initial_commit_duration_s > 0.0
            and elapsed <= cfg.initial_commit_duration_s
            and state == "break_turn"
            and ata > cfg.initial_commit_ata_deg
            and abs(bank) < cfg.initial_commit_release_bank_deg
        )
        if (
            cfg.coarse_turn_bank_deg > 0.0
            and cfg.coarse_turn_reentry_ata_deg > cfg.coarse_turn_ata_deg
        ):
            if not self._coarse_turn_initialized:
                self._coarse_turn_latched = ata >= cfg.coarse_turn_ata_deg
                self._coarse_turn_initialized = True
            elif self._coarse_turn_latched and ata < cfg.coarse_turn_ata_deg:
                self._coarse_turn_latched = False
            elif (
                not self._coarse_turn_latched
                and ata >= cfg.coarse_turn_reentry_ata_deg
            ):
                self._coarse_turn_latched = True
            coarse_turn = self._coarse_turn_latched
        else:
            coarse_turn = (
                cfg.coarse_turn_bank_deg > 0.0
                and ata >= cfg.coarse_turn_ata_deg
            )
        direct_course_bank = cfg.direct_course_bank_full_error_deg > 0.0
        integrated_rate_bank_active = (
            cfg.integrated_rate_bank_authority
            and (rear_rate_match_active or overshoot_control_active)
            and ata <= cfg.integrated_rate_bank_max_ata_deg
        )
        formula_rate_bank_active = (
            cfg.formula_rate_bank_authority
            and (
                formula_vpp_active
                or formula_zem_defense_active
                or attack_conversion_active
            )
        )
        if (
            terminal_track_active
            or turn_match_active
            or integrated_rate_bank_active
            or predictive_override_active
            or planner3d_active
            or formula_rate_bank_active
        ):
            target_bank = float(np.clip(
                bank_from_rate, -effective_max_bank, effective_max_bank
            ))
        elif direct_course_bank:
            # Re-evaluate directly from the newest horizontal target error on
            # every synchronized telemetry pair.  Only the tiny +/-180deg
            # ambiguity zone retains the previous side to prevent numerical
            # sign chatter at exactly six o'clock.
            if abs(guidance_az) < cfg.direct_course_rear_ambiguity_deg:
                direct_sign = signed_unit(guidance_az, self._turn_sign)
                self._turn_sign = direct_sign
            else:
                direct_sign = self._turn_sign
            error_fraction = float(np.clip(
                abs(guidance_az) / cfg.direct_course_bank_full_error_deg,
                0.0,
                1.0,
            ))
            target_bank = (
                direct_sign
                * effective_max_bank
                * error_fraction ** cfg.direct_course_bank_exponent
            )
        elif coarse_turn:
            target_bank = self._turn_sign * cfg.coarse_turn_bank_deg
        elif initial_commit:
            target_bank = self._turn_sign * cfg.initial_commit_bank_deg
        elif state == "break_turn" and ata > 60.0:
            target_bank = self._turn_sign * cfg.break_bank_deg
        else:
            target_bank = float(np.clip(
                bank_from_rate, -effective_max_bank, effective_max_bank
            ))
        if self._sequential_phase in ("setup", "vertical"):
            # Roll out before using elevator authority vertically. At high
            # bank the same pitch command only tightens the horizontal spiral.
            target_bank = 0.0
        vertical_bank_relief_active = (
            cfg.vertical_bank_relief_elevation_deg > 0.0
            and ata <= cfg.vertical_maneuver_ata_gate_deg
            and abs(world_elevation) >= cfg.vertical_bank_relief_elevation_deg
        )
        if vertical_bank_relief_active:
            excess_elevation = (
                abs(world_elevation) - cfg.vertical_bank_relief_elevation_deg
            )
            relief_fraction = float(np.clip(excess_elevation / 35.0, 0.0, 1.0))
            bank_scale = 1.0 - relief_fraction * (
                1.0 - cfg.vertical_bank_relief_min_scale
            )
            target_bank *= bank_scale
        terminal_vertical_unload_active = (
            terminal_track_active
            and cfg.terminal_vertical_unload_el_deg > 0.0
            and abs(world_elevation - pitch)
            >= cfg.terminal_vertical_unload_el_deg
            and abs(world_elevation - pitch) >= (
                cfg.terminal_vertical_unload_ratio * abs(aim_az)
            )
        )
        if terminal_vertical_unload_active:
            # A large bank rotates elevator authority into the horizontal
            # plane.  When the terminal miss is substantially vertical,
            # unload toward wings-level so full pitch command can move the
            # nose in elevation before the high-closure pass completes.
            target_bank *= cfg.terminal_vertical_unload_bank_scale
        if (
            direct_course_bank
            and not terminal_track_active
            and not vertical_bank_relief_active
            and ata >= cfg.direct_course_min_bank_ata_deg
            and cfg.direct_course_min_bank_deg > 0.0
            and abs(target_bank) < cfg.direct_course_min_bank_deg
        ):
            target_bank = (
                signed_unit(target_bank, self._turn_sign)
                * cfg.direct_course_min_bank_deg
            )
        if lift_vector_active:
            lift_blend = (
                cfg.lift_vector_defensive_blend
                if state == "defensive"
                else cfg.lift_vector_blend
            )
            lift_blend = float(np.clip(lift_blend, 0.0, 1.0))
            target_bank = float(np.clip(
                (1.0 - lift_blend) * target_bank
                + lift_blend * lift_vector_target_bank,
                -effective_max_bank,
                effective_max_bank,
            ))
        bank_error = wrap180(target_bank - bank)
        desired_roll_rate = float(np.clip(
            cfg.bank_kp * bank_error,
            -cfg.roll_rate_limit_degps,
            cfg.roll_rate_limit_degps,
        ))
        roll_cmd = float(np.clip(
            (desired_roll_rate - self._roll_rate) / cfg.roll_rate_gain_degps_per_unit,
            -cfg.roll_cmd_limit,
            cfg.roll_cmd_limit,
        ))
        bank_reversal_active = (
            cfg.bank_reversal_full_cmd > 0.0
            and target_bank * bank < 0.0
            and abs(bank_error) >= cfg.bank_reversal_error_deg
            and abs(bank) >= cfg.bank_reversal_release_bank_deg
        )
        if bank_reversal_active:
            # During a +80/-80 reversal the ordinary rate cascade removes
            # authority as soon as roll rate builds.  Hold full authority
            # until the old bank is mostly removed; then return to damping.
            roll_cmd = (
                signed_unit(target_bank) * cfg.bank_reversal_full_cmd
            )
        if coarse_turn and cfg.coarse_turn_roll_bias != 0.0:
            roll_cmd = float(np.clip(
                roll_cmd + self._turn_sign * cfg.coarse_turn_roll_bias,
                -cfg.roll_cmd_limit,
                cfg.roll_cmd_limit,
            ))
        if (
            direct_course_bank
            and cfg.direct_course_roll_bias != 0.0
            and abs(target_bank) > 5.0
        ):
            roll_cmd = float(np.clip(
                roll_cmd + signed_unit(target_bank) * cfg.direct_course_roll_bias,
                -cfg.roll_cmd_limit,
                cfg.roll_cmd_limit,
            ))
        if initial_commit and abs(roll_cmd) < cfg.initial_commit_min_roll_cmd:
            roll_cmd = self._turn_sign * cfg.initial_commit_min_roll_cmd
        max_test_active = (
            cfg.max_test_duration_s > 0.0
            and elapsed <= cfg.max_test_duration_s
            and (coarse_turn or not cfg.max_test_requires_coarse_turn)
        )
        if max_test_active:
            abs_bank = abs(bank)
            if abs_bank < cfg.max_test_roll_full_until_deg:
                roll_cmd = self._turn_sign * cfg.max_test_roll_full_cmd
            elif abs_bank < cfg.max_test_roll_taper_until_deg:
                roll_cmd = self._turn_sign * cfg.max_test_roll_taper_cmd
            elif abs_bank > cfg.max_test_roll_brake_above_deg:
                roll_cmd = -self._turn_sign * cfg.max_test_roll_brake_cmd

        horizontal_range = max(1.0, float(np.hypot(relative_position[0], relative_position[1])))
        predicted_target_altitude = target_altitude
        vertical_prediction_horizon = cfg.vertical_prediction_horizon_s
        vertical_prediction_stable = True
        if cfg.use_altitude_rate_vertical_prediction:
            vertical_prediction_stable = (
                self._target_vertical_stable_s
                >= cfg.vertical_prediction_min_stable_s
                and abs(self._target_vertical_accel)
                <= cfg.vertical_prediction_accel_limit_mps2
            )
            vertical_prediction_horizon = (
                cfg.vertical_prediction_stable_horizon_s
                if vertical_prediction_stable
                else cfg.vertical_prediction_unstable_horizon_s
            )
            predicted_target_altitude += (
                self._target_climb_rate * vertical_prediction_horizon
            )
        elif cfg.vertical_prediction_horizon_s > 0.0:
            # State D is down-positive, so positive D-rate means descending.
            predicted_target_altitude -= (
                self._target_velocity[2] * cfg.vertical_prediction_horizon_s
            )
        desired_gamma = math.degrees(math.atan2(
            predicted_target_altitude - altitude, horizontal_range
        ))
        if self._sequential_phase == "setup":
            desired_gamma = 0.0
        elif self._sequential_phase == "vertical":
            desired_gamma = (
                self._sequential_vertical_sign
                * cfg.sequential_vertical_gamma_deg
            )
        horizontal_energy_hold_active = (
            cfg.integrated_manager_enabled
            and ata > cfg.integrated_vertical_follow_ata_deg
        )
        if horizontal_energy_hold_active:
            # With the target outside the forward cone, chasing its altitude
            # creates the observed spiral climb/dive without reducing ATA.
            # Preserve a level flight path until horizontal acquisition.
            desired_gamma = 0.0
        if state in ("offensive_track", "weapons_track"):
            desired_gamma += cfg.fine_vertical_los_blend * aim_el
        if formula_vpp_result is not None and cfg.formula_vpp_vertical_enabled:
            desired_gamma = formula_vpp_result.gamma_deg
        formula_energy_gamma_correction = 0.0
        if (
            cfg.formula_energy_vertical_enabled
            and formula_vpp_result is not None
            and state != "defensive"
        ):
            energy_height = (own_speed * own_speed - target_speed * target_speed) / (2.0 * 9.80665)
            formula_energy_gamma_correction = float(np.clip(
                energy_height * cfg.formula_energy_gamma_gain_deg_per_m,
                -cfg.formula_energy_gamma_limit_deg,
                cfg.formula_energy_gamma_limit_deg,
            ))
            desired_gamma = float(np.clip(
                desired_gamma + formula_energy_gamma_correction,
                -cfg.formula_vpp_gamma_limit_deg,
                cfg.formula_vpp_gamma_limit_deg,
            ))
        gamma_limit = cfg.gamma_limit_deg
        if (
            cfg.vertical_maneuver_gamma_limit_deg > gamma_limit
            and ata <= cfg.vertical_maneuver_ata_gate_deg
        ):
            gamma_limit = cfg.vertical_maneuver_gamma_limit_deg
        defensive_escape_active = (
            cfg.defensive_vertical_escape
            and state == "defensive"
            and not mutual_commit_candidate
            and not overshoot_control_active
            and threat_ata <= cfg.defensive_escape_threat_ata_deg
            and ata >= cfg.defensive_escape_own_ata_deg
            and distance <= cfg.defensive_escape_range_m
        )
        defensive_escape_sign = 0
        if defensive_escape_active:
            altitude_separation = target_altitude - altitude
            if (
                cfg.defensive_escape_floor_m > 0.0
                and altitude <= cfg.defensive_escape_floor_m
            ):
                # At low altitude, separation must be upward regardless of
                # target altitude; the old rule could repeatedly choose a
                # descending escape and drive the aircraft into the floor.
                defensive_escape_sign = 1
            elif abs(altitude_separation) >= cfg.defensive_escape_deadband_m:
                # Increase vertical separation: descend from a higher bandit,
                # climb away from one below us.
                defensive_escape_sign = -signed_unit(altitude_separation)
            else:
                # At co-altitude, alternate the escape plane so a persistent
                # horizontal circle is not an easy tracking solution.
                phase = int(elapsed / max(cfg.defensive_escape_switch_s, 0.2))
                defensive_escape_sign = 1 if phase % 2 == 0 else -1
            desired_gamma = (
                defensive_escape_sign * cfg.defensive_escape_gamma_deg
            )
            gamma_limit = max(gamma_limit, cfg.defensive_escape_gamma_deg)
        if headon_deconflict_active and cfg.headon_deconflict_gamma_deg > 0.0:
            # Split away from the bandit's altitude.  At co-altitude use an
            # upward break while ample height remains; unlike the periodic
            # defensive escape this sign cannot flip during the short pass.
            altitude_separation = target_altitude - altitude
            if altitude <= max(cfg.defensive_escape_floor_m, 1200.0):
                headon_gamma_sign = 1
            elif altitude_separation > 75.0:
                headon_gamma_sign = -1
            else:
                headon_gamma_sign = 1
            desired_gamma = (
                headon_gamma_sign * cfg.headon_deconflict_gamma_deg
            )
            gamma_limit = max(gamma_limit, cfg.headon_deconflict_gamma_deg)
        if lift_vector_active:
            lift_blend = (
                cfg.lift_vector_defensive_blend
                if state == "defensive"
                else cfg.lift_vector_blend
            )
            lift_blend = float(np.clip(lift_blend, 0.0, 1.0))
            desired_gamma = (
                (1.0 - lift_blend) * desired_gamma
                + lift_blend * lift_vector_target_gamma
            )
            gamma_limit = max(gamma_limit, cfg.lift_vector_gamma_limit_deg)
        if planner3d_active:
            # The 3-D planner owns both axes as one manoeuvre. Applying the old
            # horizontal-hold or periodic vertical-escape result afterwards
            # would split the selected lift-vector command back into the two
            # contradictory controllers this mode is intended to replace.
            desired_gamma = self._planner3d_gamma
            gamma_limit = max(gamma_limit, cfg.planner3d_gamma_deg)
        energy_gamma_guard_active = (
            cfg.integrated_manager_enabled
            and cfg.integrated_energy_gamma_guard_speed_mps > 0.0
            and own_speed < cfg.integrated_energy_gamma_guard_speed_mps
            and desired_gamma > 0.0
            and not defensive_escape_active
        )
        if energy_gamma_guard_active:
            # Do not spend the last usable energy following a climbing target.
            # Level first; horizontal acquisition remains active.
            desired_gamma = 0.0
        desired_gamma = float(np.clip(desired_gamma, -gamma_limit, gamma_limit))
        flight_path_angle = math.degrees(math.asin(np.clip(vertical_speed / own_speed, -1.0, 1.0)))
        gamma_error = desired_gamma - flight_path_angle
        pitch_control_error = gamma_error
        pitch_control_kp = cfg.gamma_kp
        if terminal_track_active and cfg.terminal_pitch_attitude_kp > 0.0:
            pitch_control_error = world_elevation - pitch
            pitch_control_kp = cfg.terminal_pitch_attitude_kp
        desired_pitch_rate = float(np.clip(
            pitch_control_kp * pitch_control_error
            + (
                cfg.terminal_pitch_los_rate_gain * self._world_elevation_rate
                if terminal_track_active else 0.0
            ),
            -cfg.pitch_rate_limit_degps,
            cfg.pitch_rate_limit_degps,
        ))
        bank_fraction = float(np.clip(
            abs(bank) / effective_max_bank, 0.0, 1.0
        ))
        turn_pull = cfg.turn_pull_at_max_bank * bank_fraction * bank_fraction
        if cfg.integrated_manager_enabled and energy_deficit:
            turn_pull *= cfg.integrated_energy_pull_scale
        pitch_cmd = float(np.clip(
            cfg.pitch_trim + turn_pull - desired_pitch_rate / cfg.pitch_rate_gain_degps_per_unit,
            -cfg.pitch_cmd_limit,
            cfg.pitch_cmd_limit,
        ))
        if cfg.vertical_speed_damping_gain > 0.0 and not (
            terminal_track_active and cfg.terminal_pitch_attitude_kp > 0.0
        ):
            vertical_speed_error = vertical_speed
            if cfg.vertical_speed_damping_track_desired_gamma:
                desired_vertical_speed = own_speed * math.sin(
                    math.radians(desired_gamma)
                )
                vertical_speed_error -= desired_vertical_speed
            excess_vertical_speed = math.copysign(
                max(0.0, abs(vertical_speed_error)
                    - cfg.vertical_speed_damping_deadband_mps),
                vertical_speed_error,
            )
            # Positive live vertical speed is climb; positive pitch_cmd is
            # nose-down, so this directly brakes a climb (and vice versa).
            pitch_cmd = float(np.clip(
                pitch_cmd
                + cfg.vertical_speed_damping_gain * excess_vertical_speed,
                -cfg.pitch_cmd_limit,
                cfg.pitch_cmd_limit,
            ))
        spiral_recovery_active = (
            cfg.spiral_recovery_enabled
            and ata >= cfg.spiral_recovery_min_ata_deg
            and vertical_speed >= cfg.spiral_recovery_vertical_speed_mps
            and self._ata_rate >= cfg.spiral_recovery_stall_ata_rate_degps
            and not terminal_track_active
        )
        if spiral_recovery_active:
            relief = min(
                cfg.spiral_recovery_max_pitch_relief,
                cfg.spiral_recovery_pitch_gain
                * (vertical_speed - cfg.spiral_recovery_vertical_speed_mps),
            )
            pitch_cmd = float(np.clip(
                pitch_cmd + relief,
                -cfg.pitch_cmd_limit,
                cfg.pitch_cmd_limit,
            ))
        sustained_pull_cmd = 0.0
        sustained_pull_active = (
            cfg.sustained_pull_rear_cmd < 0.0
            and abs(bank) >= cfg.sustained_pull_min_bank_deg
            and own_speed >= cfg.sustained_pull_min_speed_mps
            and altitude >= cfg.sustained_pull_min_altitude_m
        )
        if sustained_pull_active:
            if ata >= cfg.sustained_pull_rear_ata_deg:
                sustained_pull_cmd = cfg.sustained_pull_rear_cmd
            elif (
                cfg.sustained_pull_mid_cmd < 0.0
                and ata >= cfg.sustained_pull_mid_ata_deg
            ):
                sustained_pull_cmd = cfg.sustained_pull_mid_cmd
            if sustained_pull_cmd < 0.0:
                pitch_cmd = max(-cfg.pitch_cmd_limit, min(pitch_cmd, sustained_pull_cmd))
        max_test_pull_active = (
            max_test_active
            and cfg.max_test_pitch_cmd < 0.0
            and abs(bank) >= cfg.max_test_pitch_bank_gate_deg
            and ata >= cfg.max_test_pitch_ata_gate_deg
            and own_speed >= cfg.max_test_min_speed_mps
            and altitude >= cfg.max_test_min_altitude_m
        )
        if max_test_pull_active:
            if (
                cfg.max_test_pitch_vertical_gain > 0.0
                and cfg.max_test_pitch_relaxed_cmd > cfg.max_test_pitch_cmd
            ):
                test_pitch_cmd = (
                    cfg.max_test_pitch_cmd
                    + cfg.max_test_pitch_vertical_gain
                    * (vertical_speed - cfg.max_test_target_vertical_speed_mps)
                )
                if (
                    cfg.max_test_pitch_bank_relax_gain > 0.0
                    and cfg.coarse_turn_bank_deg > 0.0
                ):
                    test_pitch_cmd += (
                        cfg.max_test_pitch_bank_relax_gain
                        * max(0.0, cfg.coarse_turn_bank_deg - abs(bank))
                    )
                pitch_cmd = float(np.clip(
                    test_pitch_cmd,
                    cfg.max_test_pitch_cmd,
                    cfg.max_test_pitch_relaxed_cmd,
                ))
            else:
                pitch_cmd = max(-cfg.pitch_cmd_limit, min(pitch_cmd, cfg.max_test_pitch_cmd))

        if state == "merge":
            target_throttle = cfg.throttle_merge
        elif state == "defensive":
            target_throttle = cfg.throttle_defensive
        elif state == "weapons_track":
            target_throttle = cfg.throttle_weapons
        elif state == "offensive_track":
            desired_speed = float(np.clip(target_speed + 15.0, 230.0, 330.0))
            target_throttle = cfg.corner_throttle_base + cfg.corner_speed_gain * (desired_speed - own_speed)
        else:
            target_throttle = cfg.corner_throttle_base + cfg.corner_speed_gain * (
                cfg.corner_speed_mps - own_speed
            )
        high_bank_speed_control_active = (
            cfg.high_bank_target_speed_mps > 0.0
            and abs(bank) >= cfg.high_bank_speed_min_bank_deg
            and ata >= cfg.high_bank_speed_min_ata_deg
            and state != "merge"
        )
        high_bank_desired_speed = cfg.high_bank_target_speed_mps
        if high_bank_speed_control_active:
            if cfg.high_bank_dynamic_speed_enabled:
                range_span = max(
                    cfg.high_bank_dynamic_far_range_m
                    - cfg.high_bank_dynamic_near_range_m,
                    1.0,
                )
                range_blend = float(np.clip(
                    (distance - cfg.high_bank_dynamic_near_range_m) / range_span,
                    0.0,
                    1.0,
                ))
                # Negative closure means the target is escaping. Increase
                # speed smoothly before range alone reaches the far gate.
                escape_blend = float(np.clip(
                    (-closure - 10.0) / 60.0,
                    0.0,
                    1.0,
                ))
                speed_blend = max(range_blend, escape_blend)
                chase_speed = float(np.clip(
                    target_speed + cfg.high_bank_dynamic_target_margin_mps,
                    cfg.high_bank_target_speed_mps,
                    cfg.high_bank_dynamic_max_speed_mps,
                ))
                high_bank_desired_speed = (
                    cfg.high_bank_target_speed_mps
                    + speed_blend
                    * (chase_speed - cfg.high_bank_target_speed_mps)
                )
            target_throttle = (
                cfg.high_bank_speed_throttle_base
                + cfg.high_bank_speed_throttle_gain
                * (high_bank_desired_speed - own_speed)
            )
        desired_closure = None
        if (
            cfg.closure_throttle_gain > 0.0
            and ata <= cfg.closure_throttle_ata_deg
            and distance <= cfg.closure_throttle_range_m
        ):
            if distance >= cfg.closure_throttle_far_range_m:
                desired_closure = cfg.closure_target_far_mps
            elif distance >= cfg.closure_throttle_near_range_m:
                desired_closure = cfg.closure_target_mid_mps
            else:
                desired_closure = cfg.closure_target_near_mps
            if vertical_alignment_active:
                desired_closure = min(
                    desired_closure,
                    cfg.vertical_alignment_target_closure_mps,
                )
            # run0193 live audit: closure_throttle_gain=0.005 was tuned
            # against modest closure errors, but a real merge routinely
            # produces closure of 250-330 m/s against a ~45 m/s target
            # (excess 200-280 m/s). Unclamped, that drove target_throttle
            # to -0.5 to -1.0, clipped to a literal 0.0 for over a second
            # at a time during exactly the highest-energy-demand moment of
            # the fight (own_speed measured falling 208->180 m/s here while
            # throttle sat at 0), bleeding the turn-rate margin the rest of
            # the engagement never recovered. Clamp the error term so the
            # correction cannot exceed roughly the base value.
            closure_error = closure - desired_closure
            if math.isfinite(cfg.closure_throttle_error_limit_mps):
                limit = max(0.0, cfg.closure_throttle_error_limit_mps)
                closure_error = float(np.clip(closure_error, -limit, limit))
            target_throttle = (
                cfg.closure_throttle_base
                - cfg.closure_throttle_gain * closure_error
            )
        lag_energy_preserve_active = (
            lag_pursuit_active
            and cfg.lag_pursuit_energy_target_speed_mps > 0.0
        )
        if lag_energy_preserve_active:
            lag_energy_throttle = (
                cfg.lag_pursuit_energy_throttle_base
                + cfg.lag_pursuit_energy_throttle_gain
                * (cfg.lag_pursuit_energy_target_speed_mps - own_speed)
            )
            target_throttle = max(target_throttle, lag_energy_throttle)
        if planner3d_active:
            planner_throttle = (
                0.55
                + 0.015 * (self._planner3d_target_speed - own_speed)
            )
            target_throttle = float(np.clip(
                planner_throttle, cfg.throttle_min, cfg.throttle_max
            ))
        energy_recovery_active = (
            cfg.minimum_energy_speed_mps > 0.0
            and own_speed < cfg.minimum_energy_speed_mps
        )
        if energy_recovery_active:
            target_throttle = max(
                target_throttle, cfg.minimum_energy_throttle
            )
        target_throttle = float(np.clip(target_throttle, cfg.throttle_min, cfg.throttle_max))
        if dt:
            max_step = cfg.throttle_slew_per_s * dt
            self._throttle += float(np.clip(target_throttle - self._throttle, -max_step, max_step))
        else:
            self._throttle = target_throttle
        throttle = float(np.clip(self._throttle, cfg.throttle_min, cfg.throttle_max))

        yaw_cmd = 0.0
        if (
            cfg.turn_rudder_assist > 0.0
            and abs(bank) >= cfg.turn_rudder_min_bank_deg
            and ata >= cfg.turn_rudder_min_ata_deg
        ):
            bank_scale = float(np.clip(
                (abs(bank) - cfg.turn_rudder_min_bank_deg)
                / max(1.0, effective_max_bank - cfg.turn_rudder_min_bank_deg),
                0.0,
                1.0,
            ))
            yaw_cmd = float(np.clip(
                -signed_unit(target_bank, self._turn_sign)
                * cfg.turn_rudder_assist * bank_scale,
                -1.0,
                1.0,
            ))
        action = np.array(
            [roll_cmd, pitch_cmd, yaw_cmd, throttle], dtype=np.float32
        )
        info = {
            "state": state,
            "engagement_mode": engagement_mode,
            "ata": ata,
            "aa": aa,
            "threat_ata": threat_ata,
            "advantage_score": self._advantage_score,
            "advantage_ttc": self._advantage_ttc,
            "advantage_manager_reason": self._advantage_manager_reason,
            "distance": distance,
            "closure_rate": closure,
            "los_az": los_az,
            "los_el": los_el,
            "aim_az": guidance_az,
            "aim_course_error": aim_course_error,
            "aim_el": aim_el,
            "los_rate": guidance_los_rate,
            "ata_rate": self._ata_rate,
            "own_yaw_rate": self._own_yaw_rate,
            "target_yaw_rate": self._target_yaw_rate,
            "target_yaw_accel": self._target_yaw_accel,
            "target_turn_stable_s": self._target_turn_stable_s,
            "prediction_stable": prediction_stable,
            "intercept_horizon": horizon,
            "lag_pursuit_active": lag_pursuit_active,
            "lag_offset_m": lag_offset_m,
            "lag_pursuit_scale": self._lag_pursuit_scale,
            "lag_energy_preserve_active": lag_energy_preserve_active,
            "vertical_alignment_active": vertical_alignment_active,
            "turn_match_active": turn_match_active,
            "turn_match_candidate_s": self._turn_match_candidate_s,
            "terminal_track_active": terminal_track_active,
            "terminal_vertical_unload_active": terminal_vertical_unload_active,
            "desired_turn_rate": desired_turn_rate,
            "rule_desired_turn_rate": rule_desired_turn_rate,
            "predictive_guidance_active": predictive_guidance_active,
            "spiral_recovery_active": spiral_recovery_active,
            "rate_deficit_overbank_active": rate_deficit_overbank_active,
            "sequential_maneuver_phase": self._sequential_phase,
            "predictive_horizon": predictive_horizon,
            "predictive_score": predictive_score,
            "predictive_rule_score": predictive_rule_score,
            "predictive_score_gain": predictive_score_gain,
            "predictive_override_active": predictive_override_active,
            "predictive_future_error": predictive_future_error,
            "predictive_future_threat_error": predictive_future_threat_error,
            "predictive_worst_target_rate": predictive_worst_target_rate,
            "predictive_sign_guard_active": predictive_sign_guard_active,
            "planner3d_active": planner3d_active,
            "planner3d_replanned": planner3d_replanned,
            "planner3d_manoeuvre": self._planner3d_name,
            "planner3d_score": self._planner3d_score,
            "planner3d_horizon": planner_horizon,
            "planner3d_threat_pressure": self._planner3d_threat_pressure,
            "planner3d_turn_rate": self._planner3d_turn_rate,
            "planner3d_gamma": self._planner3d_gamma,
            "planner3d_target_speed": self._planner3d_target_speed,
            "planner3d_future_ata": self._planner3d_future_ata,
            "planner3d_future_threat_ata": self._planner3d_future_threat_ata,
            "planner3d_future_range": self._planner3d_future_range,
            "planner3d_worst_target_turn": self._planner3d_worst_target_turn,
            "planner3d_worst_target_gamma": self._planner3d_worst_target_gamma,
            "target_bank": target_bank,
            "current_bank": bank,
            "bank_error": bank_error,
            "bank_reversal_active": bank_reversal_active,
            "rear_rate_match_active": rear_rate_match_active,
            "integrated_rate_bank_active": integrated_rate_bank_active,
            "formula_rate_bank_active": formula_rate_bank_active,
            "overshoot_control_active": overshoot_control_active,
            "attack_conversion_active": attack_conversion_active,
            "headon_deconflict_active": headon_deconflict_active,
            "lift_vector_active": lift_vector_active,
            "lift_vector_target_bank": lift_vector_target_bank,
            "lift_vector_target_gamma": lift_vector_target_gamma,
            "lift_vector_los_rate": lift_vector_los_rate,
            "lift_vector_accel": lift_vector_accel,
            "formula_vpp_active": formula_vpp_active,
            "formula_vpp_blend": self._formula_vpp_blend,
            "formula_vpp_mode": (
                formula_vpp_result.mode if formula_vpp_result is not None else "inactive"
            ),
            "formula_vpp_turn_rate_degps": (
                formula_vpp_result.turn_rate_degps if formula_vpp_result is not None else 0.0
            ),
            "formula_vpp_t_cpa_s": (
                formula_vpp_result.t_cpa_s if formula_vpp_result is not None else 99.0
            ),
            "formula_vpp_d_cpa_m": (
                formula_vpp_result.d_cpa_m if formula_vpp_result is not None else 99999.0
            ),
            "formula_vpp_turn_circle_active": (
                formula_vpp_result.turn_circle_active if formula_vpp_result is not None else False
            ),
            "formula_vpp_pass_detected": formula_vpp_pass_detected,
            "formula_vpp_recommit_active": formula_vpp_recommit_active,
            "formula_vpp_recommit_remaining_s": max(
                0.0, self._formula_vpp_recommit_until - now
            ),
            "formula_projectile_tof_s": (
                formula_vpp_result.projectile_tof_s if formula_vpp_result is not None else 0.0
            ),
            "formula_apg_weight": (
                formula_vpp_result.apg_weight if formula_vpp_result is not None else 0.0
            ),
            "formula_energy_gamma_correction": formula_energy_gamma_correction,
            "formula_zem_defense_active": formula_zem_defense_active,
            "formula_zem_escape_sign": formula_zem_escape_sign,
            "formula_zem_burst_remaining_s": max(
                0.0, self._formula_zem_burst_until - now
            ),
            "formula_zem_cooldown_remaining_s": max(
                0.0, self._formula_zem_cooldown_until - now
            ),
            "vertical_follow_active": vertical_follow_active,
            "horizontal_energy_hold_active": horizontal_energy_hold_active,
            "mutual_commit_active": mutual_commit_candidate,
            "energy_deficit": energy_deficit,
            "energy_gamma_guard_active": energy_gamma_guard_active,
            "desired_roll_rate": desired_roll_rate,
            "measured_roll_rate": self._roll_rate,
            "desired_gamma": desired_gamma,
            "world_elevation": world_elevation,
            "world_elevation_rate": self._world_elevation_rate,
            "target_climb_rate": self._target_climb_rate,
            "target_vertical_accel": self._target_vertical_accel,
            "target_vertical_stable_s": self._target_vertical_stable_s,
            "vertical_prediction_horizon": vertical_prediction_horizon,
            "vertical_prediction_stable": vertical_prediction_stable,
            "vertical_bank_relief_active": vertical_bank_relief_active,
            "defensive_escape_active": defensive_escape_active,
            "defensive_escape_sign": defensive_escape_sign,
            "post_defense_conversion_active": post_defense_conversion_active,
            "post_defense_conversion_armed": (
                now < self._post_defense_conversion_armed_until
            ),
            "conversion_rear_offset_m": conversion_rear_offset_m,
            "conversion_lateral_offset_m": conversion_lateral_offset_m,
            "flight_path_angle": flight_path_angle,
            "gamma_error": gamma_error,
            "desired_pitch_rate": desired_pitch_rate,
            "measured_pitch_rate": self._pitch_rate,
            "turn_pitch_feedforward": turn_pull,
            "own_pitch": pitch,
            "own_alt": altitude,
            "vertical_speed": vertical_speed,
            "target_speed": target_speed,
            "speed_error": cfg.corner_speed_mps - own_speed,
            "target_throttle": target_throttle,
            "desired_closure": desired_closure,
            "energy_recovery_active": energy_recovery_active,
            "roll_cmd": roll_cmd,
            "initial_commit": initial_commit,
            "coarse_turn": coarse_turn,
            "max_test_active": max_test_active,
            "pitch_cmd": pitch_cmd,
            "yaw_cmd": yaw_cmd,
            "high_bank_speed_control_active": high_bank_speed_control_active,
            "high_bank_desired_speed": high_bank_desired_speed,
            "sustained_pull_active": sustained_pull_cmd < 0.0,
            "sustained_pull_cmd": sustained_pull_cmd,
            "max_test_pull_active": max_test_pull_active,
        }

        self._prev_time = now
        self._prev_distance = distance
        self._prev_closure = closure
        self._prev_ata = ata
        self._prev_los_az = los_az
        self._prev_horizontal_los_error = horizontal_los_error
        self._prev_own_yaw = own_yaw
        self._prev_target_yaw = target_yaw
        self._prev_bank = bank
        self._prev_pitch = pitch
        self._prev_alt = altitude
        self._prev_target_alt = target_altitude
        self._prev_target_position = target_position.copy()
        self.state_log.append(state)
        self.info_log.append(info)
        return ActionResult(
            action=action,
            source=f"{cfg.controller_name}[{state}]",
            confidence=1.0,
            info=info,
        )
