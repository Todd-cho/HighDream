"""Frozen tuned profiles shared by live inference and training."""
from __future__ import annotations

from dogfight.ai.integrated_bfm_controller import IntegratedBFMConfig, IntegratedBFMController


def build_w53_config() -> IntegratedBFMConfig:
    """Return the frozen W53 residual-RL baseline configuration."""
    return IntegratedBFMConfig(
        controller_name="w53",
        use_horizontal_course_guidance=True,
        use_constant_turn_prediction=True,
        adaptive_turn_prediction=True,
        turn_prediction_max_arc_deg=30.0,
        turn_prediction_unstable_horizon_s=0.8,
        turn_prediction_min_stable_s=0.7,
        turn_prediction_yaw_accel_limit_degps2=18.0,
        lag_pursuit_ata_deg=50.0,
        lag_pursuit_range_m=2800.0,
        lag_pursuit_closure_mps=170.0,
        lag_pursuit_offset_min_m=250.0,
        lag_pursuit_offset_max_m=750.0,
        lag_pursuit_offset_gain_s=2.0,
        vertical_alignment_elevation_deg=20.0,
        vertical_alignment_range_m=3200.0,
        vertical_alignment_closure_mps=60.0,
        vertical_alignment_lag_offset_m=1400.0,
        vertical_alignment_lag_gain_m_per_deg=40.0,
        vertical_alignment_lag_max_m=3000.0,
        vertical_alignment_target_closure_mps=0.0,
        turn_match_enter_ata_deg=18.0,
        turn_match_exit_ata_deg=32.0,
        turn_match_enter_range_m=3000.0,
        turn_match_exit_range_m=3400.0,
        turn_match_min_threat_ata_deg=40.0,
        turn_match_enter_course_error_deg=12.0,
        turn_match_exit_course_error_deg=24.0,
        turn_match_enter_max_closure_mps=220.0,
        turn_match_entry_hold_s=0.5,
        turn_match_require_stable_prediction=True,
        turn_match_yaw_rate_gain=0.65,
        turn_match_course_gain=0.55,
        turn_match_los_rate_gain=0.20,
        turn_match_rate_limit_degps=12.0,
        turn_match_sign_guard_error_deg=4.0,
        turn_match_sign_guard_rate_degps=3.0,
        terminal_track_enter_ata_deg=20.0,
        terminal_track_exit_ata_deg=30.0,
        terminal_track_enter_range_m=1800.0,
        terminal_track_exit_range_m=2100.0,
        terminal_track_prelock_ata_deg=0.0,
        terminal_track_prelock_range_m=0.0,
        terminal_track_min_threat_ata_deg=0.0,
        terminal_track_yaw_rate_gain=0.50,
        terminal_track_course_gain=2.00,
        terminal_track_los_rate_gain=0.40,
        terminal_track_rate_limit_degps=16.0,
        terminal_pitch_attitude_kp=2.0,
        terminal_pitch_los_rate_gain=1.0,
        terminal_vertical_unload_el_deg=6.0,
        terminal_vertical_unload_ratio=0.4,
        target_velocity_alpha=0.80,
        horizon_defensive_s=0.8,
        guidance_rear_commit_deg=180.0,
        direct_course_bank_full_error_deg=60.0,
        direct_course_bank_exponent=0.50,
        direct_course_rear_ambiguity_deg=170.0,
        direct_course_roll_bias=0.07,
        direct_course_min_bank_deg=78.0,
        direct_course_min_bank_ata_deg=35.0,
        bank_reversal_error_deg=130.0,
        bank_reversal_full_cmd=0.75,
        bank_reversal_release_bank_deg=30.0,
        turn_rate_break_degps=12.0,
        turn_rate_reacquire_degps=11.0,
        turn_rate_track_degps=8.0,
        turn_rate_defensive_degps=12.0,
        acquisition_min_turn_rate_degps=0.0,
        max_bank_deg=82.0,
        break_bank_deg=80.0,
        coarse_turn_bank_deg=0.0,
        roll_rate_limit_degps=90.0,
        roll_cmd_limit=0.80,
        bank_kp=1.5,
        gamma_limit_deg=30.0,
        gamma_kp=1.2,
        pitch_rate_limit_degps=16.0,
        turn_pull_at_max_bank=-0.80,
        pitch_cmd_limit=1.00,
        fine_vertical_los_blend=0.0,
        vertical_prediction_horizon_s=0.0,
        use_altitude_rate_vertical_prediction=True,
        vertical_prediction_stable_horizon_s=1.8,
        vertical_prediction_unstable_horizon_s=0.35,
        vertical_prediction_min_stable_s=0.7,
        vertical_prediction_accel_limit_mps2=35.0,
        vertical_maneuver_gamma_limit_deg=68.0,
        vertical_maneuver_ata_gate_deg=75.0,
        vertical_bank_relief_elevation_deg=18.0,
        vertical_bank_relief_min_scale=0.35,
        vertical_speed_damping_deadband_mps=20.0,
        vertical_speed_damping_gain=0.006,
        vertical_speed_damping_track_desired_gamma=True,
        corner_speed_mps=215.0,
        corner_throttle_base=0.85,
        corner_speed_gain=0.010,
        throttle_min=0.0,
        throttle_max=1.00,
        throttle_merge=0.95,
        throttle_slew_per_s=0.80,
        closure_throttle_ata_deg=55.0,
        closure_throttle_range_m=4000.0,
        closure_throttle_far_range_m=3000.0,
        closure_throttle_near_range_m=1500.0,
        closure_target_far_mps=100.0,
        closure_target_mid_mps=45.0,
        closure_target_near_mps=10.0,
        closure_throttle_base=0.48,
        closure_throttle_gain=0.005,
        minimum_energy_speed_mps=165.0,
        minimum_energy_throttle=0.90,
    )


def build_w53_controller() -> IntegratedBFMController:
    """Build a fresh, reset W53 controller."""
    return IntegratedBFMController(build_w53_config())


def build_w56_config() -> IntegratedBFMConfig:
    """Return the frozen W56 residual-RL baseline configuration.

    W56 differs from W53 in exactly the fields below (RL_TRAINING_ADDENDUM_W56_KO,
    2026-08-23): W53's terminal_vertical_unload released target bank from ~70deg
    to ~11deg during close-range alignment, reopening ATA, so W56 disables it and
    widens/enables prelock terminal tracking instead. Every other field is
    identical to build_w53_config() -- kept as its own frozen literal (not derived
    from build_w53_config()) so a future edit to the W53 profile can never leak
    into the W56 training base.
    """
    return IntegratedBFMConfig(
        controller_name="w56",
        use_horizontal_course_guidance=True,
        use_constant_turn_prediction=True,
        adaptive_turn_prediction=True,
        turn_prediction_max_arc_deg=30.0,
        turn_prediction_unstable_horizon_s=0.8,
        turn_prediction_min_stable_s=0.7,
        turn_prediction_yaw_accel_limit_degps2=18.0,
        lag_pursuit_ata_deg=50.0,
        lag_pursuit_range_m=2800.0,
        lag_pursuit_closure_mps=170.0,
        lag_pursuit_offset_min_m=250.0,
        lag_pursuit_offset_max_m=750.0,
        lag_pursuit_offset_gain_s=2.0,
        vertical_alignment_elevation_deg=20.0,
        vertical_alignment_range_m=3200.0,
        vertical_alignment_closure_mps=60.0,
        vertical_alignment_lag_offset_m=1400.0,
        vertical_alignment_lag_gain_m_per_deg=40.0,
        vertical_alignment_lag_max_m=3000.0,
        vertical_alignment_target_closure_mps=0.0,
        turn_match_enter_ata_deg=18.0,
        turn_match_exit_ata_deg=32.0,
        turn_match_enter_range_m=3000.0,
        turn_match_exit_range_m=3400.0,
        turn_match_min_threat_ata_deg=40.0,
        turn_match_enter_course_error_deg=12.0,
        turn_match_exit_course_error_deg=24.0,
        turn_match_enter_max_closure_mps=220.0,
        turn_match_entry_hold_s=0.5,
        turn_match_require_stable_prediction=True,
        turn_match_yaw_rate_gain=0.65,
        turn_match_course_gain=0.55,
        turn_match_los_rate_gain=0.20,
        turn_match_rate_limit_degps=12.0,
        turn_match_sign_guard_error_deg=4.0,
        turn_match_sign_guard_rate_degps=3.0,
        terminal_track_enter_ata_deg=20.0,
        terminal_track_exit_ata_deg=30.0,
        terminal_track_enter_range_m=1800.0,
        terminal_track_exit_range_m=4500.0,
        terminal_track_prelock_ata_deg=15.0,
        terminal_track_prelock_range_m=3000.0,
        terminal_track_min_threat_ata_deg=0.0,
        terminal_track_yaw_rate_gain=0.50,
        terminal_track_course_gain=2.00,
        terminal_track_los_rate_gain=0.40,
        terminal_track_rate_limit_degps=16.0,
        terminal_pitch_attitude_kp=2.0,
        terminal_pitch_los_rate_gain=1.0,
        terminal_vertical_unload_el_deg=0.0,
        terminal_vertical_unload_ratio=0.7,
        target_velocity_alpha=0.80,
        horizon_defensive_s=0.8,
        guidance_rear_commit_deg=180.0,
        direct_course_bank_full_error_deg=60.0,
        direct_course_bank_exponent=0.50,
        direct_course_rear_ambiguity_deg=170.0,
        direct_course_roll_bias=0.07,
        direct_course_min_bank_deg=78.0,
        direct_course_min_bank_ata_deg=35.0,
        bank_reversal_error_deg=130.0,
        bank_reversal_full_cmd=0.75,
        bank_reversal_release_bank_deg=30.0,
        turn_rate_break_degps=12.0,
        turn_rate_reacquire_degps=11.0,
        turn_rate_track_degps=8.0,
        turn_rate_defensive_degps=12.0,
        acquisition_min_turn_rate_degps=0.0,
        max_bank_deg=82.0,
        break_bank_deg=80.0,
        coarse_turn_bank_deg=0.0,
        roll_rate_limit_degps=90.0,
        roll_cmd_limit=0.80,
        bank_kp=1.5,
        gamma_limit_deg=30.0,
        gamma_kp=1.2,
        pitch_rate_limit_degps=16.0,
        turn_pull_at_max_bank=-0.80,
        pitch_cmd_limit=1.00,
        fine_vertical_los_blend=0.0,
        vertical_prediction_horizon_s=0.0,
        use_altitude_rate_vertical_prediction=True,
        vertical_prediction_stable_horizon_s=1.8,
        vertical_prediction_unstable_horizon_s=0.35,
        vertical_prediction_min_stable_s=0.7,
        vertical_prediction_accel_limit_mps2=35.0,
        vertical_maneuver_gamma_limit_deg=68.0,
        vertical_maneuver_ata_gate_deg=75.0,
        vertical_bank_relief_elevation_deg=18.0,
        vertical_bank_relief_min_scale=0.35,
        vertical_speed_damping_deadband_mps=20.0,
        vertical_speed_damping_gain=0.006,
        vertical_speed_damping_track_desired_gamma=True,
        corner_speed_mps=215.0,
        corner_throttle_base=0.85,
        corner_speed_gain=0.010,
        throttle_min=0.0,
        throttle_max=1.00,
        throttle_merge=0.95,
        throttle_slew_per_s=0.80,
        closure_throttle_ata_deg=55.0,
        closure_throttle_range_m=4000.0,
        closure_throttle_far_range_m=3000.0,
        closure_throttle_near_range_m=1500.0,
        closure_target_far_mps=100.0,
        closure_target_mid_mps=45.0,
        closure_target_near_mps=10.0,
        closure_throttle_base=0.48,
        closure_throttle_gain=0.005,
        minimum_energy_speed_mps=165.0,
        minimum_energy_throttle=0.90,
    )


def build_w56_controller() -> IntegratedBFMController:
    """Build a fresh, reset W56 controller."""
    return IntegratedBFMController(build_w56_config())


def build_w97_config() -> IntegratedBFMConfig:
    """Return the W97 live profile for residual-RL training.

    Keep this builder explicit so training and live inference cannot silently
    use different baselines.  W97 is W53 plus the measured-energy W89 fields
    and the coupled 3-D research planner introduced in W95--W97.
    """
    cfg = build_w53_config()
    cfg.controller_name = "w97"
    cfg.predictive_guidance_enabled = False
    cfg.predictive_adversarial_enabled = True
    cfg.predictive_guidance_min_ata_deg = 25.0
    cfg.predictive_horizon_min_s = 1.0
    cfg.predictive_horizon_max_s = 1.0
    cfg.predictive_candidate_count = 13
    cfg.predictive_turn_limit_degps = 17.0
    cfg.predictive_live_turn_envelope = True
    cfg.predictive_response_delay_s = 0.15
    cfg.predictive_turn_slew_degps2 = 40.0
    cfg.predictive_override_min_score_gain = 1.5
    cfg.predictive_override_min_rate_delta_degps = 1.0
    cfg.predictive_target_turn_limit_degps = 18.0
    cfg.predictive_threat_cone_deg = 30.0
    cfg.predictive_threat_weight = 1.5
    cfg.predictive_midpoint_weight = 0.65
    cfg.lag_pursuit_offset_min_m = 250.0
    cfg.lag_pursuit_offset_max_m = 800.0
    cfg.lag_pursuit_offset_gain_s = 2.5
    cfg.lag_pursuit_terminal_taper_start_deg = 35.0
    cfg.lag_pursuit_terminal_taper_end_deg = 20.0
    cfg.lag_pursuit_mutual_lateral_offset_m = 450.0
    cfg.lag_pursuit_mutual_threat_ata_deg = 30.0
    cfg.lag_pursuit_taper_max_closure_mps = 140.0
    cfg.lag_pursuit_energy_target_speed_mps = 195.0
    cfg.lag_pursuit_energy_throttle_base = 0.55
    cfg.lag_pursuit_energy_throttle_gain = 0.015
    cfg.lag_pursuit_disable_defensive = True
    cfg.lag_pursuit_advantage_full_deg = 20.0
    cfg.lag_pursuit_scale_tau_s = 0.6
    cfg.turn_rudder_assist = 0.60
    cfg.high_bank_target_speed_mps = 195.0
    cfg.high_bank_dynamic_speed_enabled = True
    cfg.high_bank_dynamic_near_range_m = 1500.0
    cfg.high_bank_dynamic_far_range_m = 3000.0
    cfg.high_bank_dynamic_target_margin_mps = 10.0
    cfg.high_bank_dynamic_max_speed_mps = 265.0
    cfg.advantage_manager_enabled = True
    cfg.advantage_min_attack_ttc_s = 2.5
    cfg.terminal_track_min_threat_ata_deg = 40.0
    cfg.defensive_threat_ata_deg = 15.0
    cfg.defensive_range_m = 1600.0
    cfg.defensive_closure_mps = 0.0
    cfg.defensive_min_hold_s = 1.2
    cfg.defensive_vertical_escape = False
    cfg.defensive_escape_threat_ata_deg = 15.0
    cfg.defensive_escape_own_ata_deg = 20.0
    cfg.defensive_escape_range_m = 1600.0
    cfg.defensive_escape_gamma_deg = 20.0
    cfg.defensive_escape_switch_s = 1.8
    cfg.defensive_escape_floor_m = 2500.0
    cfg.planner3d_enabled = True
    cfg.planner3d_horizon_s = 2.5
    cfg.planner3d_turn_limit_degps = 16.0
    cfg.planner3d_target_turn_limit_degps = 18.0
    cfg.planner3d_gamma_deg = 24.0
    cfg.planner3d_target_gamma_deg = 22.0
    cfg.planner3d_manoeuvre_hold_s = 0.35
    cfg.planner3d_adaptive_horizon = True
    cfg.planner3d_reachable_target_envelope = True
    cfg.planner3d_emergency_replan = True
    cfg.planner3d_emergency_hold_s = 0.15
    cfg.planner3d_research_scoring = True
    cfg.throttle_min = 0.35
    return cfg


def build_w97_controller() -> IntegratedBFMController:
    """Build a fresh W97 controller identical to the live W97 profile."""
    return IntegratedBFMController(build_w97_config())
