from __future__ import annotations

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class DogFightCallbacks(DefaultCallbacks):
    """RLlib callbacks that collect per-episode dogfight metrics.

    Metrics recorded in episode.custom_metrics (auto-aggregated by RLlib):
      Outcome    : win, loss, draw, timeout, crash
      Reward     : ep_reward_{step,pursuit,damage,safety,terminal}
      Tactical   : ep_wez_steps, ep_mean_distance, ep_min_distance,
                   ep_altitude_penalty_steps, initial/final ATA/AA,
                   headon_guard_fail
      Action     : action_{roll,pitch,rudder,throttle}_{mean,std},
                   action_saturation_rate
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, *, episode, **kwargs):
        self._episode_data(episode)["actions"] = []

    def on_episode_step(self, *, episode, **kwargs):
        action = self._last_action(episode)
        if action is not None:
            try:
                self._episode_data(episode)["actions"].append(
                    np.asarray(action, dtype=np.float32)
                )
            except Exception:
                pass

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        info = self._last_info(episode)
        if not info:
            return

        # ── Outcome ───────────────────────────────────────────────────────
        outcome = info.get("outcome", "other")
        for key in ("win", "loss", "draw", "timeout", "crash"):
            self._record_metric(episode, metrics_logger, key, float(outcome == key))

        # ── Reward components (cumulative episode totals) ─────────────────
        for key, val in info.get("ep_reward_components", {}).items():
            self._record_metric(
                episode, metrics_logger, f"ep_reward_{key}", float(val)
            )

        # ── Tactical metrics ──────────────────────────────────────────────
        self._record_metric(
            episode, metrics_logger, "ep_wez_steps", float(info.get("ep_wez_steps", 0))
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_mean_distance",
            float(info.get("ep_mean_distance", 0.0)),
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_min_distance",
            float(info.get("ep_min_distance", 0.0)),
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_altitude_penalty_steps",
            float(info.get("ep_altitude_penalty_steps", 0)),
        )
        # ── Residual-RL gate/damage-exchange metrics (only meaningful for
        # ResidualW53Env/ResidualW56Env; absent otherwise, so this is a no-op
        # for other envs) ───────────────────────────────────────────────────
        if "ep_gate_active_rate" in info:
            self._record_metric(
                episode,
                metrics_logger,
                "gate_active_rate",
                float(info.get("ep_gate_active_rate", 0.0)),
            )
        if "ownship_health" in info and "target_health" in info:
            # Cumulative episode damage, NOT the env's own info["ownship_damage"]/
            # ["target_damage"] fields -- those are per-control-tick deltas
            # (see single_agent_env.py's _advance_simulation_step_ratio), so at
            # episode end they reflect only the last tick, not the whole
            # episode, and are ~always 0 for a timeout ending mid-air.
            self._record_metric(
                episode,
                metrics_logger,
                "target_health_loss",
                float(1.0 - float(info.get("target_health", 1.0))),
            )
            self._record_metric(
                episode,
                metrics_logger,
                "ownship_health_loss",
                float(1.0 - float(info.get("ownship_health", 1.0))),
            )
        for key in (
            "initial_scenario_index",
            "initial_alpha_deg",
            "initial_ata_deg",
            "initial_aa_deg",
            "initial_distance_m",
            "final_ata_deg",
            "final_aa_deg",
        ):
            if key in info:
                self._record_metric(
                    episode, metrics_logger, key, float(info.get(key, 0.0))
                )
        self._record_metric(
            episode,
            metrics_logger,
            "headon_guard_fail",
            float(bool(info.get("headon_guard_fail", False))),
        )

        # ── Action distribution ───────────────────────────────────────────
        # Axis count/meaning depends on the env: DogFightWrapper's raw action
        # is 4-axis roll/pitch/rudder/throttle, but ResidualW53Env (2-axis)
        # and ResidualW56Env (3-axis) record their own RL residual action
        # here instead (see each env's step()) -- indexing a fixed 4-tuple of
        # names crashed the env-runner actor (IndexError, auto-restarted by
        # Ray) every time a residual episode actually completed.
        actions = self._episode_data(episode).get("actions", [])
        if actions:
            arr = np.stack(actions)  # (steps, num_axes)
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            num_axes = arr.shape[1]
            if num_axes == 4:
                axis_names = ("roll", "pitch", "rudder", "throttle")
            elif num_axes == 3:
                axis_names = ("residual_roll", "residual_pitch", "residual_throttle")
            elif num_axes == 2:
                axis_names = ("residual_roll", "residual_pitch")
            else:
                axis_names = tuple(f"axis{i}" for i in range(num_axes))
            for i, name in enumerate(axis_names):
                self._record_metric(
                    episode, metrics_logger, f"action_{name}_mean", float(means[i])
                )
                self._record_metric(
                    episode, metrics_logger, f"action_{name}_std", float(stds[i])
                )
            # Saturation: fraction of steps where any axis hits ±1
            self._record_metric(
                episode,
                metrics_logger,
                "action_saturation_rate",
                float(np.mean(np.abs(arr) >= 0.99)),
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _episode_data(episode) -> dict:
        """Return mutable per-episode storage for old and new RLlib APIs."""
        if hasattr(episode, "user_data"):
            return episode.user_data
        return episode.custom_data

    @staticmethod
    def _record_metric(episode, metrics_logger, key: str, value: float) -> None:
        """Record a metric through the callback API available in this RLlib version."""
        if hasattr(episode, "custom_metrics"):
            episode.custom_metrics[key] = value
        elif metrics_logger is not None:
            metrics_logger.log_value(("custom_metrics", key), value, reduce="mean")

    @staticmethod
    def _last_action(episode):
        """Retrieve last action, handling both old and new RLlib API."""
        try:
            return episode.last_action_for()
        except Exception:
            pass

        try:
            return episode.get_actions(-1)
        except Exception:
            return None

    @staticmethod
    def _last_info(episode) -> dict:
        """Retrieve last info dict, handling both old and new RLlib API."""
        try:
            return episode.last_info_for() or {}
        except TypeError:
            # New API: requires agent_id kwarg
            try:
                return episode.last_info_for(agent_id=None) or {}
            except Exception:
                pass
        except Exception:
            pass
        try:
            return episode.get_infos(-1) or {}
        except Exception:
            pass
        return {}
