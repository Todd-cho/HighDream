"""AI package with lazy public imports for lightweight UDP inference."""
from importlib import import_module

_EXPORTS = {
    "ActionContext": ("dogfight.ai.action_provider", "ActionContext"),
    "ActionProvider": ("dogfight.ai.action_provider", "ActionProvider"),
    "ActionResult": ("dogfight.ai.action_provider", "ActionResult"),
    "AIPilot": ("dogfight.ai.native_bt", "AIPilot"),
    "BTActionProvider": ("dogfight.ai.bt_action_provider", "BTActionProvider"),
    "activate_rule_xml": ("dogfight.ai.bt_rule_manager", "activate_rule_xml"),
    "HybridActionProvider": ("dogfight.ai.hybrid_action_provider", "HybridActionProvider"),
    "RLActionProvider": ("dogfight.ai.rl_action_provider", "RLActionProvider"),
    "build_algorithm_config": ("dogfight.ai.rllib_utils", "build_algorithm_config"),
    "build_algorithm_from_bundle": ("dogfight.ai.rllib_utils", "build_algorithm_from_bundle"),
    "normalize_algorithm_name": ("dogfight.ai.rllib_utils", "normalize_algorithm_name"),
    "load_lightweight_policy_bundle": ("dogfight.ai.checkpoint_io", "load_lightweight_policy_bundle"),
    "save_lightweight_policy_bundle": ("dogfight.ai.checkpoint_io", "save_lightweight_policy_bundle"),
    "DashboardJsonlLogger": ("dogfight.ai.dashboard_logger", "DashboardJsonlLogger"),
    "copy_experiment_yaml": ("dogfight.ai.dashboard_logger", "copy_experiment_yaml"),
    "load_experiment_metadata": ("dogfight.ai.dashboard_logger", "load_experiment_metadata"),
    "training_row_to_dashboard_metrics": ("dogfight.ai.dashboard_logger", "training_row_to_dashboard_metrics"),
    "save_training_record": ("dogfight.ai.training_record", "save_training_record"),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
