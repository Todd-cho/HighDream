__all__ = ["DogFightEnv"]


def __getattr__(name):
    """Avoid loading the JSBSim training environment for UDP-only inference."""
    if name == "DogFightEnv":
        from .envs.single_agent_env import DogFightEnv
        return DogFightEnv
    raise AttributeError(name)
