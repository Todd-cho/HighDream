__all__ = ["DogFightEnv"]

def __getattr__(name):
    if name == "DogFightEnv":
        from .single_agent_env import DogFightEnv
        return DogFightEnv
    raise AttributeError(name)
