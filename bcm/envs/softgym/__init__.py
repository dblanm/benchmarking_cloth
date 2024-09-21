from collections import OrderedDict

from bcm.envs.softgym.cloth_dyn_softgym import ClothFlattenEnv

from .normalized_env import MyNormalizedEnv


SOFTGYM_ENVS = OrderedDict({"ClothFlatten": ClothFlattenEnv})

env_arg_dict = {
    "ClothFlatten": {
        "observation_mode": "key_point",
        "action_mode": "none",
        "num_picker": 2,
        "render": False,
        "headless": True,
        "horizon": 100,
        "action_repeat": 8,
        "render_mode": "cloth",
        "num_variations": 1000,
        "use_cached_states": True,
        "deterministic": False,
    }
}


def get_cloth_softgym_env(env_name="ClothFlatten", **kwargs):
    if env_name in SOFTGYM_ENVS:
        env_kwargs = env_arg_dict[env_name]
        # Generate and save the initial states for running this environment
        # for the first time
        env_kwargs["use_cached_states"] = False
        env_kwargs["save_cached_states"] = False
        env_kwargs["num_variations"] = 1
        env_kwargs["render"] = False
        env_kwargs["headless"] = True

        kwargs.update(env_kwargs)

        return MyNormalizedEnv(SOFTGYM_ENVS[env_name](**kwargs))
    else:
        raise ValueError(f"Invalid environment {env_name}")
