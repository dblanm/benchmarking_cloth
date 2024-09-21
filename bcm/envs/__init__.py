from packaging import version


def get_env(cfg, *args, **kwargs):
    env_name = cfg.name
    kwargs.update(cfg)
    if "mujoco" in env_name:
        import mujoco

        if version.parse(mujoco.__version__) < version.parse("3"):
            is_v3 = False
            assert (
                env_name == "mujoco"
            ), "You have Mujoco < 3. Select the correct configuration: mujoco"
        else:
            is_v3 = True
            assert (
                env_name == "mujoco3"
            ), "You have Mujoco >= 3. Select the correct configuration: mujoco3"
        from .mujoco import get_cloth_mujoco_env

        return get_cloth_mujoco_env(*args, is_v3=is_v3, **kwargs)
    elif env_name == "bullet":
        from .bullet import get_cloth_pybullet_env

        return get_cloth_pybullet_env(*args, **kwargs)
    elif env_name == "softgym":
        from .softgym import get_cloth_softgym_env

        return get_cloth_softgym_env(*args, **kwargs)
    elif env_name == "sofa":
        from .sofa import get_cloth_sofa_env

        return get_cloth_sofa_env(*args, **kwargs)
    else:
        raise ValueError(f"Environment {env_name} not recognized.")
