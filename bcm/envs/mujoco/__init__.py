import gym
from gym.envs.registration import register


def get_cloth_mujoco_env(**kwargs):
    register(
        id="ClothMujocoEnv-v0",
        entry_point="bcm.envs.mujoco.cloth_dyn_mujoco:ClothMujocoEnv",
        max_episode_steps=300,
    )

    return gym.make("ClothMujocoEnv-v0", **kwargs)
