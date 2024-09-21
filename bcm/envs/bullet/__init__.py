import gym
from gym.envs.registration import register


def get_cloth_pybullet_env(**kwargs):
    register(
        id="ClothBulletEnv-v0",
        entry_point="bcm.envs.bullet.cloth_dyn_bullet:ClothBulletEnv",
        max_episode_steps=300,
    )

    return gym.make("ClothBulletEnv-v0", **kwargs)
