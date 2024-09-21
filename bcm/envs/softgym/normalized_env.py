import numpy as np

from softgym.utils.normalized_env import NormalizedEnv
from softgym.utils.overrides import overrides


class MyNormalizedEnv(NormalizedEnv):
    @overrides
    def step(self, action, **kwargs):
        wrapped_step = self._wrapped_env.step(action, **kwargs)
        next_obs, reward, done, truncated, info = wrapped_step
        if self._clip_obs is not None:
            next_obs = np.clip(next_obs, self._clip_obs[0], self._clip_obs[1])

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
        return next_obs, reward * self._scale_reward, done, truncated, info
