import os
from typing import Optional

import mujoco

import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from gym.utils import EzPickle
from sim_utils.mujoco import get_camera_transform_matrices

from ..utils import get_texture_file_name
from .utils import XMLModel

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class ClothMujocoEnv(MujocoEnv, EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        is_v3,
        width=480,
        height=480,
        depth=False,
        name=None,
        params={},
        real_setup={},
        target=None,
        **kwargs,
    ):
        self.depth = depth

        observation_space = Box(
            low=0, high=255, dtype=np.uint8, shape=(height, width, 3)
        )

        if is_v3:
            # frame_skip = 10
            # self.metadata["render_fps"] = 10
            frame_skip = 1
            self.metadata["render_fps"] = 100
        else:
            frame_skip = 1

        # self.render_mode = "rgb_array"
        self.render_mode = None

        self.assets_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../assets/"
        )
        if is_v3:
            mujoco_model = XMLModel(
                os.path.join(self.assets_path, "mujoco3_cloth.xml"), is_v3=is_v3
            )
        else:
            mujoco_model = XMLModel(
                os.path.join(self.assets_path, "mujoco_cloth.xml"), is_v3=is_v3
            )

        if target is not None:
            mujoco_model.change_texture(
                texture_file=os.path.join(
                    "textures",
                    get_texture_file_name(target=target, assets_path=self.assets_path),
                )
            )

        mujoco_model.modify_params(params)

        MujocoEnv.__init__(
            self,
            model_path=mujoco_model.path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            width=width,
            height=height,
            **kwargs,
        )
        self.trajectory_dt = self.dt

        print(f" camera width ={self.width} camera height={self.height}")
        self.bullet_render = True
        self.og_cam_xpos = None
        self.xpos_ids, self.faces = mujoco_model.get_mesh_ids(self.model)

    def _set_action_space(self):
        self.action_space = Box(-1, 1, shape=(6,), dtype=float)
        return self.action_space

    def _get_info(self, reset=False):
        info = {}
        if not reset:
            if self.render_mode == "human":
                self.render()
            elif self.render_mode == "rgb_array":
                # TODO: Can we somehow change the camera only once in viewer setup?
                if self.og_cam_xpos is None:
                    self.og_cam_xpos = self.data.cam_xpos.copy()
                self.data.cam_xpos = self.og_cam_xpos + np.array(
                    # The Z + 0.2 is to set-up both the camera, robots and table higher than the ground
                    [[-0.15422 + 0.104, 1.7106 + 0.063, 0.60807 + 0.2]]
                )

                if self.depth:
                    self._get_viewer("depth_array").render(camera_id=0)
                    reversed_rgb, reversed_depth = self._get_viewer(
                        "depth_array"
                    ).read_pixels(depth=True)
                    info["depth"] = reversed_depth[::-1, :]
                else:
                    self._get_viewer("rgb_array").render(camera_id=0)
                    reversed_rgb = self._get_viewer("rgb_array").read_pixels(
                        depth=False
                    )

                info["rgb"] = reversed_rgb[::-1, :, :]
            else:
                pass
        intr, extr = get_camera_transform_matrices(
            self.width,
            self.height,
            self.model.cam_fovy[0],
            self.data.cam_xpos,
            self.data.cam_xmat,
        )
        info["intr"] = intr
        info["extr"] = extr
        info["vertices"] = self.data.xpos[self.xpos_ids]
        info["faces"] = self.faces
        return info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, _ = super(MujocoEnv, self).reset(seed=seed, options=options)
        info = self._get_info(reset=True)
        return obs, info

    def step(self, action):
        # Action should be dx dy dz of the mocap
        self.do_simulation(action, self.frame_skip)

        reward = 0
        done = False
        info = self._get_info()
        if self.render_mode == "None":
            observation = None
        else:
            observation = info["rgb"]

        return observation, reward, done, False, info

    def _step_mujoco_simulation(self, ctrl, n_frames):
        # ctrl action should be dx, dy, dz
        self.data.mocap_pos[1][:] = ctrl[:3]
        self.data.mocap_pos[2][:] = ctrl[3:]

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        mujoco.mj_rnePostConstraint(self.model, self.data)

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        info = self._get_info()
        if self.render_mode == "None":
            return None
        observation = info["rgb"]
        return observation
