from copy import deepcopy

import numpy as np
import pyflex
from sim_utils.camera import get_rotation_matrix
from sim_utils.softgym import get_camera_transform_matrices
from softgym.utils.misc import quatFromAxisAngle

from ..utils import get_faces_regular_mesh
from .cloth_env import ClothEnv
from .picker_w_objects import PickerWithObjects


class ClothFlattenEnv(ClothEnv):
    def __init__(
        self,
        depth=False,
        cached_states_path="cloth_flatten_init_states.pkl",
        num_picker=2,
        num_objects=3,
        picker_radius=0.05,
        picker_threshold=0.005,
        particle_radius=0.00625,
        params={},
        **kwargs,
    ):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        self.cloth_params = params
        # Define the table position
        self.pos_table_z = 0.15
        super().__init__(**kwargs)
        self.num_objects = num_objects
        self.action_tool = PickerWithObjects(
            num_picker,
            picker_radius=0.001,
            particle_radius=particle_radius,
            picker_threshold=picker_threshold,
            picker_low=(-1.0, 0.0, -1.0),
            picker_high=(1.5, 1.5, 1.5),
            nr_objects=num_objects,
        )
        self.particle_radius = particle_radius
        self.action_space = self.action_tool.action_space
        self.picker_radius = picker_radius

        self.coord_change = np.eye(3)[:, [0, 2, 1]]
        self.inv_coord_change = np.linalg.inv(self.coord_change)

        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        self.return_depth = depth
        self.depth = None
        self.sim_frequency = 100  # Line main.cpp 686
        self.trajectory_dt = 1 / self.sim_frequency

        n_cols, n_rows = self.get_current_config()["cloth"]["ClothSize"]
        self.faces = get_faces_regular_mesh(n_rows=n_rows, n_cols=n_cols)
        assert self.action_tool.num_picker == 2  # Two drop points for this task

    def get_default_config(self):
        """Set the default config of the environment and load it to self.config

        Angles are three values in radians indicating:
            - Rotation along the vertical axis (np.pi front view and 0 side view)
            - Rotation along the horizontal axis (0 flat view and -np.pi / 2 top view)
            - Unused angle in z axis
        """
        angle = np.pi / 2  # front view

        cam_angles = np.array([0.0, np.pi / 2 - angle, 0.0])
        # cam_angles = np.array([0.0, angle, 0.0])
        cam_x_angle, cam_y_angle, _ = cam_angles
        print(
            f"x angle = {cam_x_angle}, y angle={cam_y_angle}, y_angle={-cam_y_angle - np.pi}"
        )
        matrix1 = get_rotation_matrix(-cam_x_angle, [0, 1, 0])
        matrix2 = get_rotation_matrix(-cam_y_angle - np.pi, [1, 0, 0])
        rotation_matrix = matrix2 @ matrix1
        dist_x = -0.15422 + 0.104
        dist_y = 0.60807 + 0.05
        dist_z = 1.7106 + 0.063 - 0.290
        lookat = np.array([dist_x, dist_y, dist_z])
        # The end
        d = 0.0
        shift = np.array([0.0, 0, d])
        cam_pos = lookat - np.linalg.inv(rotation_matrix) @ shift

        print(f"Cam pose is ={cam_pos}")
        assert np.allclose(shift, rotation_matrix @ lookat - rotation_matrix @ cam_pos)
        config = {
            "cloth": {
                "ClothPos": [0.0, 1.0, 0.0],
                "ClothSize": [
                    int(0.62 / self.cloth_particle_radius),  # The size in softgym is not correct
                    int(0.48 / self.cloth_particle_radius),  # particle radius = 0.00625
                ],
                # Cloth stiffness defined by stretch, bend and shear
                "ClothStiff": [
                    self.cloth_params["stretch"],
                    self.cloth_params["bend"],
                    self.cloth_params["shear"],
                ],
                "ClothMass": self.cloth_params["mass"],
            },
            # Lowerbox and upperbox set as mujoco_cloth.xml.
            # TODO: Do that programatically?
            # Note that softGym pose (y, z) <-> Mujoco (z, y)
            # Table is 80x160
            "upperbox": {  # all these will be overwritten by generate_env_variations
                "box_dist_x": 1.20,
                "box_dist_z": 2.00,
                "height": 0.05,
                # "pose_x": 0.225,
                # "pose_y": 0.175,
                "pose_x": 0.0,
                "pose_y": self.pos_table_z,
                "pose_z": 0,
            },
            "camera_name": "default_camera",
            "camera_params": {
                "default_camera": {
                    # "pos": np.array([1.7, 0.9, 0.9]),
                    "pos": cam_pos,
                    "angle": cam_angles,
                    "width": self.camera_width,
                    "height": self.camera_height,
                }
            },
            "flip_mesh": 0,
        }
        return config

    def close(self):
        pass

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config["cloth"]["ClothSize"]

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        curr_pos[:, 2] = x_low
        return curr_pos

    def _set_to_vertical(self, x_low, height_low):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        assert np.max(curr_pos[:, 1]) >= 0.5
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _sample_cloth_size(self):
        return np.random.randint(60, 100), np.random.randint(60, 100)

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img, depth = pyflex.render()
            width, height = (
                self.camera_params["default_camera"]["width"],
                self.camera_params["default_camera"]["height"],
            )
            img = img.reshape(height, width, 4)[
                ::-1, :, :3
            ]  # Need to reverse the height dimension
            self.depth = depth
            return img
        elif mode == "human":
            raise NotImplementedError

    def set_scene(self, config, state=None):
        if self.render_mode == "particle":
            render_mode = 1
        elif self.render_mode == "cloth":
            render_mode = 2
        elif self.render_mode == "both":
            render_mode = 3
        camera_params = config["camera_params"][config["camera_name"]]
        env_idx = 0 if "env_idx" not in config else config["env_idx"]

        scene_params = np.array(
            [
                *config["cloth"]["ClothPos"],
                *config["cloth"]["ClothSize"],
                *config["cloth"]["ClothStiff"],
                render_mode,
                *camera_params["pos"][:],
                *camera_params["angle"][:],
                camera_params["width"],
                camera_params["height"],
                config["cloth"]["ClothMass"],
                config["flip_mesh"],
            ]
        )
        pyflex.set_scene(env_idx, scene_params, 0)

        # After creating them let's fix the states
        # self.set_box_states(lower_config=config['lowerbox'],
        #                     upper_config=config['upperbox'])
        # Similar to the FluidEnv once we have set the pyflex scene
        # we create the rigid bodies
        if self.num_objects == 3:
            self.create_box(**config["upperbox"])
            # self.create_box(**config["lowerbox"])
            self.create_cylinder_around_box_edge(config["upperbox"])

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def create_cylinder_around_box_edge(self, box_config):
        """Create a small cylinder on the upper front facing edge of the box"""
        quat = quatFromAxisAngle([1, 0, 0], 0.0)
        # Radius of the cylinder
        radius = self.particle_radius
        # Length and pose of the cylinder
        halfheight = box_config["box_dist_x"]
        pose_x, pose_y, pose_z = (
            box_config["pose_x"],
            box_config["pose_y"] + box_config["height"],
            box_config["pose_z"] - box_config["box_dist_z"],
        )
        pyflex.add_capsule([radius, halfheight], [pose_x, pose_y, pose_z], quat)

        # Create second cylinder
        pose_x, pose_y, pose_z = (
            box_config["pose_x"],
            box_config["pose_y"] + box_config["height"],
            box_config["pose_z"] + box_config["box_dist_z"],
        )
        pyflex.add_capsule([radius, halfheight], [pose_x, pose_y, pose_z], quat)

    def create_box(self, box_dist_x, box_dist_z, height, pose_x, pose_y, pose_z):
        pose_centered = np.array([pose_x, pose_y, pose_z])
        # TODO: Added slight tilting to allow cloth to slide
        # Remove if friction of chair fixed
        quat = quatFromAxisAngle([1, 0, 0], 0.0)
        halfEdge = np.array([box_dist_x, height, box_dist_z])
        pyflex.add_box(halfEdge, pose_centered, quat)

    def _get_key_point_idx(self):
        """The keypoints are defined as the four corner points of the cloth"""
        dimx, dimy = self.current_config["cloth"]["ClothSize"]
        idx_p1 = 0
        idx_p2 = dimx * (dimy - 1)
        idx_p3 = dimx - 1
        idx_p4 = dimx * dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4])

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True):
        """Generate initial states. Note: This will also change the current states!"""
        # TODO: Remove, just for debugging
        vary_cloth_size = False

        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = (
            0.1  # Cloth stable when all particles' vel are smaller than this
        )
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(
                config["camera_name"], config["camera_params"][config["camera_name"]]
            )
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config["cloth"]["ClothSize"] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config["cloth"]["ClothSize"]
            self.set_scene(config)
            self.action_tool.reset([0.0, -1.0, 0.0])

            pickpoints = self._get_drop_point_idx()[
                :2
            ]  # Pick two corners of the cloth and wait until stablize

            # Hang cloth vertically
            self._set_to_vertical(
                x_low=-0.2,
                height_low=0.4,
            )

            # Get height of the cloth without the gravity
            # With gravity, it will be longer
            p1, _, p2, _ = self._get_key_point_idx()

            curr_pos = pyflex.get_positions().reshape(-1, 4)
            curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            original_inv_mass = curr_pos[pickpoints, 3]
            # Set mass of the pickup point to infinity so that it generates
            # enough force to the rest of the cloth
            curr_pos[pickpoints, 3] = 0
            pickpoint_pos = curr_pos[pickpoints, :3]
            pyflex.set_positions(curr_pos.flatten())

            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(
                picker_pos=pickpoint_pos + np.array([0.0, picker_radius, 0.0])
            )

            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
        return generated_configs, generated_states

    def _reset(self):
        """Right now only use one initial state"""
        if hasattr(self, "action_tool"):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.5, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(
                picker_pos=drop_point_pos + np.array([0.0, picker_radius, 0.0])
            )
            # self.action_tool.visualize_picker_boundary()
        info = self._get_info()
        return self._get_obs(), info

    def _coords_to_softgym(self, coords):
        return coords @ self.coord_change

    def _coords_from_softgym(self, coords):
        return coords @ self.inv_coord_change

    def _step(self, action):
        action = self._coords_to_softgym(np.array(action).reshape((2, 3)))

        self.action_tool.step(action, lambda: pyflex.step())
        return

    def step(self, action, record_continuous_video=False, img_size=None):
        """If record_continuous_video is set to True
        will record an image for each sub-step"""
        frames = []

        for i in range(self.action_repeat):
            # print(f"Doing action for {i}, repeats for {self.action_repeat}")
            self._step(action)
            if record_continuous_video and i % 2 == 0:  # No need to record each step
                frames.append(self.get_image(img_size, img_size))

        obs = self._get_obs()
        # TODO: Set a reward?
        reward = 0
        info = self._get_info()

        if self.recording:
            self.video_frames.append(self.render(mode="rgb_array"))
        self.time_step += 1

        # Added truncated for compatibility with Gym env.step(action)
        # truncated = True if horizon exceeded
        done = False
        truncated = False
        if self.time_step >= self.horizon:
            done = True
            truncated = True
        if record_continuous_video:
            info["flex_env_recorded_frames"] = frames
        # Add rgb (in this case observation) to info
        info["rgb"] = obs
        if self.return_depth:
            info["depth"] = self.depth.reshape(*obs.shape[:2])[::-1]
        return obs, reward, done, truncated, info

    def _get_info(self):
        info = {
            "vertices": self._coords_from_softgym(
                pyflex.get_positions().reshape(-1, 4)[:, :3]
            ),
            "faces": self.faces,
        }
        info["intr"], info["extr"] = get_camera_transform_matrices(self)
        return info
