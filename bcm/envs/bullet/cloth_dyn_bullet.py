import os

import gym
import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient
from bcm.envs.bullet.bullet_utils import get_mesh_data, load_deform_object
from gym.spaces import Box
from gym.utils import seeding
from sim_utils.pybullet import get_camera_transform_matrices

from ..utils import get_faces_regular_mesh, get_texture_file_name

DEFAULT_CAMERA_CONFIG = {
    "distance": -1.7106 - 0.063,
    "roll": -np.pi / 2,
    "pitch": 0.0,
    "yaw": -np.pi,
    "cameraTargetPosition": np.array([-0.15422 + 0.104, 0.0, 0.60807]),
    "upAxisIndex": 2,
}

DEFAULT_CAM_PROJECTION = {
    "projectionMatrix": (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0000200271606445,
        -1.0,
        0.0,
        0.0,
        -0.02000020071864128,
        0.0,
    )
}


class ClothBulletEnv(gym.Env):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 30,
    }

    def __init__(
        self,
        depth=False,
        pbd=False,
        params={},
        **kwargs,
    ):
        self.pbd = pbd
        self.bullet_render = (
            # True
            False  # By default let's not use the PyBullet rendering and use the camera
        )
        self.sim_gravity = -9.81
        # TODO Uncomment - Test without gravity
        # self.sim_gravity = 0.0

        # We could increase the sim frequency to have somewhat better results
        self.sim_freq = 100   # This should match MuJoCo frequency which is 0.01
        self.trajectory_dt = 1 / self.sim_freq
        self.frame_skip = 1  # Same as in MuJoCo

        self.assets_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../assets/"
        )

        # Set the cloth parameters
        if "target" in kwargs:
            self.cloth_texture = os.path.join(
                self.assets_path,
                "textures",
                get_texture_file_name(
                    target=kwargs["target"], assets_path=self.assets_path
                ),
            )
        else:
            self.cloth_texture = None

        self.cloth_scale = params["scale"]
        self.cloth_mass = params["mass"]
        self.deform_bending_stiffness = params["deform_bending_stiffness"]
        self.deform_damping_stiffness = params["deform_damping_stiffness"]
        self.deform_elastic_stiffness = params["deform_elastic_stiffness"]
        self.deform_friction_coeff = params["deform_friction_coeff"]

        self.cloth_obj = os.path.join(self.assets_path, "meshes/rag.obj")
        self.faces = get_faces_regular_mesh(n_rows=25, n_cols=25)
        self.cloth_id = None
        self.cloth_pos_init = np.array([-0.25, 0.0, 0.85]) * self.cloth_scale
        self.cloth_ori_init = np.array([0, 0, 0])
        self.pos_z_table = kwargs["real_setup"]["table"]["zmax"] * self.cloth_scale

        self._setup_pybullet()

        # Set the cam resolution
        self.cam_width = 320
        self.cam_height = 288
        self.depth = depth
        self.n_channels = 3

        shape = (self.cam_height, self.cam_width, self.n_channels)

        # If between 0-1 use np.float16
        self.observation_space = gym.spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=shape
        )

        self.action_space = gym.spaces.Box(
            low=-self.cloth_scale, high=self.cloth_scale, shape=(6,), dtype=float
        )

        self.render_mode = "rgb_array"

        DEFAULT_CAMERA_CONFIG["cameraTargetPosition"] *= self.cloth_scale
        DEFAULT_CAMERA_CONFIG["distance"] *= self.cloth_scale

        # Camera attributes
        self.cam_view_matrix = self.sim.computeViewMatrixFromYawPitchRoll(
            **DEFAULT_CAMERA_CONFIG
        )
        self.cam_projection_matrix = DEFAULT_CAM_PROJECTION["projectionMatrix"]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.grippers = {}
        self._reset_bullet()
        self._create_grippers()

        info = self._get_info()

        return self._get_obs(), info

    def _setup_pybullet(self):

        self.sim = bclient.BulletClient(
            connection_mode=pybullet.GUI if self.bullet_render else pybullet.DIRECT
        )

        if self.bullet_render:
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, True)

        self._reset_bullet()

    def _reset_bullet(self):
        if self.bullet_render:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, True)
            # TODO Uncomment
            # self.sim.resetDebugVisualizerCamera(DEFAULT_CAMERA_CONFIG)
            # if debug:
            #     res = self.sim.getDebugVisualizerCamera()
            #     print('Camera info for', DEFAULT_CAMERA_CONFIG)
            #     print('viewMatrix', res[2])
            #     print('projectionMatrix', res[3])
        if self.pbd:
            self.sim.resetSimulation()
        else:
            self.sim.resetSimulation(
                pybullet.RESET_USE_DEFORMABLE_WORLD
            )  # FEM deform sim
        self.sim.setGravity(0, 0, self.sim_gravity * self.cloth_scale)
        self.sim.setTimeStep(1.0 / self.sim_freq)

        # We load the default plane of PyBullet
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor_id = self.sim.loadURDF("plane.urdf")
        texture_id = self.sim.loadTexture(
            os.path.join(self.assets_path, "textures/wall.png")
        )
        self.sim.changeVisualShape(
            floor_id,
            -1,
            rgbaColor=[1, 1, 1, 1],
            textureUniqueId=texture_id,
        )

        # Load the cloth
        self.cloth_id = load_deform_object(
            sim=self.sim,
            obj_file_name=self.cloth_obj,
            texture_file_name=self.cloth_texture,
            scale=self.cloth_scale,
            mass=self.cloth_mass,
            init_pos=self.cloth_pos_init,
            init_ori=self.cloth_ori_init,
            bending_stiffness=self.deform_bending_stiffness,
            damping_stiffness=self.deform_damping_stiffness,
            elastic_stiffness=self.deform_elastic_stiffness,
            friction_coeff=self.deform_friction_coeff,
            self_collision=True,
            debug=False,
        )

        # Load a table
        self.boxes_id = self.sim.loadURDF(
            os.path.join(self.assets_path, "urdf/boxes.urdf"),
            globalScaling=self.cloth_scale,
        )
        boxes_texture_id = self.sim.loadTexture(
            os.path.join(self.assets_path, "textures/lightwood.png")
        )
        self.sim.changeVisualShape(
            self.boxes_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=boxes_texture_id
        )

        # Set the physics engine parameter
        self.sim.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

    @staticmethod
    def get_closest(init_pos, mesh, max_dist=None):
        """Find mesh points closest to the given point."""
        init_pos = np.array(init_pos).reshape(1, -1)
        mesh = np.array(mesh)
        num_to_pin = 1  # new pybullet behaves well with 1 vertex per anchor
        dists = np.linalg.norm(mesh - init_pos, axis=1)
        anchor_vertices = np.argpartition(dists, num_to_pin)[0:num_to_pin]
        if max_dist is not None:
            anchor_vertices = anchor_vertices[dists[anchor_vertices] <= max_dist]
        new_anc_pos = mesh[anchor_vertices].mean(axis=0)

        return new_anc_pos, anchor_vertices

    def _get_anchors_pose(self):
        _, mesh_vert_positions = get_mesh_data(self.sim, self.cloth_id)
        left_pos = mesh_vert_positions[self.left_anchor_vertex]
        right_pos = mesh_vert_positions[self.right_anchor_vertex]

        return left_pos, right_pos

    def _create_grippers(self):

        # Get the cloth mesh
        nr_vertices, mesh_vert_positions = get_mesh_data(self.sim, self.cloth_id)
        mesh = np.array(mesh_vert_positions)
        # When creating the grippers, the mesh is at [0.0, 0.0, 0.0].
        # Therefor the cloth vertex min=[-0.25, 0., -0.34999999], max=[0.25, 0., 0.34999999]
        # Get the upper left and upper right corners of the cloth position
        mesh_min = mesh.min(axis=0)
        mesh_max = mesh.max(axis=0)

        # Find the closest point in the mesh to the desired left and right position and define the cloth vertex id
        _, left_anchor_vertex = self.get_closest(
            [mesh_min[0], mesh_max[1], mesh_max[2]], mesh_vert_positions
        )
        _, right_anchor_vertex = self.get_closest(
            [mesh_max[0], mesh_max[1], mesh_max[2]], mesh_vert_positions
        )
        self.left_anchor_vertex = left_anchor_vertex[0]
        self.right_anchor_vertex = right_anchor_vertex[0]

        mesh_vert_positions = get_mesh_data(self.sim, self.cloth_id)[1]

        left_pos = mesh_vert_positions[self.left_anchor_vertex]
        right_pos = mesh_vert_positions[self.right_anchor_vertex]

        # Create the anchors
        self.cube_gripper_left = self.sim.loadURDF(
            os.path.join(self.assets_path, "urdf/cube_gripper_left.urdf"),
            basePosition=left_pos,
            globalScaling=self.cloth_scale,
        )
        self.sim.createSoftBodyAnchor(
            self.cloth_id, self.left_anchor_vertex, self.cube_gripper_left, 2
        )

        self.cube_gripper_right = self.sim.loadURDF(
            os.path.join(self.assets_path, "urdf/cube_gripper_right.urdf"),
            basePosition=right_pos,
            globalScaling=self.cloth_scale,
        )
        self.sim.createSoftBodyAnchor(
            self.cloth_id, self.right_anchor_vertex, self.cube_gripper_right, 2
        )

        # Create the initial position of the grippers
        self.left_gripper_pos, self.right_gripper_pos = self._get_anchors_pose()
        cubePos, cubeOrn = self.sim.getBasePositionAndOrientation(self.boxes_id)

    def _command_gripper_vel(self, action):
        left_tgt_pos = action[:3]
        right_tgt_pos = action[3:]
        mesh_vert_positions = get_mesh_data(self.sim, self.cloth_id)[1]

        left_pos = mesh_vert_positions[self.left_anchor_vertex]
        right_pos = mesh_vert_positions[self.right_anchor_vertex]
        # right_pos = self.sim.getJointState(self.cube_gripper_right, 0)

        # Compute the pos error
        left_pos_error = left_pos - left_tgt_pos
        right_pos_error = right_pos - right_tgt_pos

        # Let's use the same torque computation as for the franka
        stiffness = 1
        damping = 1
        left_force = -stiffness * left_pos_error - damping * (
            (left_pos_error - self.error_left) * self.sim_freq
        )
        right_force = -stiffness * right_pos_error - damping * (
            (right_pos_error - self.error_right) * self.sim_freq
        )

        self.error_left = left_pos_error
        self.error_right = right_pos_error

        self.sim.applyExternalForce(
            self.cube_gripper_left,
            -1,
            left_force.tolist(),
            [0, 0, 0],
            pybullet.LINK_FRAME,
        )
        self.sim.applyExternalForce(
            self.cube_gripper_right,
            -1,
            right_force.tolist(),
            [0, 0, 0],
            pybullet.LINK_FRAME,
        )

    def _set_action_space(self):
        self.action_space = Box(-1, 1, shape=(6,), dtype="float32")
        return self.action_space

    def _get_obs(self):
        return self.render(
            mode="rgb_array", width=self.cam_width, height=self.cam_height
        )

    def _get_info(self):
        info = {}
        _, vertices = get_mesh_data(self.sim, self.cloth_id)
        # Unscale vertices
        info["vertices"] = np.array(vertices) / self.cloth_scale
        info["faces"] = self.faces
        info["intr"], info["extr"] = get_camera_transform_matrices(
            width=self.cam_width,
            height=self.cam_height,
            view_matrix=self.cam_view_matrix,
            projection_matrix=self.cam_projection_matrix,
        )

        if self.depth:
            info["depth"] = self.render(
                mode="depth_array", width=self.cam_width, height=self.cam_height
            )
        return info

    def _step_grippers(self, action):
        """Compute the difference between the current gripper position and the target and move the grippers dx, dy, dz
        :param action: target (x, y, z)
        :return:
        """
        action_left = np.array(action[:3]) * self.cloth_scale
        action_right = np.array(action[3:]) * self.cloth_scale

        # Define the target position taking into account the initial position of the grippers
        tgt_left = action_left
        tgt_right = action_right

        # This looks better, but the problem is the sim gripper stiffness
        tgt_left[2] -= self.left_gripper_pos[2]
        tgt_right[2] -= self.right_gripper_pos[2]

        # The initial gripper pos is left=-0.25, right=0.25. If the action is kept as the original, the cloth
        # stretches too much. if we subtract the initial X position of the grippers, it bends
        # Therefore, we subtract only 0.2 instead of 0.25
        tgt_left[0] += 0.2 * self.cloth_scale
        tgt_right[0] -= 0.2 * self.cloth_scale

        # Step left and right joints 0=X, 1=Y, 2=Z
        for i in range(action_left.shape[0]):
            self.sim.setJointMotorControl2(
                bodyIndex=self.cube_gripper_left,
                jointIndex=i,
                targetPosition=tgt_left[i],
                controlMode=self.sim.POSITION_CONTROL,
                positionGain=1.0,
            )
            self.sim.setJointMotorControl2(
                bodyIndex=self.cube_gripper_right,
                jointIndex=i,
                targetPosition=tgt_right[i],
                controlMode=self.sim.POSITION_CONTROL,
                positionGain=1.0,
            )

        self.sim.stepSimulation()

    def step(self, action):
        self._step_grippers(action)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        reward = 0
        done = False
        info = self._get_info()
        info["rgb"] = observation
        info["mask"] = (
            pybullet.getCameraImage(
                width=self.cam_width,
                height=self.cam_height,
                flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                viewMatrix=self.cam_view_matrix,
                **DEFAULT_CAM_PROJECTION,
            )
            == 1
        )
        return observation, reward, done, False, info

    def render(self, mode="rgb_array", width=300, height=300):

        if self.bullet_render:
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, True)

        w, h, rgba_px, depth, _ = self.sim.getCameraImage(
            width=width,
            height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=self.cam_view_matrix,
            **DEFAULT_CAM_PROJECTION,
        )
        # If getCameraImage() returns a tuple instead of numpy array that
        # means that pybullet was installed without numpy being present.
        # Uninstall pybullet, then install numpy, then install pybullet.
        assert isinstance(rgba_px, np.ndarray), "Install numpy, then pybullet"
        if mode == "rgb_array":
            img = rgba_px[:, :, 0:3]
        elif mode == "depth_array":
            img = depth
        else:
            raise ValueError(f"Not implemented mode {mode}")
        return img

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
