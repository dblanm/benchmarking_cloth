import numpy as np

import pyflex
import scipy.spatial
from gym.spaces import Box
from softgym.action_space.action_space import Picker


class PickerWithObjects(Picker):
    def __init__(
        self,
        num_picker=1,
        picker_low=(-0.4, 0.0, -0.4),
        picker_high=(0.4, 0.5, 0.4),
        nr_objects=0,
        unbounded=True,
        **kwargs,
    ):
        """

        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super().__init__(
            num_picker=num_picker,
            picker_low=picker_low,
            picker_high=picker_high,
            **kwargs,
        )
        self.nr_objects = nr_objects
        self.unbounded = unbounded
        space_low = (
            np.array([-0.1, -0.1, -0.1] * self.num_picker) * 0.1
        )  # [dx, dy, dz, [0, 1]]
        space_high = np.array([0.1, 0.1, 0.1] * self.num_picker) * 0.1
        self.action_space = Box(space_low, space_high, dtype=np.float32)
        self.delta_move = 1.0
        self.steps_limit = 1
        # self.first_action = None

    def _apply_picker_boundary(self, picker_pos):
        if self.unbounded:
            return picker_pos
        else:
            clipped_picker_pos = picker_pos.copy()
            for i in range(3):
                clipped_picker_pos[i] = np.clip(
                    picker_pos[i],
                    self.picker_low[i] + self.picker_radius,
                    self.picker_high[i] - self.picker_radius,
                )
            return clipped_picker_pos

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.0
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        for i in (0, 2):
            offset = center[i] - (self.picker_high[i] + self.picker_low[i]) / 2.0
            self.picker_low[i] += offset
            self.picker_high[i] += offset
        init_picker_poses = self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
        pos = (
            pyflex.get_shape_states()
        )  # Need to call this to update the shape collision
        reshaped_pos = np.array(pos).reshape(-1, 14)
        pyflex.set_shape_states(pos)
        if reshaped_pos.shape[0] > self.num_picker:
            self.nr_objects = reshaped_pos.shape[0] - self.num_picker

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i + self.nr_objects] = np.hstack(
                [centered_picker_pos, centered_picker_pos, [1, 0, 0, 0], [1, 0, 0, 0]]
            )
        pyflex.set_shape_states(shape_state)
        # Remove this as having an additional step here may affect the cloth drop env
        # pyflex.step()
        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]
        # print('inv_mass_shape after reset:', self.particle_inv_mass.shape)

    @staticmethod
    def _get_pos():
        """Get the current pos of the pickers and the particles,
        along with the inverse mass of each particle"""
        # When creating two boxes this is (4, 14) instead of 2, 14
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    def set_picker_pos(self, picker_pos):
        """Caution! Should only be called during the reset of the environment.
        Used only for cloth drop environment."""
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[self.nr_objects :, 3:6] = picker_pos
        shape_states[self.nr_objects :, :3] = picker_pos
        pyflex.set_shape_states(shape_states)

    def step_picker(self, action):
        """action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one, for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        # Use closed grippers, only position provided
        action = action.copy().reshape(-1, 3)

        pick_flag = [True, True]
        picker_pos, particle_pos = self._get_pos()
        new_picker_pos, new_particle_pos = picker_pos.copy(), particle_pos.copy()

        # Un-pick the particles
        # print(
        #     "check pick id:",
        #     self.picked_particles,
        #     new_particle_pos.shape,
        #     self.particle_inv_mass.shape,
        # )
        for i in range(self.num_picker):
            if not pick_flag[i] and self.picked_particles[i] is not None:
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[
                    self.picked_particles[i]
                ]  # Revert the mass
                self.picked_particles[i] = None

        # Pick new particles and update the mass and the positions
        for i in range(self.num_picker):
            new_picker_pos[i + self.nr_objects, :] = self._apply_picker_boundary(
                picker_pos[i + self.nr_objects, :] + action[i, :3]
            )
            # print(f"New picker position={new_picker_pos}")
            if pick_flag[i]:
                if self.picked_particles[i] is None:
                    # No particle is currently picked and thus need to
                    # select a particle to pick
                    dists = scipy.spatial.distance.cdist(
                        picker_pos[i + self.nr_objects].reshape((-1, 3)),
                        particle_pos[:, :3].reshape((-1, 3)),
                    )
                    idx_dists = np.hstack(
                        [
                            np.arange(particle_pos.shape[0]).reshape((-1, 1)),
                            dists.reshape((-1, 1)),
                        ]
                    )
                    mask = (
                        dists.flatten()
                        <= self.picker_threshold
                        + self.picker_radius
                        + self.particle_radius
                    )
                    idx_dists = idx_dists[mask, :].reshape((-1, 2))
                    if idx_dists.shape[0] > 0:
                        pick_id, pick_dist = None, None
                        for j in range(idx_dists.shape[0]):
                            if idx_dists[j, 0] not in self.picked_particles and (
                                pick_id is None or idx_dists[j, 1] < pick_dist
                            ):
                                pick_id = idx_dists[j, 0]
                                pick_dist = idx_dists[j, 1]
                        if pick_id is not None:
                            self.picked_particles[i] = int(pick_id)

                if self.picked_particles[i] is not None:
                    # TODO The position of the particle needs to be updated such that
                    # it is close to the picker particle
                    new_particle_pos[self.picked_particles[i], :3] = (
                        particle_pos[self.picked_particles[i], :3]
                        + new_picker_pos[i + self.nr_objects, :]
                        - picker_pos[i + self.nr_objects, :]
                    )
                    new_particle_pos[
                        self.picked_particles[i], 3
                    ] = 0  # Set the mass to infinity

        # check for e.g., rope, the picker is not dragging the particles too far away
        # that violates the actual physicals constraints.
        if self.init_particle_pos is not None:
            picked_particle_idices = []
            active_picker_indices = []
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    picked_particle_idices.append(self.picked_particles[i])
                    active_picker_indices.append(i + self.nr_objects)

            l = len(picked_particle_idices)
            for i in range(l):
                for j in range(i + 1, l):
                    init_distance = np.linalg.norm(
                        self.init_particle_pos[picked_particle_idices[i], :3]
                        - self.init_particle_pos[picked_particle_idices[j], :3]
                    )
                    now_distance = np.linalg.norm(
                        new_particle_pos[picked_particle_idices[i], :3]
                        - new_particle_pos[picked_particle_idices[j], :3]
                    )
                    if (
                        now_distance >= init_distance * self.spring_coef
                    ):  # if dragged too long, make the action has no effect; revert it
                        new_picker_pos[active_picker_indices[i], :] = picker_pos[
                            active_picker_indices[i], :
                        ].copy()
                        new_picker_pos[active_picker_indices[j], :] = picker_pos[
                            active_picker_indices[j], :
                        ].copy()
                        new_particle_pos[picked_particle_idices[i], :3] = particle_pos[
                            picked_particle_idices[i], :3
                        ].copy()
                        new_particle_pos[picked_particle_idices[j], :3] = particle_pos[
                            picked_particle_idices[j], :3
                        ].copy()

        self._set_pos(new_picker_pos, new_particle_pos)

    def step(self, action, step_sim_fn=lambda: pyflex.step()):
        """
        action: Array of pick_num x 4. For each picker,
         the action should be [x, y, z, pick/drop].
        The picker will then first pick/drop, and keep
         the pick/drop state while moving towards x, y, x.
        """
        total_steps = 0

        curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[
            self.nr_objects :, :3
        ]
        end_pos = np.vstack([picker_pos for picker_pos in action[:, :3]])

        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        for i in range(int(min(num_step, self.steps_limit))):
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[
                self.nr_objects :, :3
            ]
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.alltrue(dist < norm_delta):
                delta = end_pos - curr_pos
            self.step_picker(delta)
            step_sim_fn()
            total_steps += 1
            if np.alltrue(dist < self.delta_move):
                break
        return total_steps
