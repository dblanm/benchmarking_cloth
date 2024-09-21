import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_full_trajectory(
    dt, target, target_path, stabilise_steps, primitive, sim
):
    # The trajectory is composed out of # trajectories
    trajectory = []

    # Define the initial pose of the grippers and the initial robot pose
    action_0 = np.array([0.0, 0.0, 1.0, 0.5, 0.0, 1.0])

    # These distances are based on the real-world set-up
    if sim == "sofa":
        pose_1 = [0.02, 0.0, 1.0, 0.498, 0.0, 1.0]  # For SOFA
    else:
        pose_1 = [0.02, 0.0, 1.0, 0.498, 0.0, 1.0]

    # First part of the trajectory is for stabilising the cloth
    stable_steps = stabilise_steps

    for _ in range(stable_steps):
        trajectory.append(action_0)

    # Move the robot to the initial position
    pose_1_steps = stabilise_steps
    pose_diff = (pose_1 - action_0) / pose_1_steps
    for i in range(pose_1_steps):
        action = action_0 + pose_diff * i
        trajectory.append(action)

    # Stabilise the cloth again / keep the same position
    for _ in range(stable_steps):
        trajectory.append(action)

    # Finally we execute the quintic trajectory
    if primitive == "dynamic":
        traj_left, traj_right = generate_fling_trajectory(
            pose_1[:3],
            pose_1[3:],
            dt=dt,
            fling_height=pose_1[2] + 0.06,
            grasp_height=pose_1[2] - 0.51051,
        )
    elif primitive == "quasi_static":  # quasi-static trajectory
        traj_left, traj_right = generate_quasi_static_trajectory(
            pose_1[:3],
            pose_1[3:],
            dt=dt,
            middle_height=pose_1[2] - 0.05,
            touch_height=pose_1[2] - 0.31051,
        )
    else:
        raise ValueError(f"Primitive type {primitive} not supported")

    if target != "None":
        trajectory_steps = traj_left.shape[0]
    else:
        trajectory_steps = traj_left.shape[0]

    pre_traj_length = len(trajectory)

    for i in range(trajectory_steps):
        if i < traj_left.shape[0]:
            action = np.array(
                [
                    traj_left[i, 0],
                    traj_left[i, 1],
                    traj_left[i, 2],
                    traj_right[i, 0],
                    traj_right[i, 1],
                    traj_right[i, 2],
                ]
            )
        else:  # Once we finish the trajectory stay stable for cfg["max_steps"]
            action = np.array(
                [
                    traj_left[-1, 0],
                    traj_left[-1, 1],
                    traj_left[-1, 2],
                    traj_right[-1, 0],
                    traj_right[-1, 1],
                    traj_right[-1, 2],
                ]
            )
        trajectory.append(action)

    return trajectory, pre_traj_length


def get_xz_w_initial_conds(x0, xf, z0, zf, t0, tf, v0, vf, a0, af, dt):

    x_vec = np.array([x0, v0, a0, xf, vf, af])
    z_vec = np.array([z0, v0, a0, zf, vf, af])
    M = np.array(
        [
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
            [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
            [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
        ]
    )

    x_coeffs, _, _, _ = np.linalg.lstsq(M, x_vec, rcond=None)
    z_coeffs, _, _, _ = np.linalg.lstsq(M, z_vec, rcond=None)

    t = np.arange(start=t0, step=dt, stop=tf + dt)
    xnew = (
        x_coeffs[0]
        + x_coeffs[1] * (t - t0)
        + x_coeffs[2] * (t - t0) ** 2
        + x_coeffs[3] * (t - t0) ** 3
        + x_coeffs[4] * (t - t0) ** 4
        + x_coeffs[5] * (t - t0) ** 5
    )
    znew = (
        z_coeffs[0]
        + z_coeffs[1] * (t - t0)
        + z_coeffs[2] * (t - t0) ** 2
        + z_coeffs[3] * (t - t0) ** 3
        + z_coeffs[4] * (t - t0) ** 4
        + z_coeffs[5] * (t - t0) ** 5
    )

    return xnew, znew, x_coeffs, z_coeffs


def compute_x(
    pos_start, pos_back, pos_front, pos_center, pos_last, tf1, tf2, tf3, tf4, dt
):
    x0 = pos_start[0][0]
    xf = pos_back[0][0]
    z0 = pos_start[0][2]
    zf = pos_back[0][2]

    t0 = 0
    v0 = 0
    a0 = 0
    vf = -0.7
    af = 3.0

    xnew, _, _, _ = get_xz_w_initial_conds(x0, xf, z0, zf, t0, tf1, v0, vf, a0, af, dt)

    x0_2 = xf
    xf_2 = pos_front[0][0]
    z0_2 = zf
    zf_2 = pos_front[0][2]
    t0 = 0.0
    v0_2 = vf
    a0_2 = af
    vf_2 = 0.0
    af2 = -1.0

    xnew2, znew2, _, _ = get_xz_w_initial_conds(
        x0_2, xf_2, z0_2, zf_2, t0, tf2, v0_2, vf_2, a0_2, af2, dt
    )
    znew2[:] = zf_2

    x0_3 = xf_2
    xf_3 = pos_center[0][0]
    z0_3 = zf_2
    zf_3 = pos_center[0][2]
    t0 = 0.0
    v0_3 = vf_2
    a0_3 = af2
    vf_3 = -0.3
    af3 = -0.5

    xnew3, _, _, _ = get_xz_w_initial_conds(
        x0_3, xf_3, z0_3, zf_3, t0, tf3, v0_3, vf_3, a0_3, af3, dt
    )

    x0_4 = xf_3
    xf_4 = pos_last[0][0]
    z0_4 = zf_3
    zf_4 = pos_last[0][2]
    t0 = 0.0
    v0_4 = vf_3
    a0_4 = af3
    vf_4 = 0.0
    af4 = 0.0

    xnew4, _, _, _ = get_xz_w_initial_conds(
        x0_4, xf_4, z0_4, zf_4, t0, tf4, v0_4, vf_4, a0_4, af4, dt
    )

    return xnew, xnew2, xnew3, xnew4


def compute_z(pos_start, pos_back, pos_front, pos_center, pos_last, tf1, tf3, tf4, dt):
    z0 = pos_start[0][2]
    zf = pos_back[0][2]
    zf_2 = pos_front[0][2]
    z0_3 = zf_2
    zf_3 = pos_center[0][2]

    t0 = 0
    v0 = 0
    a0 = 0
    vf = 0.0
    af = 0.0

    znew, _, _, _ = get_xz_w_initial_conds(z0, zf, z0, zf, t0, tf1, v0, vf, a0, af, dt)

    vf_2 = 0.0
    af2 = 0.0
    v0_3 = vf_2
    a0_3 = af2
    vf_3 = 0.0
    af3 = 0.02

    znew3, _, _, _ = get_xz_w_initial_conds(
        z0_3, zf_3, z0_3, zf_3, t0, tf3, v0_3, vf_3, a0_3, af3, dt
    )

    z0_4 = zf_3
    zf_4 = pos_last[0][2]
    v0_4 = vf_3
    a0_4 = af3
    vf_4 = 0.0
    af4 = 0.0

    znew4, _, _, _ = get_xz_w_initial_conds(
        z0_4, zf_4, z0_4, zf_4, t0, tf4, v0_4, vf_4, a0_4, af4, dt
    )
    znew4[:] = zf_4

    return znew, znew3, znew4


def compute_angle(angle, tf1, tf2, tf3, dt):
    p0 = angle[0]
    p1 = angle[1]
    a0 = 0
    t0 = 0.0
    v0 = 0.0
    vf = 10.8
    af = 180.0

    pitch_1, _, _, _ = get_xz_w_initial_conds(
        p0, p1, p0, p0, t0, tf1, v0, vf, a0, af, dt
    )

    p2 = angle[2]
    t0 = 0.0
    v0_2 = vf
    a0_2 = af
    vf_2 = -10.0
    af2 = -180.0

    pitch_2, _, _, _ = get_xz_w_initial_conds(
        p1, p2, p0, p0, t0, tf2, v0_2, vf_2, a0_2, af2, dt
    )

    p3 = angle[3]
    t0 = 0.0
    v0_3 = vf_2
    a0_3 = af2
    vf_3 = 0.0
    af3 = 0.0

    pitch_3, _, _, _ = get_xz_w_initial_conds(
        p2, p3, p0, p0, t0, tf3, v0_3, vf_3, a0_3, af3, dt
    )

    return pitch_1, pitch_2, pitch_3


def compute_quaternion_traj(roll_traj, euler_angle):
    # input euler angle is in X, Y, Z
    quaternion = np.zeros((roll_traj.shape[0], 4))  # Should be X, Y , Z, W

    # First fix the roll trajectory. In the euler angles there should be a sign change between
    # -180 -> -220, making it
    fixed_roll_traj = roll_traj.copy()
    for i in range(fixed_roll_traj.shape[0]):
        if fixed_roll_traj[i] < -180:
            fixed_roll_traj[i] = 360 - fixed_roll_traj[i]

    for i in range(roll_traj.shape[0]):
        temp_euler = R.from_euler(
            "xyz", [fixed_roll_traj[i], euler_angle[1], euler_angle[2]], degrees=True
        )
        quat = temp_euler.as_quat()
        quaternion[i, :] = quat

    return quaternion


def compute_x_z_angle(
    pos_start, pos_back, pos_front, pos_center, pos_last, angle_steps, euler_angle, dt
):
    tf1 = 1.1
    tf2 = 1.5
    tf3 = 1.3
    tf4 = 0.5

    xnew, xnew2, xnew3, xnew4 = compute_x(
        pos_start, pos_back, pos_front, pos_center, pos_last, tf1, tf2, tf3, tf4, dt
    )

    znew, znew3, znew4 = compute_z(
        pos_start, pos_back, pos_front, pos_center, pos_last, tf1, tf3, tf4, dt
    )
    znew2 = xnew2.copy()
    znew2[:] = pos_front[0][2]

    pitch_1, pitch_2, pitch_3 = compute_angle(angle_steps, tf1, tf2, tf3, dt)
    pitch_4 = znew4.copy()
    pitch_4[:] = angle_steps[-1]

    x = np.concatenate((xnew[:-1], xnew2[:-1], xnew3[:-1], xnew4))
    z = np.concatenate((znew[:-1], znew2[:-1], znew3[:-1], znew4))
    roll_traj = np.concatenate((pitch_1[:-1], pitch_2[:-1], pitch_3[:-1], pitch_4))

    quaternion_traj = compute_quaternion_traj(roll_traj, euler_angle)

    # Create the quintic trajectory with the orientation
    quintic_traj = np.zeros(
        (xnew.shape[0] + xnew2.shape[0] + xnew3.shape[0] + xnew4.shape[0] - 3, 7)
    )
    quintic_traj[:, 0] = x
    quintic_traj[:, 1] = pos_front[0][1]
    quintic_traj[:, 2] = z
    quintic_traj[:, 3] = quaternion_traj[:, 0]
    quintic_traj[:, 4] = quaternion_traj[:, 1]
    quintic_traj[:, 5] = quaternion_traj[:, 2]
    quintic_traj[:, 6] = quaternion_traj[:, 3]

    tf = tf1 + tf2 + tf3 + tf4

    return quintic_traj, tf


def compute_z_quasistatic(pos_start, pos_center, pos_down, tf1, tf2, dt):
    z0 = pos_start[0][2]
    zf = pos_center[0][2]
    zf_2 = pos_down[0][2]

    t0 = 0
    v0 = 0
    a0 = 0
    vf = 0.0
    af = 0.0

    znew, _, _, _ = get_xz_w_initial_conds(z0, zf, z0, zf, t0, tf1, v0, vf, a0, af, dt)

    z0_2 = zf
    vf_2 = 0.0
    af2 = 0.0
    v0_3 = vf_2
    a0_3 = af2
    vf_3 = 0.0
    af3 = 0.02

    znew2, _, _, _ = get_xz_w_initial_conds(
        z0_2, zf_2, z0_2, zf_2, t0, tf2, v0_3, vf_3, a0_3, af3, dt
    )

    return znew, znew2


def compute_x_quasistatic(pos_start, pos_end, tf234, dt):
    x0 = pos_start[0][0]
    xf = pos_end[0][0]

    t0 = 0
    v0 = 0
    a0 = 0
    vf = 0.0
    af = 0.0

    xnew, _, _, _ = get_xz_w_initial_conds(
        x0, xf, x0, xf, t0, tf234, v0, vf, a0, af, dt
    )

    return xnew


def compute_x_z_quasi(pos_start, pos_center, pos_down, euler_angle, dt):
    tf1 = 1.0
    tf2 = 1.0
    tf3 = 6.0
    tf4 = 6.0
    tf34 = tf3 + tf4

    tf = tf1 + tf2 + tf3 + tf4

    xnew = compute_x_quasistatic(
        pos_start=pos_start, pos_end=pos_down, tf234=tf34, dt=dt
    )

    znew, znew2 = compute_z_quasistatic(
        pos_start=pos_start,
        pos_center=pos_center,
        pos_down=pos_down,
        tf1=tf1,
        tf2=tf3,
        dt=dt,
    )
    znew1 = np.zeros_like(znew)
    znew1[:] = znew[-1]

    xnew0 = znew.copy()
    xnew0[:] = pos_start[0][0]
    xnew1 = xnew0.copy()

    x = np.concatenate((xnew0[:-1], xnew1[:-1], xnew))
    z_aux = np.concatenate((znew[:-1], znew1[:-1], znew2))
    z = np.zeros_like(x)
    z[: z_aux.shape[0]] = z_aux[:]
    z[z_aux.shape[0] :] = pos_down[0][2]

    roll_traj = np.zeros_like(x)
    roll_traj[:] = euler_angle[0]

    quaternion_traj = compute_quaternion_traj(roll_traj, euler_angle)

    quintic_traj = np.zeros((x.shape[0], 7))  # For

    quintic_traj[:, 0] = x
    quintic_traj[:, 1] = pos_start[0][1]
    quintic_traj[:, 2] = z
    quintic_traj[:, 3] = quaternion_traj[:, 0]
    quintic_traj[:, 4] = quaternion_traj[:, 1]
    quintic_traj[:, 5] = quaternion_traj[:, 2]
    quintic_traj[:, 6] = quaternion_traj[:, 3]

    return quintic_traj, tf


def generate_fling_trajectory(
    initial_pos_left, initial_pos_right, fling_height, grasp_height, dt, modify_x=False
):
    """

    :param initial_pos_left: Initial position of the left picker. np.array, shape(3,)
    :param initial_pos_right: Initial position of the right picker. np.array, shape(3,)
    :param dt: dt of the simulator to compute the length of the trajectory points
    :param modify_x: whether to apply the trajectory in the X axis fixing the values of Y to the initial position.
    :return:
    """

    x_start = initial_pos_left[0]
    y_start = initial_pos_left[1]
    z_start = initial_pos_left[2]
    z_middle = fling_height
    z_end = grasp_height

    if modify_x:
        pos_start = [[x_start, y_start, z_start]]
        pos_back = [[x_start - 0.25, y_start, z_middle]]
        pos_front = [
            [x_start + 0.3, y_start, z_middle]
        ]  # It should go at least up to 0.4 be 0.64
        pos_center = [[x_start + 0.15, y_start, z_end]]
        pos_last = [[x_start - 0.1, y_start, z_end]]
    else:
        pos_start = [[y_start, x_start, z_start]]
        pos_back = [[y_start - 0.25, x_start, z_middle]]
        pos_front = [
            [y_start + 0.3, x_start, z_middle]
        ]  # It should go at least up to 0.4 be 0.64
        pos_center = [[y_start + 0.15, x_start, z_end]]
        pos_last = [[y_start - 0.1, x_start, z_end]]
    angle_steps = [-179.24297822, -140, -220, -140.0, -140.0]
    euler_angle = [-179.24297822, -2.93434324, 42.65659799]  # Defined as X, Y, Z

    quintic_traj, tf = compute_x_z_angle(
        pos_start,
        pos_back,
        pos_front,
        pos_center,
        pos_last,
        angle_steps,
        euler_angle,
        dt,
    )
    quintic_left = quintic_traj.copy()
    quintic_right = quintic_traj.copy()
    if modify_x:
        quintic_right[:, 1] = initial_pos_right[1]
    else:
        quintic_left[:, 1] = quintic_traj[:, 0]
        quintic_left[:, 0] = x_start
        quintic_right[:, 1] = quintic_traj[:, 0]
        quintic_right[:, 0] = initial_pos_right[0]

    return quintic_left, quintic_right


def generate_quasi_static_trajectory(
    initial_pose_left,
    initial_pose_right,
    dt,
    touch_height,
    middle_height,
    modify_x=False,
):
    """
    :param initial_pose_left: Initial position of the left picker. np.array, shape(3,)
    :param initial_pose_right: Initial position of the right picker. np.array, shape(3,)
    :param dt: dt of the simulator to compute the length of the trajectory points
    :param modify_x: whether to apply the trajectory in the X axis fixing the values of Y to the initial position.
    :return:
    """
    x_start = initial_pose_left[0]
    y_start = initial_pose_left[1]
    z_start = initial_pose_left[2]
    z_middle = middle_height
    z_end = touch_height

    if modify_x:
        pos_start = [[x_start, y_start, z_start]]
        pos_center = [[x_start, y_start, z_middle]]
        pos_down = [[x_start - 0.6, y_start, z_end]]
    else:
        pos_start = [[y_start, x_start, z_start]]
        pos_center = [[y_start, x_start, z_middle]]
        pos_down = [[y_start - 0.6, x_start, z_end]]
    euler_init = [-177.31653135, -4.27029547, -4.35228107]

    quintic_traj, tf = compute_x_z_quasi(
        pos_start, pos_center, pos_down, euler_init, dt
    )
    quintic_left = quintic_traj.copy()
    quintic_right = quintic_traj.copy()

    if modify_x:
        quintic_right[:, 1] = initial_pose_right[1]
    else:
        quintic_left[:, 0] = initial_pose_left[0]
        quintic_left[:, 1] = quintic_traj[:, 0]
        quintic_right[:, 1] = quintic_traj[:, 0]
        quintic_right[:, 0] = initial_pose_right[0]

    return quintic_left, quintic_right
