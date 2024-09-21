import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch


def get_dynamic_target_cutoff(target_name, primitive):
    if primitive == "quasi_static":
        target = 420  # max length of trajectory!
    elif primitive == "dynamic":
        if target_name == "chequered_rag_0":
            target = 95
        elif target_name == "chequered_rag_1":
            target = 95
        elif target_name == "chequered_rag_2":
            target = 99
        elif target_name == "cotton_rag_0":
            target = 99
        elif target_name == "cotton_rag_1":
            target = 98
        elif target_name == "cotton_rag_2":
            target = 99
        elif target_name == "linen_rag_0":
            target = 90
        elif target_name == "linen_rag_1":
            target = 89
        elif target_name == "linen_rag_2":
            target = 89
        else:
            raise ValueError(f"Undefined target name {target_name}")
    else:
        raise ValueError(f"Undefined primitive {primitive}")

    return target


def map_error_metrics_dict(dict_input, dict_output, fabric):
    """
    :param dict_input: Dictionary with error values
    :param dict_output: dictionary to modify
    :param fabric:
    :return: dict_output with values for the fabric or contact
    """

    if fabric:
        cost_type = "_fabric"
    else:
        cost_type = "_contact"

    for key, value in dict_input.items():
        dict_output[key + cost_type] = value

    return dict_output


def plot_sim_and_target(sim_cloud, target_cloud, it_nr, output_path, elev=8, azim=13):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sim_cloud_npy = sim_cloud.numpy()[0]

    ax.scatter(
        sim_cloud_npy[:, 0],
        sim_cloud_npy[:, 1],
        sim_cloud_npy[:, 2],
        marker="*",
        color="r",
        label="Sim",
    )
    if target_cloud is not None:
        target_vertices = target_cloud.numpy()[0]
        ax.scatter(
            target_vertices[:, 0],
            target_vertices[:, 1],
            target_vertices[:, 2],
            marker="^",
            color="k",
            alpha=0.1,
            label="Real",
        )

    plt.legend()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([0.00, 1.1])

    if it_nr < 10:
        it_str = f"00{it_nr}"
    elif it_nr < 100:
        it_str = f"0{it_nr}"
    else:
        it_str = f"{it_nr}"

    plt.savefig(f"{output_path}/target_and_source_elev_{elev}_azim_{azim}_{it_str}.png")
    plt.close()


def load_targets(target_dir):
    return [
        np.asarray(o3d.io.read_point_cloud(os.path.join(target_dir, file_name)).points)
        for file_name in sorted(os.listdir(target_dir))
    ]


def depth_as_3d_array(depth, cmap="viridis"):
    min_depth = np.min(depth)
    max_depth = np.max(depth)
    depth = (depth - min_depth) / (max_depth - min_depth)
    return (plt.cm.get_cmap(cmap)(depth)[:, :, :3] * 255).astype(int)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def find_cloud_median(cloud):
    x_median = cloud[:, 0].median()
    y_median = cloud[:, 1].median()
    z_median = cloud[:, 2].median()

    return x_median, y_median, z_median


def draw_registration_result(source, target):

    source_temp = o3d.geometry.PointCloud()
    source_temp.points = o3d.utility.Vector3dVector(source.numpy()[0, :, :])
    target_temp = o3d.geometry.PointCloud()
    target_temp.points = o3d.utility.Vector3dVector(target.numpy()[0, :, :])

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
    )