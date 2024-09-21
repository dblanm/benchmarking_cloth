import os
import sys
import pickle
import matplotlib
import numpy as np
import pandas as pd
# import seaborn as sns

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from argparse import ArgumentParser

from filelock import FileLock
from pprint import pprint
from PIL import Image

matplotlib.use('TkAgg')

curr_dir = os.getcwd()
if curr_dir != '/home/mulerod1/projects/dynamic_dual_manip/':
    os.chdir('/home/mulerod1/projects/dynamic_dual_manip/')
print(f"Current working directory is {os.getcwd()}")


import open3d as o3d
import torch
import point_cloud_utils as pcu



def load_targets(target_dir):
    return [
        np.asarray(o3d.io.read_point_cloud(os.path.join(target_dir, file_name)).points)
        for file_name in sorted(os.listdir(target_dir))
    ]

nr_decimals=3

def get_target_and_sim_cloud(sim_cloud_traj, timestep, skip_frames):
    # Load the target
    freq_relation = sim_cloud_traj.shape[0] / len(target_tgt)

    # target_comparison = (60 + skip_frames) * freq_relation

    iteration_nr = int(timestep * freq_relation)

    # target_idx = int(target_it[iteration_nr] - skip_frames)
    target_idx = int(iteration_nr / freq_relation - skip_frames)
    # print(f"Target idx={target_idx}, target compare = {target_comparison}")
    if target_idx < 110:
        # print(f"Free dynamics")
        fabric_w_contact = False
    else:
        # print(f"Contact")
        fabric_w_contact = True
    # print(f"Freq relation={freq_relation}, iteration = {iteration_nr} target idx={target_idx}")

    target_cloud = torch.from_numpy(target_tgt[target_idx]).unsqueeze(0).float()

    # transformed_target = tranform_target_to_mujoco(target_cloud)
    target_vertices = target_cloud.numpy()[0]

    # Select the source cloud to match
    # source_nr = target_it[iteration_nr]
    sim_cloud_npy = sim_cloud_traj[iteration_nr]
    sim_cloud_torch = torch.from_numpy(sim_cloud_npy).unsqueeze(0)

    # print(f"Shape sim = {sim_cloud_torch.shape} taret={transformed_target.shape}")
    # cd_cost_l1 = \
    # chamfer_distance_w_unidirectional(sim_cloud_torch.type(torch.FloatTensor), target_cloud.type(torch.FloatTensor),
    #                                   single_directional=True, norm=1)[0]
    if sim_cloud_npy.shape[0] < target_vertices.shape[0]:
        first_sim_cloud = target_vertices.copy()
        first_sim_cloud[:sim_cloud_npy.shape[0], :] = sim_cloud_npy
        first_sim_cloud[sim_cloud_npy.shape[0]:, :] = np.nan
        sim_cloud_npy = first_sim_cloud[~np.isnan(first_sim_cloud[:, 0]), :]
    else:
        first_sim_cloud = sim_cloud_npy.copy()
        first_sim_cloud[:target_vertices.shape[0], :] = target_vertices
        first_sim_cloud[target_vertices.shape[0]:, :] = np.nan
        target_vertices = first_sim_cloud[~np.isnan(first_sim_cloud[:, 0]), :]
    # hd_p1_to_p2 = pcu.one_sided_hausdorff_distance(sim_cloud_npy, target_vertices, return_index=False)
    # print(f"Sim = {sim}, nr points = {sim_cloud_npy.shape[0]}, target points={target_vertices.shape[0]}")
    # print(f"Sim = {sim}, cd_cost_l1 = {cd_cost_l1}, hd_p1_to_p2={hd_p1_to_p2}, iteration = {iteration_nr} target idx={target_idx}")

    # Reduce the number of points so that they can be plotted
    source_temp = o3d.geometry.PointCloud()
    source_temp.points = o3d.utility.Vector3dVector(sim_cloud_npy)
    downpcd = source_temp.voxel_down_sample(voxel_size=0.02)

    min_voxel_size = 0.025
    voxel_step = 0.005
    target_shape = 1000

    if np.asarray(downpcd.points).shape[0] > target_shape:
        downpcd = downpcd.voxel_down_sample(voxel_size=min_voxel_size + voxel_step)

    sim_cloud_reduced = np.asarray(downpcd.points)

    target_temp = o3d.geometry.PointCloud()
    target_temp.points = o3d.utility.Vector3dVector(target_vertices)
    # 0.03 500 - 1k, 1k4
    # 0.02 = 1-2k points, 2k
    # 0.01 = ~5-6k

    # print(f"Target size={target_vertices.shape}")

    target_downpcd = target_temp.voxel_down_sample(voxel_size=min_voxel_size)
    if np.asarray(target_downpcd.points).shape[0] > target_shape:
        target_downpcd = target_temp.voxel_down_sample(voxel_size=min_voxel_size + voxel_step)
        if np.asarray(target_downpcd.points).shape[0] > target_shape:
            target_downpcd = target_temp.voxel_down_sample(voxel_size=min_voxel_size + voxel_step * 2)
        if np.asarray(target_downpcd.points).shape[0] > target_shape:
            target_downpcd = target_temp.voxel_down_sample(voxel_size=min_voxel_size + voxel_step * 3)
            if np.asarray(target_downpcd.points).shape[0] > target_shape:
                target_downpcd = target_temp.voxel_down_sample(voxel_size=min_voxel_size + voxel_step * 4)

    target_cloud_reduced = np.asarray(target_downpcd.points)
    # print(f"Target reduced size={target_cloud_reduced.shape}")

    # print(f"Reduced cloud sim nr points = {sim_cloud_reduced.shape[0]}, target points={target_cloud_reduced.shape[0]}")

    cd_cost_l1 = 0.0
    hd_p1_to_p2 = 0.0

    # return target_cloud_reduced, sim_cloud_reduced, cd_cost_l1, hd_p1_to_p2
    return target_cloud_reduced, sim_cloud_npy, cd_cost_l1, hd_p1_to_p2, fabric_w_contact

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def plot_and_save_cloth_sequence(cloth_sequence, sim_name, out_folder, primitive: str, elev:int, azim:int,
                                 plot=True):
    print(f"Evaluating primitive type={primitive}")
    if primitive == 'dynamic':
        real_time = np.around(np.linspace(start=0.0, stop=4.4, num=121), decimals=2)
    else:
        real_time = np.around(np.linspace(start=0.0, stop=14, num=420), decimals=2)
    skip_frames = 0

    cd_f_list = []
    cd_c_list = []
    hd_f_list = []
    hd_c_list = []

    # elev = 0
    # elev = 8
    # azim = 0
    # azim = 13
    # elev = 0
    # azim = 90

    # skip_frames = 11
    # it_times = np.arange(30)
    # it_times_all = np.hstack((it_times, it_times+60))

    for i in range(real_time.shape[0]):
        # for i in range(real_time.shape[0]-10, real_time.shape[0]):
        # for i in range(20):
        # for i in it_times_all:
        # for i in range(2):

        time = real_time[i]
        if time < 2.2:
            if sim_name == 'mujoco':
                # alpha_red = 0.5  # Looks good
                alpha_red = 0.4
                alpha_black = 0.05
            elif sim_name == 'bullet':
                alpha_red = 0.2  # Maybe a bit less?
                # alpha_red = 0.15  # looks ok-ish Maybe a bit less?
                alpha_black = 0.05
            elif sim_name == 'softgym':
                # alpha_red = 0.01 # not too bad, a bit too much
                # alpha_red = 0.05 # too much
                alpha_red = 0.02
                alpha_black = 0.05
            else:  # sofa
                alpha_red = 0.2
                alpha_black = 0.05
        else:
            if sim_name == 'mujoco':
                alpha_red = 0.4
                alpha_black = 0.05
            elif sim_name == 'bullet':
                alpha_red = 0.2
                alpha_black = 0.05
            elif sim_name == 'softgym':
                alpha_red = 0.02
                alpha_black = 0.05
            else:
                alpha_red = 0.2
                alpha_black = 0.05
            # alpha_red = 1.0
            # alpha_black = 0.05
        idx = (np.abs(real_time - time)).argmin()
        # TODO CAN UNCOMMENT TO BE SURE
        # print(f'idx={idx}, timevalue={real_time[idx]}')

        target_idx = idx + skip_frames
        # print(f"Sim={sim_name} time={time}")

        (target_vertices, sim_cloud_npy,
         cd_cost_l1, hd_p1_to_p2, fabric_w_contact) = get_target_and_sim_cloud(cloth_sequence, target_idx,
                                                                               skip_frames)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection='3d'), dpi=150)
            ax.scatter(sim_cloud_npy[:, 0], sim_cloud_npy[:, 1], sim_cloud_npy[:, 2],
                       marker='*', color='r', label='Sim', alpha=alpha_red)

            ax.scatter(target_vertices[:, 0], target_vertices[:, 1], target_vertices[:, 2],
                       marker='^', color='k', label='Real', alpha=alpha_black)
            # if fabric_w_contact:
            #     ax.scatter(table_points[:, 0], table_points[:, 1], table_points[:, 2],
            #            marker='*', color='r', label='Sim Table', alpha=alpha_red)

            ax.grid(visible=False)
            ax.view_init(elev=elev, azim=azim)
            # ax.set_xlim([-0.2, 0.6])
            # ax.set_ylim([-0.8, 0.8])
            # ax.set_zlim([0.00, 1.1])
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            ax.set_xticklabels([])
            # ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m.)')
            ax.set_zlabel('Z (m.)')
            ax.set_aspect('equal')
            # Place the legend at the bottom
            # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01), fancybox=True, shadow=True, ncol=2)
            # Place the legend at the top
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fancybox=True, shadow=True, ncol=2,
                      fontsize=10)

            # ax.tick_params(axis='both', direction='out', reset=True, width=20, length=100, which='major')
            # ax.tick_params(axis='y', width=20, length=25, which='major')
            # ax.tick_params(axis='z', width=20, length=25, which='major')
            # ax.tick_params(axis='z', width=5, length=7.5, which='minor')
            plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.99)

            # Write CD
            # ax.text(0.5, -0.2, -0.5,
            #         s=f'CD={trunc(np.round(cd_cost_l1.item(), decimals=nr_decimals), decs=nr_decimals)}', fontsize=12)
            # verticalalignment='bottom', horizontalalignment='right',
            # transform=ax.transAxes)
            # ax.text(0.5, 0.5, 0.5, s=f'CD={trunc(cd_cost_l1)}', fontsize=15,
            #         verticalalignment='bottom', horizontalalignment='right',
            #         transform=ax.transAxes)
            # Write HD
            # ax.text(0.5, -0.2, -0.6, s=f'HD={trunc(np.round(hd_p1_to_p2, decimals=nr_decimals), decs=nr_decimals)}',
            #         fontsize=12)
            # ax.text(0.5, 0.01, f'CD={trunc(cd_cost_l1)}', fontsize=15)
            # ax.text(2, 6, f'an equation: $E=mc^2$', fontsize=15)

            if i < 10:
                it_str = f'00{i}'
            elif i < 100:
                it_str = f'0{i}'
            else:
                it_str = f'{i}'

            plt.savefig(f'{out_folder}/{it_str}.png')
            plt.close()
        # if fabric_w_contact:
        #     cd_f_list.append(cd_cost_l1.item())
        #     hd_f_list.append(hd_p1_to_p2)
        # else:
        #     cd_c_list.append(cd_cost_l1.item())
        #     hd_c_list.append(hd_p1_to_p2)
        # ax[i, j].set_xticks([])
        # ax[i, j].set_yticks([])
        # plt.gca().axes.get_yaxis().set_visible(False)
        # ax[i,j ].set_axis_off()
    return np.array(cd_f_list), np.array(cd_c_list), np.array(hd_f_list), np.array(hd_c_list)



source_dir = "/home/mulerod1/projects/dynamic_dual_manip/results/"
data_path = "/home/mulerod1/projects/dynamic_dual_manip/dataset/point_clouds"
primitive_type = "dynamic"
# primitive_type = "quasi_static"

primitive_types = ["quasi_static", "dynamic"]
sim_folders = ['sofa', 'mujoco3', 'softgym', 'bullet']
# sim_folders = ['mujoco3', 'bullet', 'softgym', 'sofa']


# all_cloth_types = ["towel", "chequered", "linen"]
all_cloth_types = ["chequered"]

# Create the output folders
for sim in sim_folders:
    for cloth_type in all_cloth_types:
        for primitive in primitive_types:
            out_folder = f"{source_dir}/{sim}_{cloth_type}_{primitive}"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

all_cloth_types = ["towel", "chequered", "linen"]
# all_cloth_types = ["chequered"]

for fd in os.listdir(source_dir):
    if 'cloth_sequence' in fd:
        if all_cloth_types[0] in fd:
            cloth_type = all_cloth_types[0]
            target = "towel_rag_0"
            continue
        elif all_cloth_types[1] in fd:
            cloth_type = all_cloth_types[1]
            target = "chequered_rag_0"
        else:
            cloth_type = all_cloth_types[2]
            target = "white_rag_0"
            continue

        if 'quasi_static' in fd:
            primitive_type = "quasi_static"
        elif 'dynamic' in fd:
            primitive_type = "dynamic"

        target_path = f"{data_path}/{target}/{primitive_type}/cloud"
        target_tgt = load_targets(target_path)

        cloth_sequence = np.load(source_dir + fd)

        print(f"Length of tgt={len(target_tgt)}, sequence={cloth_sequence.shape}")
        if 'mujoco3' in fd:
            env_name = 'mujoco3'
            continue
        elif 'bullet' in fd:
            env_name = 'bullet'
            continue
        elif 'softgym' in fd:
            env_name = 'softgym'
            # continue
        else:
            env_name = 'sofa'
            continue


        print(f"sim={env_name}, np shape={cloth_sequence.shape}")

        out_folder = f"{source_dir}/{env_name}_{cloth_type}_{primitive_type}"

        cd_f_arr, cd_c_arr, hd_f_arr, hd_c_arr = plot_and_save_cloth_sequence(cloth_sequence, env_name,
                                                                              out_folder,
                                                                              elev=0,
                                                                              azim=0,
                                                                              primitive=primitive_type,
                                                                              plot=True)

        # # Save the result to a txt file
        # with open(f'{out_folder}/result.txt', 'w') as f:
        #     f.write(f'CDf mean={trunc(np.round(cd_f_arr.mean(), decimals=nr_decimals), decs=nr_decimals)}'
        #             f' var={trunc(np.round(cd_f_arr.std(), decimals=nr_decimals), decs=nr_decimals)}\n'
        #             f'CDc mean={trunc(np.round(cd_c_arr.mean(), decimals=nr_decimals), decs=nr_decimals)}'
        #             f' var={trunc(np.round(cd_c_arr.std(), decimals=nr_decimals), decs=nr_decimals)}\n'
        #             f'HDf mean={trunc(np.round(hd_f_arr.mean(), decimals=nr_decimals), decs=nr_decimals)}'
        #             f' var={trunc(np.round(hd_f_arr.std(), decimals=nr_decimals), decs=nr_decimals)}\n'
        #             f'HDc mean={trunc(np.round(hd_c_arr.mean(), decimals=nr_decimals), decs=nr_decimals)}'
        #             f' var={trunc(np.round(hd_c_arr.std(), decimals=nr_decimals), decs=nr_decimals)}\n')

