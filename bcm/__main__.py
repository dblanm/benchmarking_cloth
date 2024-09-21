import os
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import point_cloud_utils as pcu
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
# from pytorch3d.loss.chamfer import chamfer_distance

from .envs import get_env
from .manipulation_utils import generate_full_trajectory
from .utils import (
    get_dynamic_target_cutoff,
    load_targets,
    map_error_metrics_dict,
    plot_sim_and_target,
)
from .vision.utils import (
    chamfer_distance_w_unidirectional
)


def compute_cost(
    env_name,
    sim_vertices,
    target,
    it_nr,
    save_gif,
    output_path,
):
    sim_cloud = torch.from_numpy(sim_vertices).unsqueeze(0).float()
    target = torch.from_numpy(target).unsqueeze(0).float()

    final_target = target

    if save_gif:
        plot_sim_and_target(sim_cloud, final_target, it_nr, output_path)
        plot_sim_and_target(sim_cloud, final_target, it_nr, output_path, azim=90)
        plot_sim_and_target(sim_cloud, final_target, it_nr, output_path, elev=0)
        plot_sim_and_target(
            sim_cloud, final_target, it_nr, output_path, elev=0, azim=90
        )

    # Compute the Chamfer distance
    x_cloud = sim_cloud
    y_cloud = final_target

    cd_cost_l1 = chamfer_distance_w_unidirectional(
        x_cloud, y_cloud, single_directional=True, norm=1
    )[0]

    target_cloud_npy = final_target.numpy()[0]
    
    if env_name == "sofa":  # Fixing error in pcu with sofa sim vertices
        first_sim_cloud = target_cloud_npy.copy()
        first_sim_cloud[: sim_vertices.shape[0], :] = sim_vertices
        first_sim_cloud[sim_vertices.shape[0] :, :] = np.nan
        sim_cloud_npy = first_sim_cloud[~np.isnan(first_sim_cloud[:, 0]), :]
    else:
        sim_cloud_npy = sim_cloud.numpy()[0]
    
    x_cloud_npy = sim_cloud_npy
    y_cloud_npy = target_cloud_npy
    
    # Haussdorf
    hd_p1_to_p2 = pcu.one_sided_hausdorff_distance(
        x_cloud_npy, y_cloud_npy, return_index=False
    )

    errors_dict = {
        "chamfer_l1_cost": cd_cost_l1,
        "one_side_hausdorff_cost": hd_p1_to_p2,
    }

    return errors_dict


def compute_metrics(
    targets,
    target_it,
    freq_relation,
    j,
    info,
    env_name,
    save_gif,
    primitive,
    output_path,
    target_name,
):
    # Define the error metrics
    errors_dict = {
        # Pytorch3D Metrics
        "chamfer_l1_cost_fabric": None,
        "chamfer_l1_cost_contact": None,
        # One-sided Hausdorff
        "one_side_hausdorff_cost_fabric": None,
        "one_side_hausdorff_cost_contact": None,
    }

    # The target comparison specifies the time at which the contact first starts, which we used to disentangle
    # The error of contact and the in-air. In the new version we do not disentangle them.
    # In case that we are using the dynamic, the target comparison is anytime before t= ?
    # For the quasi-static, the whole time should be compared.
    switching_timestep = get_dynamic_target_cutoff(target_name, primitive)

    # To obtain the target comparison, we compare the target value * freq relation
    target_comparison = switching_timestep * freq_relation
    # If the next iteration is in the target_it list then compare against the target
    if j in target_it:
        # Get the index of the target
        if freq_relation > 1:
            # We need to match the start of the target trajectory
            # So that it matches the sim
            target_idx = int(j / freq_relation)
        else:
            target_idx = int(target_it[j])

        target_pcd = targets[target_idx]

        sim_vertices = info["vertices"]

        metrics = compute_cost(
            env_name=env_name,
            sim_vertices=sim_vertices,
            target=target_pcd,
            it_nr=target_idx,
            save_gif=save_gif,
            output_path=output_path,
        )

        if j < target_comparison:
            modified_errors_dict = map_error_metrics_dict(
                dict_input=metrics, dict_output=errors_dict, fabric=True
            )
        else:
            # No need to compare
            errors_dict['chamfer_l1_cost_contact'] = 1000
            return errors_dict
        return modified_errors_dict

    else:
        return errors_dict


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    metric_errors = {
        # Pytorch3D metrics
        # Unidirectional Chamfer
        "chamfer_l1_cost_fabric_ls": [],
        "chamfer_l1_cost_contact_ls": [],
        "hausdorff_cost_contact_ls": [],
        # One-sided Hausdorff
        "one_side_hausdorff_cost_fabric_ls": [],
        "one_side_hausdorff_cost_contact_ls": [],
    }

    os.chdir(os.path.abspath(os.path.join(__file__, "../..")))

    # Reproducibility
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    plot = cfg.plot
    save_gif = cfg.save_gif

    if (
        HydraConfig.get().mode.name != "MULTIRUN" and (cfg.target != "None" or cfg.cloth_sample != "None")
    ):  # Then use the fine-tuned parameters
        if cfg.target != "None":
            cloth_target = cfg.target
        else:
            cloth_target =cfg.cloth_sample
        cfg.envs.params = cfg.envs[f"params_{cfg.primitive}_{cloth_target}"]
        print(f"Modified params to = {cfg.envs.params}")

    print(f"YAML Environment Params=\n {OmegaConf.to_yaml(cfg.envs.params)}")

    for key in cfg.envs.items():
        if "params_" in key[0]:
            cfg.envs.__delattr__(key[0])

    render_mode = cfg.envs.render_mode
    env = get_env(cfg.envs, real_setup=cfg.real_setup)
    dt = env.trajectory_dt
    if cfg.envs.name == "sofa" :
        # stabilise_steps = 1000  # For 0.01
        # stabilise_steps = 10000  # For 0.001
        stabilise_steps = int(10 / dt)
    elif cfg.envs.name == "bullet" or cfg.envs.name == "mujoco3":
        # stabilise_steps = 100  # For 0.01
        # stabilise_steps = 1000  # For 0.01
        stabilise_steps = int(1 / dt)
    else:
        # stabilise_steps = 500  # For 0.01
        # stabilise_steps = 5000  # For 0.001
        stabilise_steps = int(5 / dt)
    print(f"Number of stabilise steps={stabilise_steps}")

    os.makedirs(cfg.output_path, exist_ok=True)

    if not os.path.exists("results"):
        print("Creating path=results")
        os.makedirs("results")

    time_list = []
    trajectory, pre_traj_length = generate_full_trajectory(
        dt, cfg.target, cfg.target_path, stabilise_steps, primitive=cfg.primitive, sim=cfg.envs.name
    )

    if plot and render_mode == "rgb_array":
        plt.ion()

    if save_gif:
        frames = []

    if cfg.target != "None":
        targets = load_targets(cfg.target_path)
    else:
        targets = None

    quintic_length = len(trajectory) - pre_traj_length - 1
    print(f"Quintic trajectory length = {quintic_length}")

    if cfg.target != "None":
        target_length = len(os.listdir(cfg.target_path)) - 1
        print(f"Target trajectory length={target_length}")
        if quintic_length > target_length:
            target_it = np.linspace(0, quintic_length, target_length, dtype=int)
        else:
            target_it = np.linspace(0, target_length, quintic_length, dtype=int)
        freq_relation = quintic_length / target_length
        # Check whether we are going to compute the cost against all the targets or only a specific set
        if quintic_length < len(os.listdir(cfg.target_path)):
            select_targets = True
        else:
            select_targets = False
    else:
        target_it = None
        select_targets = None
        freq_relation = 1

    # Check if we are using SOFA
    if cfg.envs.name == "sofa":
        print("\n=====STARTING SOFA SIMULATION====\n")
        env.start_simulation(
            trajectory,
            pre_traj_length,
            compute_metrics,
            targets=targets,
            target_it=target_it,
            target_name=cfg.target,
            select_targets=select_targets,
        )
    else:
        observation, info = env.reset()

    cloth_obs = []

    # Run the trajectory
    for i in range(len(trajectory)):
        # Apply action
        action = trajectory[i]

        start = time.time()
        observation, reward, terminated, truncated, info = env.step(action)
        end = time.time()
        time_list.append(end - start)
        # env.render()

        if i >= pre_traj_length:

            cloth_obs.append(info["vertices"])

            j = i - pre_traj_length

            if cfg.target != "None":

                errors_dict = compute_metrics(
                    targets=targets,
                    target_it=target_it,
                    freq_relation=freq_relation,
                    target_name=cfg.target,
                    j=j,
                    info=info,
                    env_name=cfg.envs.name,
                    save_gif=save_gif,
                    primitive=cfg.primitive,
                    output_path=cfg.output_path
                )

                if errors_dict["chamfer_l1_cost_fabric"] is not None:
                    for key, value in errors_dict.items():
                        if "fabric" in key:
                            metric_errors[key + "_ls"].append(value)
                elif errors_dict["chamfer_l1_cost_contact"] is not None:
                    for key, value in errors_dict.items():
                        if "contact" in key:
                            metric_errors[key + "_ls"].append(value)
                else:
                    continue


            if save_gif:
                frames.append(info["rgb"])

        # TODO REMOVE THIS AFTER TESTING ALL
        if plot and render_mode == "rgb_array":
            to_plot = info["rgb"]

            plt.title(f"Timestep={i}")
            plt.imshow(to_plot)

            plt.show()
            plt.pause(0.001)
            plt.clf()

    env.close()
    print(f"After closing the environment")


    if save_gif and cfg.envs.name != "sofa":
        # from moviepy.editor import ImageSequenceClip
        import imageio as iio

        for i in range(len(frames)):
            iio.imwrite(
                f"{cfg.output_path}/{cfg.envs.name}_depth_{i:04d}.png",
                frames[i],
                format=None,
            )

    print(f"Simulation elapsed time mean={np.mean(time_list)} sum ={np.sum(time_list)}")

    if cfg.target != "None":
        # All the metrics are lists, first convert them to arrays
        metrics_arr_fabric = {}
        for key, val in metric_errors.items():
            if "fabric" in key:
                metrics_arr_fabric[key] = np.array(val)

        # Create the pandas dataframe
        pdf_fabric = pd.DataFrame.from_dict(metrics_arr_fabric)

        # To avoid writing too much into the disk, save only when not using the multirun optimisation
        if (
                HydraConfig.get().mode.name != "MULTIRUN"
        ):
            # Print the results and save them
            print("\n===Results===")
            print("Fabric")
            print(pdf_fabric.mean())
            print(pdf_fabric.std())

            if cfg.envs.name == "bullet":
                if cfg.envs.pbd:
                    env_name = "bullet_pbd"
                else:
                    env_name = "bullet"
            else:
                env_name = cfg.envs.name

            # Save the results
            pdf_fabric.to_csv(f"results/{env_name}_fabric_{cfg.target}_{cfg.primitive}_{cfg.seed}.csv")
            np.save(
                f"results/{env_name}_time_{cfg.target}_{cfg.seed}.npy", np.array(time_list)
            )
            cloth_sequence = np.stack(cloth_obs)
            np.save(
                f"results/{cfg.envs.name}_{cfg.target}_{cfg.primitive}_{cfg.seed}_cloth_sequence.npy",
                cloth_sequence,
            )


        mean_chamfer = (
            pdf_fabric["chamfer_l1_cost_fabric_ls"].mean()
        )
        if isinstance(mean_chamfer, float):
            pass
        else:
            mean_chamfer = mean_chamfer.item()
        return mean_chamfer


if __name__ == "__main__":
    main()
