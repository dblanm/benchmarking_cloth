import argparse
import os

import numpy as np
import open3d as o3d
import pymeshlab
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def outlier_removal(
    pcd, nb_neighbors=500, std_ratio=1.0, nb_points=30, radius=0.02, depth_trunc=2.0
):
    """

    Args:
        pcd: Point cloud.
        nb_neighbors: how many neighbors are taken into account in order to calculate the average distance for a given point.
        std_ratio: threshold level based on the standard deviation of the average distances across the point cloud. The lower this number the more aggressive the filter will be.
        nb_points: minimum amount of points that the sphere should contain.
        radius: radius of the sphere that will be used for counting the neighbors.
        depth_trunc: truncate by this depth value (in meters).

    Returns:
        Clean point cloud.

    """
    # Remove points that are further away from their neighbors compared to the average for the point cloud
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    pcd, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

    if depth_trunc is not None:
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(points[:, -1] < depth_trunc)[0].tolist())
    return pcd


def compute_translation(
    pcd, left_gripper, right_gripper, iters=1000, learning_rate=0.01
):
    points = np.asarray(pcd.points)
    mean_x = points.mean(axis=0)[0]
    # Left side are the points that have an x component smaller than the mean of x values
    left_side = points[points[:, 0] < mean_x]
    # Right side is the same but larger
    right_side = points[points[:, 0] > mean_x]

    # Find corners as the ones with higher z value
    upper_cloth_points = np.c_[
        left_side[np.argmax(left_side[:, -1], axis=0)],
        right_side[np.argmax(right_side[:, -1], axis=0)],
    ].T

    gripper_positions = np.array([left_gripper, right_gripper])

    translation_vector = np.zeros(3)
    for _ in range(iters):
        # Use L2 loss
        gradient = np.sum(
            2 * (upper_cloth_points + translation_vector - gripper_positions), axis=0
        )
        translation_vector -= learning_rate * gradient

    return translation_vector


def compute_scale(pcd, cloth_area):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(pcd.points)))
    # Obtain mesh from point cloud
    ms.apply_filter("generate_surface_reconstruction_ball_pivoting")
    # Select all faces
    ms.apply_filter("set_selection_all")
    # Compute area using previous mesh
    quantities = ms.apply_filter("get_area_and_perimeter_of_selection")
    return np.sqrt(quantities["selected_surface_area"] / cloth_area)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Path to root of the dataset. Assume inputs follow the file structure {root}/{material}/{primitive}/ which contains the files rgb_arr_{i}.npy and depth_arr_{i}.npy. If this folder also contains seq_{i}/masks/ with the PNG masks, masks will be used.",
    )
    parser.add_argument(
        "--fx", type=float, default=605.70623779, help="Horizontal pixel focal length"
    )
    parser.add_argument(
        "--fy", type=float, default=605.82971191, help="Vertical pixel focal length"
    )
    parser.add_argument(
        "--camera_position",
        nargs="+",
        default=[-0.1644, 1.5926, 0.62733],
        help="Camera position",
    )
    parser.add_argument(
        "--camera_orientation",
        nargs="+",
        default=[0.0027362, 0.78122, -0.62422, -0.0055694],
        help="Camera orientation expressed as a quaternion",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Number of trials for each of the combinations of material and motion type",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=["chequered", "cotton", "linen"],
        help="List of materials in the dataset",
    )
    parser.add_argument(
        "--primitives",
        nargs="+",
        default=["dynamic", "quasi_static"],
        help="List of motion types in the dataset",
    )
    parser.add_argument(
        "--left_gripper",
        nargs="+",
        default=[0.0, 0.0, 1.0],
        help="Initial position of the left gripper",
    )
    parser.add_argument(
        "--right_gripper",
        nargs="+",
        default=[0.5, 0.0, 1.0],
        help="Initial position of the right gripper",
    )
    parser.add_argument(
        "--cloth_area",
        type=float,
        default=0.5 * 0.7,
        help="Area of the cloth (m^2)",
    )

    args = parser.parse_args()

    global_translation = None
    material_scale = {material: None for material in args.materials}

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_quat(args.camera_orientation).as_matrix()
    extrinsic[:3, -1] = args.camera_position

    # Outputs will be structured as:
    # {root}/{material}_rag_{i}/{primitive}/cloud
    # which is the file structured required by the default Hydra config
    # and has to contain the PLY files with the sequence number
    for material in args.materials:
        # Iterate over materials
        for primitive in args.primitives:
            # Iterate over motion types
            for i in range(args.n_trials):
                # Iterate over trials

                # Load images as npy array
                images = np.load(
                    os.path.join(args.root, material, primitive, f"rgb_arr_{i}.npy")
                )

                # Load depths as npy array
                depths = np.load(
                    os.path.join(args.root, material, primitive, f"depth_arr_{i}.npy")
                )

                # Define intrinsics from focal length and image dimensions
                _, height, width, _ = images.shape
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    fx=args.fx,
                    fy=args.fy,
                    width=width,
                    height=height,
                    cx=width / 2,
                    cy=height / 2,
                )

                # Directory with PNG masks
                mask_dir = os.path.join(
                    args.root,
                    material,
                    primitive,
                    f"seq_{i}",
                    "mask",
                )
                if not os.path.isdir(mask_dir):
                    print("Mask not found")
                else:
                    # If there are masks check that there is one for every RGB and depth image
                    assert len(images) == len(depths) == len(os.listdir(mask_dir))

                # Define point cloud output directory and create if needed
                pcd_dir = os.path.join(
                    args.root,
                    f"{material}_rag_{i}",
                    primitive,
                    "cloud",
                )
                os.makedirs(pcd_dir, exist_ok=True)

                for j, (img, depth) in enumerate(
                    tqdm(
                        zip(images, depths),
                        desc=f"Processing {material} {primitive}",
                        total=len(images),
                        leave=False,
                    )
                ):
                    # If there is a mask file, use it to mask both RGB and depth
                    if os.path.isfile(os.path.join(mask_dir, f"{j:05d}.png")):
                        mask_pil = Image.open(os.path.join(mask_dir, f"{j:05d}.png"))
                        assert mask_pil.width == width and mask_pil.height == height
                        mask = np.array(mask_pil).astype(bool)
                        # Mask has shape (height, width)
                        depth_o3d = o3d.geometry.Image(mask * depth)
                        # Repeat mask over each of the color channels
                        img_o3d = o3d.geometry.Image(
                            np.repeat(mask[:, :, np.newaxis], 3, axis=2) * img
                        )
                    else:
                        depth_o3d = o3d.geometry.Image(depth)
                        img_o3d = o3d.geometry.Image(img)

                    # Create point cloud with depth scale = 1, i.e. depth values are in meters
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color=img_o3d, depth=depth_o3d, depth_scale=1
                    )
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        image=rgbd, intrinsic=intrinsic, extrinsic=extrinsic
                    )

                    # Remove outliers
                    pcd = outlier_removal(pcd)

                    # Scale for each material (rags of different sizes have slightly different dimensions)
                    if material_scale[material] is None:
                        material_scale[material] = compute_scale(pcd, args.cloth_area)

                    pcd.scale(material_scale[material], center=pcd.get_center())

                    # Compute a translation. Extrinsics accounts for the correct rotation but not the translation
                    if global_translation is None:
                        global_translation = compute_translation(
                            pcd,
                            args.left_gripper,
                            args.right_gripper,
                        )

                    pcd.translate(global_translation)

                    # Write resulting point cloud as PLY file
                    o3d.io.write_point_cloud(os.path.join(pcd_dir, f"{j:05d}.ply"), pcd)
