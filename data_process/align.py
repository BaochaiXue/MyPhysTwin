from __future__ import annotations

"""Align the TRELLIS shape prior with observed RGB-D data using feature matching and ARAP.

This script renders the TRELLIS mesh under multiple viewpoints, selects the best match against the
reference camera frame via SuperGlue, lifts correspondences into 3D, and solves for camera pose, scale,
and non-rigid deformations so the mesh agrees with the fused point clouds. It expects segmentation masks,
calibration (`calibrate.pkl`), and metadata (`metadata.json`) to be present in the case directory.

Outputs include cached intermediate artefacts in ``shape/matching/`` (best render, feature matches,
precomputed correspondences) and a refined mesh ``final_mesh.glb`` suitable for downstream sampling or
simulation. The code operates in-place on files within the case directory and assumes GPU rendering
helpers from ``utils.align_util`` are available.
"""

from typing import Any, Dict, Sequence, Tuple

import open3d as o3d  # Provides 3D geometry manipulation and visualisation utilities.
import numpy as np  # Core numerical library for matrix algebra and vector operations.
from argparse import ArgumentParser  # Handles command-line arguments specifying dataset context.
import pickle  # Loads cached intermediate results (e.g., camera calibration, matches).
import trimesh  # Supplies mesh handling with ray casting used during visibility checks.
import cv2  # Handles image I/O, feature processing, and camera pose estimation.
import json  # Reads dataset metadata describing camera intrinsics.
import torch  # Included for completeness; certain utilities may rely on GPU tensors.
import os  # Manages filesystem operations such as ensuring output directories exist.
from utils.align_util import (
    render_multi_images,
    render_image,
    as_mesh,
    project_2d_to_3d,
    plot_mesh_with_points,
    plot_image_with_points,
    select_point,
)  # Custom alignment helpers shared across scripts.
from match_pairs import image_pair_matching  # SuperGlue-based matching between rendered and observed images.
import matplotlib.pyplot as plt  # Used for diagnostic plots of matching results.
from scipy.optimize import minimize  # Optimises the post-PnP scale parameter.
from scipy.spatial import KDTree  # Accelerates nearest-neighbour lookups on mesh/point sets.

VIS: bool = True  # Global flag controlling whether intermediate visualisations are generated.
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)  # Root directory containing the dataset structure.
parser.add_argument("--case_name", type=str, required=True)  # Specific capture session to align.
parser.add_argument("--controller_name", type=str, required=True)  # Label used to differentiate controller masks from object masks.
args = parser.parse_args()

base_path = args.base_path  # Dataset root path provided by user.
case_name = args.case_name  # Name of the case being processed.
CONTROLLER_NAME = args.controller_name  # Semantic name of the controller (typically "hand").
output_dir = f"{base_path}/{case_name}/shape/matching"  # Folder storing alignment artefacts and diagnostics.


def existDir(dir_path: str) -> None:
    """Create ``dir_path`` if it does not exist so downstream file writes succeed.

    Args:
        dir_path (str): Absolute or relative directory path used to store alignment artefacts.

    Returns:
        None
    """
    if not os.path.exists(dir_path):  # Avoid re-creating directories unnecessarily.
        os.makedirs(dir_path)  # Recursively create the requested directory.


def pose_selection_render_superglue(
    raw_img: np.ndarray,
    fov: float,
    mesh_path: str,
    mesh: trimesh.Trimesh,
    crop_img: np.ndarray,
    output_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """Render candidate viewpoints of the mesh and choose the best via SuperGlue matching.

    Args:
        raw_img (np.ndarray): Full-resolution RGB frame from the reference camera (H x W x 3).
        fov (float): Horizontal field-of-view of the reference camera in radians.
        mesh_path (str): File path to the TRELLIS mesh on disk, required by the renderer.
        mesh (trimesh.Trimesh): Mesh instance used for subsequent visualisation.
        crop_img (np.ndarray): Grayscale crop of the object (background removed) for robust feature matching.
        output_dir (str): Directory where diagnostic renders and match data are cached.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
            - Colour render of the selected viewpoint (H x W x 3).
            - Depth map corresponding to the render (H x W).
            - 4x4 pose matrix describing the camera->mesh transform.
            - SuperGlue match metadata dictionary.
            - Intrinsic matrix used for rendering the candidate views.
    """
    # Calculate suitable rendering radius
    bounding_box = mesh.bounds  # Axis-aligned bounds of the TRELLIS mesh.
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])  # Diagonal length used to approximate object size.
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)  # Camera distance ensuring the mesh fits comfortably in view.

    # Render multimle images and feature matching
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )  # Sample multiple candidate viewpoints for the mesh.
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]  # Convert renders to grayscale for SuperGlue input.
    # Use superglue to match the features
    best_idx, match_result = image_pair_matching(
        grays, crop_img, output_dir, viz_best=True
    )  # Identify the rendered view that best matches the cropped observation.
    print("matched point number", np.sum(match_result["matches"] > -1))  # Report number of matched features as a quality metric.

    best_color = colors[best_idx]  # Colour render corresponding to the best match.
    best_depth = depths[best_idx]  # Depth map aligned with the chosen render.
    best_pose = camera_poses[best_idx].cpu().numpy()  # Camera pose (mesh-to-render transform) for the best view.
    return best_color, best_depth, best_pose, match_result, camera_intrinsics  # Return diagnostics for downstream processing.


def registration_pnp(
    mesh_matching_points: np.ndarray,
    raw_matching_points: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Solve PnP between mesh keypoints and 2D image correspondences.

    Args:
        mesh_matching_points (np.ndarray): Array of shape (N, 3) containing 3D keypoints on the mesh.
        raw_matching_points (np.ndarray): Array of shape (N, 2) with corresponding pixel coordinates in the image.
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix from metadata.json.

    Returns:
        np.ndarray: 4x4 transform that maps mesh coordinates into the raw camera coordinate frame.
    """
    # Solve the PNP and verify the reprojection error
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )  # Estimate pose using EPnP (suitable for many correspondences).
    assert success, "solvePnP failed"  # Abort if pose estimation fails; alignment cannot proceed.
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )  # Reproject 3D keypoints using the estimated pose to assess quality.
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()  # Compute mean reprojection error in pixels.
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed for this case {case_name}.$$$$$$$$$$$$$$$$$$$$$$$$$$")  # Warn when reprojection is unreasonably large.

    rotation_matrix, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix.
    mesh2raw_camera = np.eye(4, dtype=np.float32)  # Homogeneous transform initialised to identity.
    mesh2raw_camera[:3, :3] = rotation_matrix  # Insert rotation component.
    mesh2raw_camera[:3, 3] = tvec.squeeze()  # Insert translation component.

    return mesh2raw_camera  # Provide transform from mesh coordinates into raw camera coordinates.


def registration_scale(
    mesh_matching_points_cam: np.ndarray, matching_points_cam: np.ndarray
) -> float:
    """Optimise a scalar to best align PnP-transformed mesh points to observed 3D points in camera space.

    Args:
        mesh_matching_points_cam (np.ndarray): (N, 3) mesh keypoints transformed into camera space.
        matching_points_cam (np.ndarray): (N, 3) observed 3D points sampled from fused point clouds.

    Returns:
        float: Scalar value that minimises squared distance between mesh and observed points.
    """
    # After PNP, optimize the scale in the camera coordinate
    def objective(scale, mesh_points, pcd_points):
        transformed_points = scale * mesh_points  # Apply candidate scale.
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))  # Compute squared distance to observed points.
        return loss  # Optimiser minimises this residual.

    initial_scale = 1  # Start with no scaling.
    result = minimize(
        objective,
        initial_scale,
        args=(mesh_matching_points_cam, matching_points_cam),
        method="L-BFGS-B",
    )  # Solve for scale using quasi-Newton optimisation.
    optimal_scale = result.x[0]  # Extract scalar result.
    print("Rescale:", optimal_scale)
    return optimal_scale  # Return best-fit scale factor.


def deform_ARAP(
    initial_mesh_world: o3d.geometry.TriangleMesh,
    mesh_matching_points_world: np.ndarray,
    matching_points: np.ndarray,
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Perform ARAP deformation using matched keypoints between mesh and observed points.

    Args:
        initial_mesh_world (o3d.geometry.TriangleMesh): Mesh expressed in world coordinates prior to deformation.
        mesh_matching_points_world (np.ndarray): (N, 3) mesh points in world coordinates derived from matches.
        matching_points (np.ndarray): (N, 3) world-space targets from fused point clouds.

    Returns:
        Tuple[o3d.geometry.TriangleMesh, np.ndarray]: The deformed mesh plus the vertex indices constrained by ARAP.
    """
    # Do the ARAP deformation based on the matching keypoints
    mesh_vertices = np.asarray(initial_mesh_world.vertices)  # Access vertex coordinates of the Open3D mesh.
    kdtree = KDTree(mesh_vertices)  # Build KD-tree for nearest-vertex lookup.
    _, mesh_points_indices = kdtree.query(mesh_matching_points_world)  # Find closest vertices to each matched keypoint.
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)  # Ensure indices are integer array.
    deform_mesh = initial_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matching_points),
        max_iter=1,
    )  # Run ARAP with single iteration to gently nudge mesh toward observations.
    return deform_mesh, mesh_points_indices  # Return deformed mesh and the vertex indices that were constrained.


def get_matching_ray_registration(
    mesh_world: o3d.geometry.TriangleMesh,
    obs_points_world: np.ndarray,
    mesh: trimesh.Trimesh,
    trimesh_indices: np.ndarray,
    c2w: np.ndarray,
    w2c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ray-cast mesh visibility and pair visible vertices with closest observation points along camera rays.

    Args:
        mesh_world (o3d.geometry.TriangleMesh): Mesh currently positioned in world space.
        obs_points_world (np.ndarray): (M, 3) array of observed points from the fused RGB-D reconstruction.
        mesh (trimesh.Trimesh): Trimesh copy of the mesh used to perform ray casting tests.
        trimesh_indices (np.ndarray): Index array mapping Open3D vertex order to trimesh vertex order.
        c2w (np.ndarray): 4x4 camera-to-world transform for the current viewpoint.
        w2c (np.ndarray): 4x4 world-to-camera transform (inverse of ``c2w``).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Vertex indices (ints) that passed visibility tests.
            - Corresponding world-space target positions derived from the observation point cloud.
    """
    # Get the matching indices and targets based on the viewpoint
    obs_points_cam = np.dot(
        w2c,
        np.hstack((obs_points_world, np.ones((obs_points_world.shape[0], 1)))).T,
    ).T  # Transform observed PCD to camera coordinates.
    obs_points_cam = obs_points_cam[:, :3]
    vertices_cam = np.dot(
        w2c,
        np.hstack(
            (
                np.asarray(mesh_world.vertices),
                np.ones((np.asarray(mesh_world.vertices).shape[0], 1)),
            )
        ).T,
    ).T  # Transform mesh vertices to camera coordinates.
    vertices_cam = vertices_cam[:, :3]

    obs_kd = KDTree(obs_points_cam)  # Prepare KD-tree for nearest point queries within the observation cloud.

    new_indices = []  # Track mesh vertex indices with valid correspondences.
    new_targets = []  # Store matched observation points in world coordinates.
    # trimesh used to do the ray-casting test
    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]  # Update trimesh vertices to camera-space coordinates (respecting duplication mapping).
    for index, vertex in enumerate(vertices_cam):  # Evaluate each mesh vertex visibility.
        ray_origins = np.array([[0, 0, 0]])  # Camera origin in camera space.
        ray_direction = vertex  # Vector from camera to vertex.
        ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalise direction vector.
        ray_directions = np.array([ray_direction])  # Prepare array for trimesh ray casting.
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )  # Detect intersections along the ray towards the vertex.

        ignore_flag = False  # Will be set when vertex is occluded along the ray.

        if len(locations) > 0:
            first_intersection = locations[0]  # Get intersection closest to camera.
            vertex_distance = np.linalg.norm(vertex)  # Distance to candidate vertex.
            intersection_distance = np.linalg.norm(first_intersection)  # Distance to intersection point.
            if intersection_distance < vertex_distance - 1e-4:
                # If the intersection point is not the vertex, it means the vertex is not visible from the camera viewpoint
                ignore_flag = True  # Vertex is occluded; skip it.

        if ignore_flag:
            continue  # Skip occluded vertices.
        else:
            # Select the closest point to the ray of the observation points as the matching point
            indices = obs_kd.query_ball_point(vertex, 0.02)  # Gather observed points within radius of the vertex.
            line_distances = line_point_distance(vertex, obs_points_cam[indices])  # Compute perpendicular distance from ray to each candidate.
            # Get the closest point
            if len(line_distances) > 0:
                closest_index = np.argmin(line_distances)  # Select point closest to the viewing ray.
                target = np.dot(
                    c2w, np.hstack((obs_points_cam[indices][closest_index], 1))
                )  # Convert matched observation back to world coordinates.
                new_indices.append(index)  # Record mesh vertex index.
                new_targets.append(target[:3])  # Record target location in world space.

    new_indices = np.asarray(new_indices)  # Convert lists to numpy arrays for efficient downstream use.
    new_targets = np.asarray(new_targets)

    return new_indices, new_targets  # Provide additional ARAP constraints derived from visibility rays.


def deform_ARAP_ray_registration(
    deform_kp_mesh_world: o3d.geometry.TriangleMesh,
    obs_points_world: np.ndarray,
    mesh: trimesh.Trimesh,
    trimesh_indices: np.ndarray,
    c2ws: Sequence[np.ndarray],
    w2cs: Sequence[np.ndarray],
    mesh_points_indices: np.ndarray,
    matching_points: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Augment ARAP constraints with ray-based correspondences across multiple camera viewpoints.

    Args:
        deform_kp_mesh_world (o3d.geometry.TriangleMesh): Mesh already deformed by keypoint ARAP step.
        obs_points_world (np.ndarray): (M, 3) observed world-space points aggregated from all cameras.
        mesh (trimesh.Trimesh): Trimesh instance mirroring the Open3D mesh for ray casting.
        trimesh_indices (np.ndarray): Mapping from Open3D vertex indices to trimesh vertex indices.
        c2ws (List[np.ndarray]): Camera-to-world transforms for every viewpoint considered.
        w2cs (List[np.ndarray]): Corresponding world-to-camera transforms (inverse of ``c2ws``).
        mesh_points_indices (np.ndarray): Vertex indices constrained during the keypoint-only ARAP pass.
        matching_points (np.ndarray): (N, 3) target positions associated with ``mesh_points_indices``.

    Returns:
        o3d.geometry.TriangleMesh: Mesh refined with both keypoint and ray-based ARAP constraints.
    """
    final_indices = []  # Aggregate vertex indices participating in ARAP constraints.
    final_targets = []  # Corresponding target positions in world coordinates.
    for index, target in zip(mesh_points_indices, matching_points):  # Start with keypoint-driven matches.
        if index not in final_indices:
            final_indices.append(index)
            final_targets.append(target)

    for c2w, w2c in zip(c2ws, w2cs):  # Iterate over every camera pose to gather additional correspondences.
        new_indices, new_targets = get_matching_ray_registration(
            deform_kp_mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
        )  # Acquire visibility-derived matches for this camera.
        for index, target in zip(new_indices, new_targets):
            if index not in final_indices:
                final_indices.append(index)
                final_targets.append(target)

    # Also need to adjust the positions to make sure they are above the table
    indices = np.where(np.asarray(deform_kp_mesh_world.vertices)[:, 2] > 0)[0]  # Identify vertices above zero plane.
    for index in indices:
        if index not in final_indices:
            final_indices.append(index)
            target = np.asarray(deform_kp_mesh_world.vertices)[index].copy()
            target[2] = 0  # Snap additional vertices down to the table plane.
            final_targets.append(target)
        else:
            target = final_targets[final_indices.index(index)]
            if target[2] > 0:
                target[2] = 0  # Ensure existing targets do not elevate vertices above table.
                final_targets[final_indices.index(index)] = target

    final_mesh_world = deform_kp_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )  # Apply ARAP with combined constraints to better fit partial observations.
    return final_mesh_world  # Return the final aligned mesh in world coordinates.


def line_point_distance(p: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute perpendicular distances from the ray defined by ``p`` to each observation point.

    Args:
        p (np.ndarray): 3-element direction vector pointing from camera to mesh vertex.
        points (np.ndarray): (K, 3) candidate observation points expressed in camera coordinates.

    Returns:
        np.ndarray: (K,) array of distances between each point and the ray originating at the camera.
    """
    # Compute the distance between points and the line between p and [0, 0, 0]
    p = p / np.linalg.norm(p)  # Normalise direction vector.
    points_to_origin = points  # Points already expressed in camera frame relative to origin.
    cross_product = np.linalg.norm(np.cross(points_to_origin, p), axis=1)  # Length of cross product equals area of parallelogram, related to distance.
    return cross_product / np.linalg.norm(p)  # Divide by |p| (1 after normalisation) to yield perpendicular distance.


if __name__ == "__main__":
    existDir(output_dir)  # Guarantee alignment output directory exists.

    cam_idx = 0  # Use camera 0 as the reference viewpoint for feature matching.
    img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"  # Path to the reference RGB frame.
    mesh_path = f"{base_path}/{case_name}/shape/object.glb"  # Path to the TRELLIS-generated mesh.
    # Get the mask index of the object
    with open(f"{base_path}/{case_name}/mask/mask_info_{cam_idx}.json", "r") as f:
        data = json.load(f)  # Load mapping from mask IDs to semantic labels.
    obj_idx = None  # Will hold the object mask index distinct from controller.
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")  # Guard against unexpected multi-object captures.
            obj_idx = int(key)
    mask_img_path = f"{base_path}/{case_name}/mask/{cam_idx}/{obj_idx}/0.png"  # Path to object mask for reference frame.
    # Load the metadata
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)  # Access camera intrinsics and other recorded metadata.
    intrinsic = np.array(data["intrinsics"])[cam_idx]  # Extract intrinsics corresponding to reference camera.

    # Load the c2w for the camera
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)  # Load camera-to-world transforms for all cameras.
        c2w = c2ws[cam_idx]  # Pose for reference camera.
        w2c = np.linalg.inv(c2w)  # Invert to obtain world-to-camera for reference.
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]  # Pre-compute world-to-camera for each camera viewpoint.

    # Load the shape prior
    mesh = trimesh.load_mesh(mesh_path, force="mesh")  # Load mesh using trimesh for ray casting.
    mesh = as_mesh(mesh)  # Convert to a format with separated vertices/faces (handles textured meshes).

    # Load and process the image to get a cropped version for easy superglue
    raw_img = cv2.imread(img_path)  # Load reference RGB frame using OpenCV.
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualisations and rendering alignment.
    # Get mask bounding box, larger than the original bounding box
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)  # Read object mask to isolate region of interest.

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))  # Approximate horizontal field of view from intrinsics.

    if not os.path.exists(f"{output_dir}/best_match.pkl"):
        # 2D feature Matching to get the best pose of the object
        bbox = np.argwhere(mask_img > 0.8 * 255)  # Extract all foreground pixels (robust to soft/blurred edges).
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )  # Determine tight bounding box around object.
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Compute centre for square crop expansion.
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])  # Use max dimension to form square crop.
        size = int(size * 1.2)  # Add 20% padding for context.
        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2),
        )  # Build expanded bounding box coordinates.
        # Make sure the bounding box is within the image
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(raw_img.shape[1], bbox[2]),
            min(raw_img.shape[0], bbox[3]),
        )  # Clamp to image boundaries to avoid indexing errors.
        # Get the masked cropped image used for superglue
        crop_img = raw_img.copy()
        mask_bool = mask_img > 0  # Binary mask of the object region.
        crop_img[~mask_bool] = 0  # Zero out background to focus on object features.
        crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]  # Crop to bounding box for higher SNR in matching.
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for SuperGlue.

        # Render the object and match the features
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                fov,
                mesh_path,
                mesh,
                crop_img,
                output_dir=output_dir,
            )
        )  # Render candidate poses and select the one best matching the cropped observation.
        with open(f"{output_dir}/best_match.pkl", "wb") as f:
            pickle.dump(
                [
                    best_color,
                    best_depth,
                    best_pose,
                    match_result,
                    camera_intrinsics,
                    bbox,
                ],
                f,
            )  # Cache results to avoid recomputation on subsequent runs.
    else:
        with open(f"{output_dir}/best_match.pkl", "rb") as f:
            best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                pickle.load(f)
            )  # Reload cached best match data.

    # Process to get the matching points on the mesh and on the image
    # Get the projected 3D matching points on the mesh
    valid_matches = match_result["matches"] > -1  # Filter out unmatched keypoints.
    render_matching_points = match_result["keypoints0"][valid_matches]  # 2D keypoints on rendered view.
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )  # Lift render keypoints into 3D mesh coordinates using depth map and camera pose.
    render_matching_points = render_matching_points[valid_mask]  # Keep only keypoints with valid depth values.
    # Get the matching points on the raw image
    raw_matching_points_box = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]  # Corresponding 2D keypoints from crop image (box coordinates).
    raw_matching_points_box = raw_matching_points_box[valid_mask]  # Remove entries without valid depth counterpart.
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])  # Shift crop coordinates back into full-image frame.

    if VIS:
        # Do visualization for the matching
        plot_mesh_with_points(
            mesh,
            mesh_matching_points,
            f"{output_dir}/mesh_matching.png",
        )  # Save render of mesh with highlighted matched keypoints.
        plot_image_with_points(
            best_depth,
            render_matching_points,
            f"{output_dir}/render_matching.png",
        )  # Visualise keypoints on the rendered depth map.
        plot_image_with_points(
            raw_img,
            raw_matching_points,
            f"{output_dir}/raw_matching.png",
        )  # Overlay raw image with matched 2D keypoints.

    # Do PnP optimization to optimize the rotation between the 3D mesh keypoints and the 2D image keypoints
    mesh2raw_camera = registration_pnp(
        mesh_matching_points, raw_matching_points, intrinsic
    )  # Estimate mesh->camera transform aligning mesh keypoints to observed 2D points.

    if VIS:
        pnp_camera_pose = np.eye(4, dtype=np.float32)  # Prepare pose matrix for rendering from PnP-estimated camera.
        pnp_camera_pose[:3, :3] = np.linalg.inv(mesh2raw_camera[:3, :3])  # Invert rotation to map camera->mesh.
        pnp_camera_pose[3, :3] = mesh2raw_camera[:3, 3]  # Position camera at translation estimated by PnP.
        pnp_camera_pose[:, :2] = -pnp_camera_pose[:, :2]  # Adjust axes to match renderer convention.
        color, depth = render_image(
            mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
        )  # Render mesh using PnP pose for qualitative check.
        vis_mask = depth > 0  # Identify pixels where mesh is rendered.
        color[0][~vis_mask] = raw_img[~vis_mask]  # Composite render over raw image for comparison.
        plt.imsave(f"{output_dir}/pnp_results.png", color[0])  # Save overlay image for manual inspection.

    # Transform the mesh into the real world coordinate
    mesh_matching_points_cam = np.dot(
        mesh2raw_camera,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T  # Apply PnP transform to mesh keypoints to express them in camera coordinates.
    mesh_matching_points_cam = mesh_matching_points_cam[:, :3]

    # Load the pcd in world coordinate of raw image matching points
    obs_points = []  # Will accumulate observed point clouds from all cameras.
    obs_colors = []  # Colour companion arrays for the same points.
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"  # Use first frame's fused point cloud for matching.
    mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"  # Load processed masks for filtering observed points.
    data = np.load(pcd_path)  # Load fused point cloud tensors (points, colours, masks).
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)  # Load mask dictionary keyed by frame/camera.
    for i in range(3):  # Aggregate observations from all cameras (assuming 3-camera rig).
        points = data["points"][i]
        colors = data["colors"][i]
        mask = processed_masks[0][i]["object"]  # Select only object points in each camera view.
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])
        if i == 0:
            first_points = points  # Cache camera-0 points before masking for later nearest-neighbour lookups.
            first_mask = mask

    obs_points = np.vstack(obs_points)  # Merge all observed object points into one array.
    obs_colors = np.vstack(obs_colors)  # Merge corresponding colours.

    # Find the cloest points for the raw_matching_points
    new_match, matching_points = select_point(
        first_points, raw_matching_points, first_mask
    )  # Snap 2D keypoints to nearest valid 3D observation in camera-0 point cloud.
    matching_points_cam = np.dot(
        w2c, np.hstack((matching_points, np.ones((matching_points.shape[0], 1)))).T
    ).T  # Convert selected observation points into camera coordinates for scale optimisation.
    matching_points_cam = matching_points_cam[:, :3]

    if VIS:
        # Draw the raw_matching_points and new matching points on the masked
        vis_img = raw_img.copy()
        vis_img[~first_mask] = 0  # Zero out background for clarity.
        plot_image_with_points(
            vis_img,
            raw_matching_points,
            f"{output_dir}/raw_matching_valid.png",
            new_match,
        )  # Visualise correspondences chosen for scale estimation.

    # Use the matching points in the camera coordinate to optimize the scame between the mesh and the observation
    optimal_scale = registration_scale(mesh_matching_points_cam, matching_points_cam)  # Solve for global scale aligning mesh to observations.

    # Compute the rigid transformation from the original mesh to the final world coordinate
    scale_matrix = np.eye(4) * optimal_scale  # Construct homogeneous scaling matrix.
    scale_matrix[3, 3] = 1  # Keep homogeneous coordinate unaffected.
    mesh2world = np.dot(c2w, np.dot(scale_matrix, mesh2raw_camera))  # Combine scale and PnP transform, then lift to world coordinates.

    mesh_matching_points_world = np.dot(
        mesh2world,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T  # Express mesh keypoints in world space after alignment.
    mesh_matching_points_world = mesh_matching_points_world[:, :3]

    # Do the ARAP based on the matching keypoints
    # Convert the mesh to open3d to use the ARAP function
    initial_mesh_world = o3d.geometry.TriangleMesh()  # Start with empty Open3D mesh container.
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))  # Fill with mesh vertices (trimesh ordering).
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))  # Fill with triangle indices.
    # Need to remove the duplicated vertices to enable open3d, however, the duplicated points are important in trimesh for texture
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()  # Deduplicate vertices for ARAP compatibility.
    # Get the index from original vertices to the mesh vertices, mapping between trimesh and open3d
    kdtree = KDTree(initial_mesh_world.vertices)
    _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))  # Map back from original (possibly duplicated) vertices to deduplicated set.
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(mesh2world)  # Apply global transform to Open3D mesh so it shares world coordinates with observations.

    # ARAP based on the keypoints
    deform_kp_mesh_world, mesh_points_indices = deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, matching_points
    )  # Perform initial ARAP using keypoint correspondences.

    # Do the ARAP based on both the ray-casting matching and the keypoints
    # Identify the vertex which blocks or blocked by the observation, then match them with the observation points on the ray
    final_mesh_world = deform_ARAP_ray_registration(
        deform_kp_mesh_world,
        obs_points,
        mesh,
        trimesh_indices,
        c2ws,
        w2cs,
        mesh_points_indices,
        matching_points,
    )  # Augment ARAP with visibility-derived constraints to better match partial scans.

    if VIS:
        final_mesh_world.compute_vertex_normals()  # Prepare normals for rendering.

        # Visualize the partial observation and the mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)  # Observed object points used for alignment.
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)  # Their associated colours.

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # Optional world axis for context.

        # Render the final stuffs as a turntable video
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # Off-screen rendering of the final alignment.
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))  # Capture frame to infer resolution.
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 MP4 encoder.
        video_writer = cv2.VideoWriter(
            f"{output_dir}/final_matching.mp4", fourcc, 30, (width, height)
        )  # Prepare video writer for 360-degree turntable.
        # final_mesh_world.compute_vertex_normals()
        # final_mesh_world.translate([0, 0, 0.2])
        # mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(final_mesh_world)
        # o3d.visualization.draw_geometries([pcd, final_mesh_world], window_name="Matching")
        vis.add_geometry(pcd)
        vis.add_geometry(final_mesh_world)
        # vis.add_geometry(coordinate)
        view_control = vis.get_view_control()

        for j in range(360):  # Rotate camera for full turntable video.
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))  # Capture current frame as float image.
            frame = (frame * 255).astype(np.uint8)  # Convert to 8-bit image for video encoding.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Switch to BGR for OpenCV writer.
            video_writer.write(frame)  # Append frame to video.
        vis.destroy_window()  # Clean up off-screen viewer once done.

    mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]  # Map deduplicated vertex positions back to original ordering.
    mesh.export(f"{output_dir}/final_mesh.glb")  # Save the aligned mesh for downstream use.
