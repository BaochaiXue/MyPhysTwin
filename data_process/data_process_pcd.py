from __future__ import annotations

"""Fuse multi-view RGB-D frames into world-aligned point clouds with basic depth filtering."""

from typing import List, Tuple

import numpy as np  # Fundamental numerical library used for matrix operations and point manipulations.
import open3d as o3d  # Handles point-cloud/mesh representations and visualisation utilities.
import json  # Reads metadata describing camera intrinsics and frame counts.
import pickle  # Loads calibration matrices (camera-to-world transforms).
import cv2  # Reads RGB/depth images from disk.
from tqdm import tqdm  # Provides progress bars for long-running frame loops.
import os  # Used for filesystem inspection and directory creation.
from argparse import (
    ArgumentParser,
)  # Parses command-line arguments specifying dataset paths.

# Configure CLI options so the script can be reused standalone or via orchestrators.
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)  # Root directory that stores all capture cases.
parser.add_argument(
    "--case_name", type=str, required=True
)  # Specific case folder to convert into point clouds.
args = parser.parse_args()  # Parse arguments immediately for convenience.

base_path = args.base_path  # Dataset root provided by the caller.
case_name = args.case_name  # Name of the case to process.


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float = 1,
    coordinate: bool = True,
    shoot: bool = False,
    length: float = 4,
    color: np.ndarray = np.array([0, 1, 0]),
    z_flip: bool = False,
) -> List[o3d.geometry.Geometry]:
    """Create Open3D primitives that depict a calibrated camera frustum in world coordinates.

    Args:
        transformation (np.ndarray): 4x4 camera-to-world homogeneous transform.
        fx (float): Focal length in pixels along the x-axis.
        fy (float): Focal length in pixels along the y-axis.
        cx (float): Principal point horizontal offset.
        cy (float): Principal point vertical offset.
        scale (float, optional): Overall size scaling factor for the rendered frustum. Defaults to 1.
        coordinate (bool, optional): Whether to include a coordinate-frame mesh at the camera origin. Defaults to True.
        shoot (bool, optional): If True, add a ray extending from the camera into the scene. Defaults to False.
        length (float, optional): Length of the optional ray visualising the principal axis. Defaults to 4.
        color (np.ndarray, optional): RGB colour applied to the optional ray. Defaults to green.
        z_flip (bool, optional): Whether to flip the viewing direction to match Open3D conventions. Defaults to False.

    Returns:
        list[o3d.geometry.Geometry]: Collection of Open3D geometries representing the camera.
    """
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale
        )  # Build a local axis-aligned frame for the camera origin.
        camera.transform(
            transformation
        )  # Move the coordinate frame into the camera pose.
    else:
        camera = (
            o3d.geometry.TriangleMesh()
        )  # Fallback dummy mesh when coordinate frame is undesired.
    # Add origin and four corner points in image plane
    points = []  # Collect vertices describing the frustum wireframe.
    camera_origin = np.array(
        [0, 0, 0, 1]
    )  # Homogeneous origin used for transformation.
    points.append(
        np.dot(transformation, camera_origin)[0:3]
    )  # Transform origin into world coordinates and record.
    # Calculate the four points for of the image plane
    magnitude = (
        cy**2 + cx**2 + fx**2
    ) ** 0.5  # Normalising factor derived from intrinsics.
    if z_flip:
        plane_points = [
            [-cx, -cy, fx],
            [-cx, cy, fx],
            [cx, -cy, fx],
            [cx, cy, fx],
        ]  # Flip image plane orientation when needed.
    else:
        plane_points = [
            [-cx, -cy, -fx],
            [-cx, cy, -fx],
            [cx, -cy, -fx],
            [cx, cy, -fx],
        ]  # Default image plane coordinates.
    for point in plane_points:
        point = list(
            np.array(point) / magnitude * scale
        )  # Normalise and scale the plane point.
        temp_point = np.array(
            point + [1]
        )  # Promote to homogeneous coordinates for transformation.
        points.append(
            np.dot(transformation, temp_point)[0:3]
        )  # Transform each frustum corner into world space.
    # Draw the camera framework
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 4],
        [1, 3],
        [3, 4],
    ]  # Indices describing frustum edges.
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )  # Construct a wireframe from the collected points.

    meshes = [camera, line_set]  # Base geometries included in the output list.

    if shoot:
        shoot_points = []  # Extra geometry showing the viewing ray when requested.
        shoot_points.append(
            np.dot(transformation, camera_origin)[0:3]
        )  # Start point at the camera origin.
        shoot_points.append(
            np.dot(transformation, np.array([0, 0, -length, 1]))[0:3]
        )  # End point along camera forward direction.
        shoot_lines = [[0, 1]]  # Single line segment connecting the two points.
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )  # Build the line geometry.
        shoot_line_set.paint_uniform_color(color)  # Colour the ray for visibility.
        meshes.append(shoot_line_set)  # Include the ray in the returned list.

    return meshes  # Caller receives all generated geometries.


def getPcdFromDepth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project a depth map into camera-space XYZ coordinates using the provided intrinsics.

    Args:
        depth (np.ndarray): 2D array of depth values (metres) captured by the RGB-D sensor.
        intrinsic (np.ndarray): 3x3 camera-intrinsic matrix for the view.

    Returns:
        np.ndarray: Array of shape (H, W, 3) containing 3D points expressed in the camera coordinate frame.
    """
    H, W = depth.shape  # Extract height/width of the depth map.
    x, y = np.meshgrid(np.arange(W), np.arange(H))  # Grid of pixel coordinates.
    x = x.reshape(-1)  # Flatten into 1D array for vectorised math.
    y = y.reshape(-1)  # Flatten into 1D array for vectorised math.
    depth = depth.reshape(-1)  # Flatten depth to align with pixel indices.
    points = np.stack(
        [x, y, np.ones_like(x)], axis=1
    )  # Assemble homogeneous pixel coordinates.
    points = (
        points * depth[:, None]
    )  # Scale by depth to obtain un-normalised camera rays.
    points = (
        points @ np.linalg.inv(intrinsic).T
    )  # Apply inverse intrinsics to reach metric camera coordinates.
    points = points.reshape(H, W, 3)  # Reshape back to image grid layout.
    return points  # Return XYZ coordinates for every pixel.


def get_pcd_from_data(
    path: str,
    frame_idx: int,
    num_cam: int,
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RGB-D data for a frame across all cameras and transform it into world-aligned point clouds.

    Args:
        path (str): Root directory of the case being processed.
        frame_idx (int): Frame number to load from each camera.
        num_cam (int): Number of calibrated cameras expected in the dataset.
        intrinsics (np.ndarray): Array of shape (num_cam, 3, 3) storing per-camera intrinsics.
        c2ws (np.ndarray): Array of shape (num_cam, 4, 4) storing camera-to-world extrinsics.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: World-space points, RGB colours, and boolean masks per camera.
    """
    total_points = []  # Accumulate world-space point volumes for each camera.
    total_colors = []  # Collect corresponding RGB values.
    total_masks = []  # Store per-pixel validity masks.
    for i in range(
        num_cam
    ):  # Iterate over all cameras contributing to the fused cloud.
        color = cv2.imread(
            f"{path}/color/{i}/{frame_idx}.png"
        )  # Read the per-camera colour image.
        color = cv2.cvtColor(
            color, cv2.COLOR_BGR2RGB
        )  # Convert OpenCV's BGR ordering back to RGB.
        color = (
            color.astype(np.float32) / 255.0
        )  # Normalise colours to [0, 1] floats for Open3D compatibility.
        depth = (
            np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0
        )  # Load the depth map and convert millimetres to metres.

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsics[i],
        )  # Reconstruct camera-space XYZ coordinates per pixel.
        masks = np.logical_and(
            points[:, :, 2] > 0.2, points[:, :, 2] < 1.5
        )  # Keep depths within a plausible range to remove sensor noise.
        points_flat = points.reshape(
            -1, 3
        )  # Flatten for homogeneous transform application.
        # Transform points to world coordinates using homogeneous transformation
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )  # Append ones so 4x4 matrices can be applied.
        points_world = np.dot(c2ws[i], homogeneous_points.T).T[
            :, :3
        ]  # Transform into world space using camera-to-world matrix.
        points_final = points_world.reshape(
            points.shape
        )  # Reshape back into image grid layout.
        total_points.append(
            points_final
        )  # Store world-space coordinates for this camera.
        total_colors.append(color)  # Store corresponding RGB values.
        total_masks.append(masks)  # Track which pixels were considered valid.
    # pcd = o3d.geometry.PointCloud()
    # visualize_points = []
    # visualize_colors = []
    # for i in range(num_cam):
    #     visualize_points.append(
    #         total_points[i][total_masks[i]].reshape(-1, 3)
    #     )
    #     visualize_colors.append(
    #         total_colors[i][total_masks[i]].reshape(-1, 3)
    #     )
    # visualize_points = np.concatenate(visualize_points)
    # visualize_colors = np.concatenate(visualize_colors)
    # coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # mask = np.logical_and(visualize_points[:, 2] > -0.15, visualize_points[:, 0] > -0.05)
    # mask = np.logical_and(mask, visualize_points[:, 0] < 0.4)
    # mask = np.logical_and(mask, visualize_points[:, 1] < 0.5)
    # mask = np.logical_and(mask, visualize_points[:, 1] > -0.2)
    # mask = np.logical_and(mask, visualize_points[:, 2] < 0.2)
    # visualize_points = visualize_points[mask]
    # visualize_colors = visualize_colors[mask]

    # pcd.points = o3d.utility.Vector3dVector(np.concatenate(visualize_points).reshape(-1, 3))
    # pcd.colors = o3d.utility.Vector3dVector(np.concatenate(visualize_colors).reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd])
    total_points = np.asarray(
        total_points
    )  # Convert lists to numpy arrays for compact storage.
    total_colors = np.asarray(
        total_colors
    )  # Convert to array form to simplify downstream indexing.
    total_masks = np.asarray(total_masks)  # Convert mask list to boolean array.
    return (
        total_points,
        total_colors,
        total_masks,
    )  # Provide the per-camera tensors to the caller.


def exist_dir(dir: str) -> None:
    """Create ``dir`` if missing.

    Args:
        dir (str): Filesystem path to create when absent.

    Returns:
        None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)  # Recursively create the directory tree.


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(
            f
        )  # Load dataset metadata including intrinsics and frame count.
    intrinsics = np.array(
        data["intrinsics"]
    )  # Convert intrinsics list into numpy array for numeric operations.
    WH = data["WH"]  # Unused image shape info, retained for completeness/debugging.
    frame_num = data["frame_num"]  # Total number of frames captured per camera.
    print(data["serial_numbers"])  # Display camera serials to confirm ordering.

    num_cam = len(intrinsics)  # Determine how many cameras were calibrated.
    c2ws = pickle.load(
        open(f"{base_path}/{case_name}/calibrate.pkl", "rb")
    )  # Load camera-to-world transforms from disk.

    exist_dir(
        f"{base_path}/{case_name}/pcd"
    )  # Ensure output directory for point clouds exists.

    cameras = []
    # Visualize the cameras
    for i in range(num_cam):
        camera = getCamera(
            c2ws[i],
            intrinsics[i, 0, 0],
            intrinsics[i, 1, 1],
            intrinsics[i, 0, 2],
            intrinsics[i, 1, 2],
            z_flip=True,
            scale=0.2,
        )  # Build camera frustum geometries for visualisation.
        cameras += camera  # Aggregate geometry from all cameras into a single list.

    vis = o3d.visualization.Visualizer()  # Create an interactive Open3D window.
    vis.create_window()  # Open the rendering window.
    for camera in cameras:
        vis.add_geometry(camera)  # Add each camera geometry to the scene for context.

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5
    )  # Global coordinate frame for reference.
    vis.add_geometry(coordinate)  # Visualise the world axes alongside the cameras.

    pcd = None  # Will hold the point cloud geometry reused across frames.
    for i in tqdm(range(frame_num)):  # Iterate over all frames with a progress bar.
        points, colors, masks = get_pcd_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws
        )  # Fetch per-camera world-space points, colours, and masks for the current frame.

        if i == 0:
            pcd = (
                o3d.geometry.PointCloud()
            )  # Create a fresh point cloud geometry for the first frame.
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )  # Flatten camera dimension and filter by mask before assigning XYZ points.
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )  # Apply matching colours to the same valid points.
            vis.add_geometry(pcd)  # Insert the merged cloud into the visualiser.
            # Adjust the viewpoint
            view_control = (
                vis.get_view_control()
            )  # Access camera control interface to orient the scene.
            view_control.set_front(
                [1, 0, -2]
            )  # Configure camera orientation for a clear initial view.
            view_control.set_up(
                [0, 0, -1]
            )  # Set upward direction relative to world axes.
            view_control.set_zoom(
                1
            )  # Zoom to a comfortable level for the dataset scale.
        else:
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )  # Update point positions for the latest frame.
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )  # Update colour attributes accordingly.
            vis.update_geometry(pcd)  # Notify Open3D that geometry data changed.

            vis.poll_events()  # Process GUI events to keep the window responsive.
            vis.update_renderer()  # Redraw the scene with updated geometry.

        np.savez(
            f"{base_path}/{case_name}/pcd/{i}.npz",
            points=points,
            colors=colors,
            masks=masks,
        )  # Persist per-camera point tensors and masks for future processing.
