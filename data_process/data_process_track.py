"""Post-process dense tracks to remove outliers and extract stable controller anchor points."""

import numpy as np  # Numerical backbone for array manipulation and vectorised filtering.
import open3d as o3d  # Offers point-cloud processing, KD-tree queries, and visualisation utilities.
from tqdm import tqdm  # Provides progress bars for per-frame iteration loops.
import os  # Manages filesystem operations such as directory creation.
import glob  # Counts files to infer numbers of cameras/frames.
import pickle  # Serialises/deserialises processed tracking data and masks.
import matplotlib.pyplot as plt  # Supplies colour maps for visualisation of motion quality.
from argparse import (
    ArgumentParser,
)  # Parses CLI arguments describing dataset locations.

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)  # Root folder containing the processed dataset.
parser.add_argument(
    "--case_name", type=str, required=True
)  # Specific case directory to process.
args = parser.parse_args()

base_path = args.base_path  # Dataset root path provided by user.
case_name = args.case_name  # Case identifier under the dataset root.
IGNORE_COTRACKER_TRAJECTORIES_TOO_LESS: bool = (
    os.getenv("IGNORE_COTRACKER_TRAJECTORIES_TOO_LESS", "0") == "1"
)  # Environment variable to skip cases with too few Co-Tracker trajectories.


def exist_dir(dir):
    """Create ``dir`` if missing.

    Args:
        dir (str): Filesystem path that should exist prior to writing artefacts.

    Returns:
        None
    """
    if not os.path.exists(
        dir
    ):  # Avoid redundant mkdir calls when directory already exists.
        os.makedirs(dir)  # Recursively create path so downstream writes succeed.


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    """Create an Open3D sphere mesh at the specified centre for visualising controller points.

    Args:
        center (Iterable[float]): XYZ coordinate of the sphere centre in world space.
        radius (float, optional): Sphere radius controlling marker size in the viewer. Defaults to 0.1.
        color (Iterable[float], optional): RGB colour applied to the sphere surface. Defaults to black.

    Returns:
        o3d.geometry.TriangleMesh: Configured sphere mesh ready to be added to an Open3D scene.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(
        center
    )  # Build and position the primitive sphere.
    sphere.paint_uniform_color(color)  # Colour the sphere for easier identification.
    return sphere  # Return geometry to caller for inclusion in visualisation scenes.


# Based on the valid mask, filter out the bad tracking data
def filter_track(track_path, pcd_path, mask_path, frame_num, num_cam):
    """Load raw Co-Tracker outputs and prune trajectories inconsistent with processed object/controller masks.

    Args:
        track_path (str): Directory containing ``cotracker/{camera}.npz`` files with pixel trajectories.
        pcd_path (str): Directory containing fused world-space point clouds per frame.
        mask_path (str): Directory storing processed mask metadata and PNGs.
        frame_num (int): Number of frames in the current sequence.
        num_cam (int): Number of calibrated cameras recorded for the case.

    Returns:
        Dict[str, np.ndarray]: Dictionary with object/controller positions, colours, and visibility flags.
    """
    with open(f"{mask_path}/processed_masks.pkl", "rb") as f:
        processed_masks = pickle.load(
            f
        )  # Refined per-frame masks computed during mask post-processing.

    # Filter out the points not valid in the first frame
    object_points = (
        []
    )  # Will store object trajectories (XYZ) concatenated across cameras.
    object_colors = []  # Corresponding RGB colours for object trajectories.
    object_visibilities = (
        []
    )  # Binary visibility flags indicating when each trajectory is present.
    controller_points = []  # Controller trajectories (XYZ).
    controller_colors = []  # Controller colours for visualisation.
    controller_visibilities = []  # Visibility mask for controller trajectories.
    for i in range(
        num_cam
    ):  # Process each camera's tracking data independently before merging.
        current_track_data = np.load(
            f"{track_path}/{i}.npz"
        )  # Load tracked pixel coordinates and visibility flags.
        # Filter out the track data
        tracks = current_track_data[
            "tracks"
        ]  # Shape: (frame_num, num_points, 2) storing pixel coordinates.
        tracks = np.round(tracks).astype(
            int
        )  # Round to nearest pixel indices so they can index mask arrays.
        visibility = current_track_data[
            "visibility"
        ]  # Binary matrix (frame_num, num_points) indicating tracker confidence.
        assert (
            tracks.shape[0] == frame_num
        )  # Sanity-check that track duration matches frame count.
        num_points = np.shape(tracks)[
            1
        ]  # Total number of tracked points for this camera.

        # Locate the track points in the object mask of the first frame
        object_mask = processed_masks[0][i][
            "object"
        ]  # Binary mask describing object pixels in frame 0.
        track_object_idx = np.zeros(
            (num_points), dtype=int
        )  # Placeholder storing whether each track belongs to the object.
        for j in range(num_points):  # Evaluate every trajectory.
            if (
                visibility[0, j] == 1
            ):  # Only consider tracks visible in the first frame for classification.
                track_object_idx[j] = object_mask[
                    tracks[0, j, 0], tracks[0, j, 1]
                ]  # Mark if starting pixel lies inside object mask.
        # Locate the controller points in the controller mask of the first frame
        controller_mask = processed_masks[0][i][
            "controller"
        ]  # Binary mask highlighting controller pixels in frame 0.
        track_controller_idx = np.zeros(
            (num_points), dtype=int
        )  # Flag array tracking controller membership per trajectory.
        for j in range(num_points):
            if (
                visibility[0, j] == 1
            ):  # Only classify points visible in reference frame.
                track_controller_idx[j] = controller_mask[
                    tracks[0, j, 0], tracks[0, j, 1]
                ]  # Set flag if pixel begins within the controller mask.

        # Filter out bad tracking in other frames
        for frame_idx in range(
            1, frame_num
        ):  # Inspect every subsequent frame to drop inconsistent tracks.
            # Filter based on object_mask
            object_mask = processed_masks[frame_idx][i][
                "object"
            ]  # Object mask at current frame.
            for j in range(num_points):
                try:
                    if track_object_idx[j] == 1 and visibility[frame_idx, j] == 1:
                        if not object_mask[
                            tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                        ]:
                            visibility[frame_idx, j] = (
                                0  # Invalidate track when projected pixel leaves object mask.
                            )
                except:
                    # Sometimes the track coordinate is out of image
                    visibility[frame_idx, j] = (
                        0  # Drop coordinates that fall outside valid image bounds.
                    )
            # Filter based on controller_mask
            controller_mask = processed_masks[frame_idx][i][
                "controller"
            ]  # Controller mask at current frame.
            for j in range(num_points):
                if track_controller_idx[j] == 1 and visibility[frame_idx, j] == 1:
                    if not controller_mask[
                        tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                    ]:
                        visibility[frame_idx, j] = (
                            0  # Remove controller track when it drifts outside controller segmentation.
                        )

        # Get the track point cloud
        track_points = np.zeros(
            (frame_num, num_points, 3)
        )  # Placeholder for per-frame 3D points.
        track_colors = np.zeros(
            (frame_num, num_points, 3)
        )  # Placeholder for per-frame RGB colours.
        for frame_idx in range(
            frame_num
        ):  # For each frame, gather corresponding 3D sample from fused PCD.
            data = np.load(
                f"{pcd_path}/{frame_idx}.npz"
            )  # Load fused point cloud arrays for current frame.
            points = data["points"]  # Shape: (num_cam, H, W, 3).
            colors = data["colors"]  # Shape: (num_cam, H, W, 3).

            track_points[frame_idx][np.where(visibility[frame_idx])] = points[i][
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 0],
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 1],
            ]  # Sample 3D positions from the camera-specific point grid.
            track_colors[frame_idx][np.where(visibility[frame_idx])] = colors[i][
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 0],
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 1],
            ]  # Capture colour at the same pixel locations.

        object_points.append(
            track_points[:, np.where(track_object_idx)[0], :]
        )  # Collect only the trajectories identified as object points.
        object_colors.append(
            track_colors[:, np.where(track_object_idx)[0], :]
        )  # Store their colours for visualisation.
        object_visibilities.append(
            visibility[:, np.where(track_object_idx)[0]]
        )  # Retain visibility flags for the same subset.
        controller_points.append(
            track_points[:, np.where(track_controller_idx)[0], :]
        )  # Extract controller-associated tracks.
        controller_colors.append(
            track_colors[:, np.where(track_controller_idx)[0], :]
        )  # Save controller colours.
        controller_visibilities.append(
            visibility[:, np.where(track_controller_idx)[0]]
        )  # Save controller visibility masks.

    object_points = np.concatenate(
        object_points, axis=1
    )  # Merge object tracks from all cameras along point dimension.
    object_colors = np.concatenate(
        object_colors, axis=1
    )  # Combine object colour arrays accordingly.
    object_visibilities = np.concatenate(
        object_visibilities, axis=1
    )  # Merge object visibility matrices.
    controller_points = np.concatenate(
        controller_points, axis=1
    )  # Merge controller tracks across cameras.
    controller_colors = np.concatenate(
        controller_colors, axis=1
    )  # Merge controller colours across cameras.
    controller_visibilities = np.concatenate(
        controller_visibilities, axis=1
    )  # Merge controller visibility flags.

    track_data = {}  # Collect filtered track payload in one dictionary.
    track_data["object_points"] = object_points  # World-space object trajectories.
    track_data["object_colors"] = (
        object_colors  # Associated RGB colours for object tracks.
    )
    track_data["object_visibilities"] = (
        object_visibilities  # Frame-by-frame visibility of object tracks.
    )
    track_data["controller_points"] = (
        controller_points  # World-space controller trajectories.
    )
    track_data["controller_colors"] = controller_colors  # Controller colour samples.
    track_data["controller_visibilities"] = (
        controller_visibilities  # Controller visibility flags.
    )

    return track_data  # Provide filtered data for further refinement.


def filter_motion(track_data, neighbor_dist=0.01):
    """Suppress trajectories whose motion deviates significantly from local neighbours to reduce jitter/noise.

    Args:
        track_data (Dict[str, np.ndarray]): Output of :func:`filter_track` containing trajectories and metadata.
        neighbor_dist (float, optional): Spatial radius (metres) used to query neighbouring points when
            computing motion consistency. Defaults to 0.01.

    Returns:
        Dict[str, np.ndarray]: ``track_data`` augmented with ``object_motions_valid`` and ``controller_mask`` entries.
    """
    # Calculate the motion of each point
    object_points = track_data[
        "object_points"
    ]  # (num_frames, num_points, 3) object trajectory positions.
    object_colors = track_data["object_colors"]  # RGB colours for object tracks.
    object_visibilities = track_data[
        "object_visibilities"
    ]  # Visibility flags for each object track per frame.
    object_motions = np.zeros_like(
        object_points
    )  # Placeholder for per-frame motion vectors.
    object_motions[:-1] = (
        object_points[1:] - object_points[:-1]
    )  # Finite difference to approximate motion between consecutive frames.
    object_motions_valid = np.zeros_like(
        object_visibilities
    )  # Flags to indicate when both consecutive frames are valid.
    object_motions_valid[:-1] = np.logical_and(
        object_visibilities[:-1], object_visibilities[1:]
    )  # Mark motion as valid only when the point is visible in both frames being compared.

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(
        object_points[0, :, 1]
    )  # Determine y-range for colouring.
    y_normalized = (object_points[0, :, 1] - y_min) / (
        y_max - y_min
    )  # Normalise heights to [0, 1].
    rainbow_colors = plt.cm.rainbow(y_normalized)[
        :, :3
    ]  # Assign rainbow colours based on vertical position.

    num_frames = object_points.shape[0]  # Total number of frames.
    num_points = object_points.shape[1]  # Total number of object tracks.

    vis = (
        o3d.visualization.Visualizer()
    )  # Create Open3D window for interactive pruning supervision.
    vis.create_window()  # Display the window.
    for i in tqdm(
        range(num_frames - 1)
    ):  # Inspect each motion segment across frames (frame i -> i+1).
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            object_points[i]
        )  # Set positions for frame i.
        pcd.colors = o3d.utility.Vector3dVector(
            object_colors[i]
        )  # Use RGB colours captured from fused PCD.
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(
            pcd
        )  # Precompute nearest neighbours for the current frame.
        # modified_points = []
        # new_points = []
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):  # Evaluate each track individually.
            if object_motions_valid[i, j] == 0:
                continue  # Skip when motion definition is invalid due to missing frames.
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                object_points[i, j], neighbor_dist
            )  # Query neighbours inside the spatial radius.
            neighbors = [
                index for index in idx if object_motions_valid[i, index] == 1
            ]  # Keep neighbours with valid motion for comparison.
            if len(neighbors) < 5:
                object_motions_valid[i, j] = (
                    0  # Reject trajectories with insufficient local support.
                )
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])
            motion_diff = np.linalg.norm(
                object_motions[i, j] - object_motions[i, neighbors], axis=1
            )  # Compute deviation between point motion and its neighbours.
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                object_motions_valid[i, j] = (
                    0  # Invalidate motion when it disagrees with the majority of nearby tracks.
                )
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            object_points[i][np.where(object_motions_valid[i])]
        )  # Collect surviving points after motion filtering.
        motion_pcd.colors = o3d.utility.Vector3dVector(
            object_colors[i][np.where(object_motions_valid[i])]
        )  # Keep original colours for context.
        motion_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])]
        )  # Override colours with rainbow mapping to highlight spatial distribution.

        # modified_pcd = o3d.geometry.PointCloud()
        # modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
        # modified_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
        # )

        # new_pcd = o3d.geometry.PointCloud()
        # new_pcd.points = o3d.utility.Vector3dVector(new_points)
        # new_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
        # )
        if i == 0:
            render_motion_pcd = (
                motion_pcd  # Cache geometry pointer to update in-place across frames.
            )
            # render_modified_pcd = modified_pcd
            # render_new_pcd = new_pcd
            vis.add_geometry(
                render_motion_pcd
            )  # Add filtered object motion cloud to the viewer.
            # vis.add_geometry(render_modified_pcd)
            # vis.add_geometry(render_new_pcd)
            # Adjust the viewpoint
            view_control = (
                vis.get_view_control()
            )  # Configure default view for easier inspection.
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(
                motion_pcd.points
            )  # Update geometry with next frame's surviving points.
            render_motion_pcd.colors = o3d.utility.Vector3dVector(
                motion_pcd.colors
            )  # Update colours to match new frame.
            # render_modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
            # render_modified_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
            # )
            # render_new_pcd.points = o3d.utility.Vector3dVector(new_points)
            # render_new_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
            # )
            vis.update_geometry(
                render_motion_pcd
            )  # Trigger viewer refresh with latest selection.
            # vis.update_geometry(render_modified_pcd)
            # vis.update_geometry(render_new_pcd)
            vis.poll_events()  # Keep UI responsive while iterating.
            vis.update_renderer()  # Redraw with updated data.
        # modified_num = len(modified_points)
        # print(f"Object Frame {i}: {modified_num} points are modified")

    vis.destroy_window()  # Close the window once object motion filtering is complete.
    track_data["object_motions_valid"] = (
        object_motions_valid  # Persist validity mask for later use/visualisation.
    )

    controller_points = track_data["controller_points"]  # Controller trajectories.
    controller_colors = track_data["controller_colors"]  # Controller colour samples.
    controller_visibilities = track_data[
        "controller_visibilities"
    ]  # Controller visibility flags.
    controller_motions = np.zeros_like(
        controller_points
    )  # Placeholder for controller motion vectors.
    controller_motions[:-1] = (
        controller_points[1:] - controller_points[:-1]
    )  # Compute motion between consecutive frames.
    controller_motions_valid = np.zeros_like(
        controller_visibilities
    )  # Flags to track when controller motion is reliable.
    controller_motions_valid[:-1] = np.logical_and(
        controller_visibilities[:-1], controller_visibilities[1:]
    )  # Motion valid only if point visible in adjacent frames.
    num_points = controller_points.shape[1]  # Number of controller trajectories.
    # Filter all points that disappear in the sequence
    mask = np.prod(
        controller_visibilities, axis=0
    )  # Identify controller points visible in every frame (product = 1 when always visible).

    y_min, y_max = np.min(controller_points[0, :, 1]), np.max(
        controller_points[0, :, 1]
    )  # Determine y-range for controller colouring.
    y_normalized = (controller_points[0, :, 1] - y_min) / (
        y_max - y_min
    )  # Normalise heights.
    rainbow_colors = plt.cm.rainbow(y_normalized)[
        :, :3
    ]  # Precompute rainbow colours for persistent controller tracks.

    vis = o3d.visualization.Visualizer()  # New viewer for controller filtering.
    vis.create_window()

    for i in tqdm(range(num_frames - 1)):  # Iterate over controller motion segments.
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            controller_points[i]
        )  # Set controller point positions.
        pcd.colors = o3d.utility.Vector3dVector(
            controller_colors[i]
        )  # Use their RGB colours.
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)  # Prepare neighbour queries.
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):  # Evaluate each controller track.
            if mask[j] == 0:
                controller_motions_valid[i, j] = (
                    0  # Immediately drop tracks that are not visible in all frames.
                )
            if controller_motions_valid[i, j] == 0:
                continue  # Skip points without valid motion this frame pair.
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                controller_points[i, j], neighbor_dist
            )  # Query neighbours around current controller point.
            neighbors = [
                index for index in idx if controller_motions_valid[i, index] == 1
            ]  # Keep neighbours whose motion is defined.
            if len(neighbors) < 5:
                controller_motions_valid[i, j] = 0  # Reject under-supported tracks.
                mask[j] = 0  # Mark track as unusable globally.

            motion_diff = np.linalg.norm(
                controller_motions[i, j] - controller_motions[i, neighbors], axis=1
            )  # Compare motion vectors of neighbours.
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                controller_motions_valid[i, j] = (
                    0  # Drop track when motion deviates from neighbours.
                )
                mask[j] = 0  # Remove from globally valid set.

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            controller_points[i][np.where(mask)]
        )  # Visualise only controller points that remain valid across frames.
        motion_pcd.colors = o3d.utility.Vector3dVector(
            controller_colors[i][np.where(controller_motions_valid[i])]
        )  # Colour surviving points for context (not necessarily same indexing as mask but approximate).

        if i == 0:
            render_motion_pcd = motion_pcd  # Cache geometry for incremental updates.
            vis.add_geometry(render_motion_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            vis.update_geometry(render_motion_pcd)
            vis.poll_events()
            vis.update_renderer()

    track_data["controller_mask"] = (
        mask  # Store binary indicator of controller tracks that survived filtering.
    )
    return track_data  # Return updated track_data dictionary for subsequent processing.


def get_final_track_data(track_data, controller_threhsold=0.01):
    """Select a stable subset of controller points and keep object track metadata for downstream use.

    Args:
        track_data (Dict[str, np.ndarray]): Result dictionary produced by :func:`filter_motion`.
        controller_threhsold (float, optional): Legacy parameter retained for compatibility; unused by the
            current implementation but kept to document historical tuning knobs.

    Returns:
        Dict[str, np.ndarray]: ``track_data`` with controller trajectories reduced via farthest point sampling.
    """
    object_points = track_data[
        "object_points"
    ]  # Object trajectories for potential future use/visualisation.
    object_colors = track_data["object_colors"]  # Colours for the object trajectories.
    object_visibilities = track_data[
        "object_visibilities"
    ]  # Visibility flags for object points.
    object_motions_valid = track_data[
        "object_motions_valid"
    ]  # Motion validity mask computed earlier.
    controller_points = track_data[
        "controller_points"
    ]  # Controller trajectories retained from motion filtering.
    mask = track_data[
        "controller_mask"
    ]  # Boolean mask selecting controller tracks valid across sequence.

    new_controller_points = controller_points[
        :, np.where(mask)[0], :
    ]  # Keep only globally valid controller tracks.
    surviving = new_controller_points.shape[1]
    print(f"[Track Debug] surviving controller trajectories: {surviving}")
    if IGNORE_COTRACKER_TRAJECTORIES_TOO_LESS and surviving < 30:
        print(
            f"[Track Debug] Ignore case with too less cotracker trajectories: {surviving}"
        )
    else:
        assert surviving >= 30, (
            f"Expected at least 30 controller trajectories after filtering, "
            f"got {surviving}. Consider relaxing thresholds or improving masks."
        )  # Sanity-check that enough points survived for farthest-point sampling.
    # Do farthest point sampling on the valid controller points to select the final controller points
    valid_indices = np.arange(
        len(new_controller_points[0])
    )  # Candidate indices among surviving tracks.
    points_map = {}  # Map from 3D coordinate tuples to index for quick lookup.
    sample_points = []  # List of points used for FPS input geometry.
    for i in valid_indices:
        points_map[tuple(new_controller_points[0, i])] = (
            i  # Remember which index owns each point.
        )
        sample_points.append(new_controller_points[0, i])  # Collect points for FPS.
    sample_points = np.array(
        sample_points
    )  # Convert to numpy array for Open3D operations.
    sample_pcd = (
        o3d.geometry.PointCloud()
    )  # Build Open3D point cloud from candidate points.
    sample_pcd.points = o3d.utility.Vector3dVector(sample_points)
    fps_pcd = sample_pcd.farthest_point_down_sample(
        30
    )  # Select 30 representative controller points using FPS.
    final_indices = (
        []
    )  # Indices of points selected by FPS in the original array ordering.
    for point in fps_pcd.points:
        final_indices.append(
            points_map[tuple(point)]
        )  # Map each sampled point back to its index.

    print(
        f"Controller Point Number: {len(final_indices)}"
    )  # Report how many controller anchors remain.

    # Get the nearest controller points and their colors
    nearest_controller_points = new_controller_points[
        :, final_indices
    ]  # Keep trajectories for the selected controller anchors.

    # object_pcd = o3d.geometry.PointCloud()
    # object_pcd.points = o3d.utility.Vector3dVector(valid_object_points)
    # object_pcd.colors = o3d.utility.Vector3dVector(
    #     object_colors[0][np.where(object_motions_valid[0])]
    # )
    # controller_meshes = []
    # for j in range(nearest_controller_points.shape[1]):
    #     origin = nearest_controller_points[0, j]
    #     origin_color = [1, 0, 0]
    #     controller_meshes.append(
    #         getSphereMesh(origin, color=origin_color, radius=0.005)
    #     )
    # o3d.visualization.draw_geometries([object_pcd])
    # o3d.visualization.draw_geometries([object_pcd] + controller_meshes)

    track_data.pop(
        "controller_points"
    )  # Remove original (larger) controller set to avoid confusion.
    track_data.pop("controller_colors")  # Remove colours aligned with removed points.
    track_data.pop(
        "controller_visibilities"
    )  # Remove visibility info for removed points.
    track_data["controller_points"] = (
        nearest_controller_points  # Replace with compact controller subset.
    )

    return track_data  # Pass reduced dataset onward.


def visualize_track(track_data):
    """Play back filtered object/controller trajectories in Open3D for qualitative validation.

    Args:
        track_data (Dict[str, np.ndarray]): Dictionary containing trajectories, visibilities, and controller points.

    Returns:
        None
    """
    object_points = track_data["object_points"]  # Object positions per frame.
    object_colors = track_data["object_colors"]  # Corresponding RGB colours.
    object_visibilities = track_data[
        "object_visibilities"
    ]  # Visibility flags for object trajectories.
    object_motions_valid = track_data[
        "object_motions_valid"
    ]  # Mask of object motions surviving filtering.
    controller_points = track_data[
        "controller_points"
    ]  # Final controller anchor trajectories.

    frame_num = object_points.shape[0]  # Number of frames to visualise.

    vis = o3d.visualization.Visualizer()
    vis.create_window()  # Launch viewer window.
    controller_meshes = []  # Will hold sphere geometry per controller anchor.
    prev_center = []  # Track previous positions to update spheres efficiently.

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(
        object_points[0, :, 1]
    )  # Determine y-range for consistent colouring.
    y_normalized = (object_points[0, :, 1] - y_min) / (
        y_max - y_min
    )  # Normalise to [0, 1].
    rainbow_colors = plt.cm.rainbow(y_normalized)[
        :, :3
    ]  # Colour look-up for object trajectories.

    for i in range(frame_num):  # Iterate through all frames to animate trajectories.
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_motions_valid[i])[0], :]
        )  # Display only object points whose motion remained valid in this frame.
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])[0]]
        )  # Apply rainbow colouring for easier temporal tracking.

        if i == 0:
            render_object_pcd = object_pcd  # Store pointer to update in later frames.
            vis.add_geometry(render_object_pcd)  # Add filtered object cloud to viewer.
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]  # Controller position at frame 0.
                origin_color = [1, 0, 0]  # Colour anchors in red for visibility.
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )  # Create sphere representing controller anchor.
                vis.add_geometry(controller_meshes[-1])  # Add to viewer.
                prev_center.append(origin)  # Track centre for later motion updates.
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(
                object_pcd.points
            )  # Update point positions.
            render_object_pcd.colors = o3d.utility.Vector3dVector(
                object_pcd.colors
            )  # Update colours.
            vis.update_geometry(render_object_pcd)  # Notify Open3D of updates.
            for j in range(controller_points.shape[1]):
                origin = controller_points[
                    i, j
                ]  # New controller position in current frame.
                controller_meshes[j].translate(
                    origin - prev_center[j]
                )  # Move sphere to new location relative to previous frame.
                vis.update_geometry(controller_meshes[j])  # Refresh geometry in viewer.
                prev_center[j] = origin  # Cache position for next iteration.
            vis.poll_events()  # Process UI events while animating.
            vis.update_renderer()  # Redraw scene for current frame.


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"  # Directory containing fused point clouds per frame.
    mask_path = f"{base_path}/{case_name}/mask"  # Directory containing processed masks.
    track_path = (
        f"{base_path}/{case_name}/cotracker"  # Directory with Co-Tracker raw outputs.
    )

    num_cam = len(
        glob.glob(f"{mask_path}/mask_info_*.json")
    )  # Infer number of cameras from mask metadata files.
    frame_num = len(
        glob.glob(f"{pcd_path}/*.npz")
    )  # Count number of fused point-cloud frames.

    # Filter the track data using the semantic mask of object and controller
    track_data = filter_track(
        track_path, pcd_path, mask_path, frame_num, num_cam
    )  # Remove inconsistent trajectories using segmentation masks.
    # Filter motion
    track_data = filter_motion(
        track_data
    )  # Further prune tracks with aberrant motion patterns.
    # # Save the filtered track data
    # with open(f"test2.pkl", "wb") as f:
    #     pickle.dump(track_data, f)

    # with open(f"test2.pkl", "rb") as f:
    #     track_data = pickle.load(f)

    track_data = get_final_track_data(
        track_data
    )  # Reduce controller tracks to a representative subset via FPS.

    with open(f"{base_path}/{case_name}/track_process_data.pkl", "wb") as f:
        pickle.dump(
            track_data, f
        )  # Persist filtered trajectories for downstream optimisation.

    visualize_track(
        track_data
    )  # Launch interactive playback so users can confirm track quality.
