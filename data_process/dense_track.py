"""Track controller and object pixels across time using Meta's Co-Tracker model.

The script samples thousands of pixels inside the union of object and controller masks, feeds the
corresponding RGB frames into Co-Tracker, and stores the resulting trajectories and visibilities for
each calibrated camera. Visual overlays are also written so practitioners can quickly verify that
tracking did not drift.
"""

import torch  # Provides tensor operations and GPU access required by Co-Tracker.
import imageio.v3 as iio  # Efficient video reader used to load RGB frames into memory.
from utils.visualizer import Visualizer  # Utility for rendering tracked points over the original footage.
import glob  # Enumerates files to determine available cameras and masks.
import cv2  # Handles mask image decoding so we can compute sampling regions.
import numpy as np  # Supplies vectorised math for sampling and preprocessing.
import os  # Manages filesystem operations such as directory creation.
from argparse import ArgumentParser  # Parses command-line arguments specifying dataset locations.

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path  # Root directory containing all case folders.
case_name = args.case_name  # Identifier for the specific capture being processed.

num_cam = 3  # Dataset assumes a fixed three-camera rig; adjust here if hardware changes.
assert (
    len(glob.glob(f"{base_path}/{case_name}/depth/*")) == num_cam
), "Depth folders missing for one or more cameras."
device = "cuda"  # Co-Tracker inference is performed on the GPU for efficiency.


def read_mask(mask_path):
    """Read a mask image and convert it into a boolean array.

    Args:
        mask_path (str): Absolute or relative path to a binary mask written by SAM2.

    Returns:
        np.ndarray: ``True`` for foreground pixels, ``False`` for background.
    """

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as 8-bit grayscale intensities.
    mask = mask > 0  # Treat all non-zero values as foreground.
    return mask


def exist_dir(dir):
    """Create ``dir`` if it does not already exist.

    Args:
        dir (str): Directory path that should be present before writing output files.

    Returns:
        None
    """

    if not os.path.exists(dir):  # Skip creation when the directory already exists.
        os.makedirs(dir)  # Recursively create directory structure.


if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")  # Ensure a directory exists for storing trajectory outputs.

    for i in range(num_cam):
        print(f"Processing {i}th camera")  # Provide visibility into which camera stream is currently handled.

        # Load the full RGB video into memory. ``FFMPEG`` is used to ensure compatibility with MP4 files.
        frames = iio.imread(f"{base_path}/{case_name}/color/{i}.mp4", plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # Reshape to ``B (batch) x T (frames) x C x H x W`` and move to GPU for Co-Tracker.

        # Gather all masks from the first frame so we can sample seed queries covering both object and controller.
        mask_paths = glob.glob(f"{base_path}/{case_name}/mask/{i}/*/0.png")
        mask = None  # Accumulate a union of all relevant masks so both object and controller are covered.
        for mask_path in mask_paths:
            current_mask = read_mask(mask_path)
            if mask is None:
                mask = current_mask  # Initialise with the first mask encountered.
            else:
                mask = np.logical_or(mask, current_mask)  # Union subsequent masks into a single sampling region.

        # Convert the 2D mask into a list of (row, col) indices representing candidate pixels to track.
        query_pixels = np.argwhere(mask)
        query_pixels = query_pixels[:, ::-1]  # Swap to (x, y) ordering expected by Co-Tracker.
        query_pixels = np.concatenate(
            [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
        )  # Prepend a zero time index because queries are defined on the first frame.
        query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)

        # Randomly shuffle candidate pixels and cap the set at 5000 points to keep runtime manageable.
        if query_pixels.shape[0] > 0:
            sample_order = torch.randperm(
                query_pixels.shape[0], device=query_pixels.device
            )  # Random permutation on the same device as ``query_pixels``.
            if sample_order.numel() > 5000:
                sample_order = sample_order[:5000]
            query_pixels = query_pixels[sample_order]

        # Instantiate the online Co-Tracker model. The online variant handles long videos by iterative updates.
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])  # Prime the tracker with the first chunk and query set.

        # Slide a temporal window across the video so the online tracker can incrementally update trajectories.
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # Returns per-query pixel locations ``(B, T, N, 2)`` and visibility flags ``(B, T, N, 1)``.

        # Render qualitative overlays showing the tracked points to aid manual validation.
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")

        # Persist the raw trajectories and visibilities for downstream processing scripts.
        track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]  # Convert to numpy and flip coordinates back to (row, col).
        visibility_to_save = pred_visibility[0].cpu().numpy()
        np.savez(
            f"{base_path}/{case_name}/cotracker/{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )
