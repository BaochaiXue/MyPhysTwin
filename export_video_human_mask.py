"""Generate per-frame human masks for all captured videos."""

from __future__ import annotations

import os
import glob

base_path: str = "./data/different_types"
output_path: str = "./data/different_types_human_mask"

def existDir(dir_path: str) -> None:
    """Create ``dir_path`` if it does not exist."""

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Iterate through all captured scenes and compute human masks for each frame.
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    existDir(f"{output_path}/{case_name}")
    # Process to get the whole human mask for the video

    # Use a fixed prompt for the segmentation model
    TEXT_PROMPT = "human"
    camera_num = 3
    # Ensure expected number of cameras are available
    assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num

    for camera_idx in range(camera_num):
        print(f"Processing {case_name} camera {camera_idx}")
        # Run the segmentation script for each camera frame
        os.system(
            f"python ./data_process/segment_util_video.py --output_path {output_path}/{case_name} --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}"
        )
        # Clean up temporary files produced by the segmentation script
        os.system(f"rm -rf {base_path}/{case_name}/tmp_data")