"""Orchestrate segmentation for each calibrated camera by invoking the video utility.

The script loops over all calibrated cameras for a given capture, runs the GroundingDINO+SAM2 video
segmentation helper, and then prunes temporary artefacts created during the process. It assumes each
dataset contains exactly three camera folders to align with the rig used during data collection.
"""
import os  # Executes helper scripts and cleans temporary folders between camera runs.
import glob  # Determines how many camera depth folders are available for the case.
from argparse import ArgumentParser  # Handles command-line configuration for segmentation.

# Configure the CLI so callers can specify which dataset and prompt to process.
parser = ArgumentParser()  # Parser describing segmentation inputs.
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)  # Root folder containing all captured cases.
parser.add_argument("--case_name", type=str, required=True)  # Name of the specific capture sequence to segment.
parser.add_argument("--TEXT_PROMPT", type=str, required=True)  # Natural-language grounding prompt shared across cameras.
args = parser.parse_args()  # Resolve CLI inputs immediately.

# Cache the parsed arguments for subsequent use.
base_path = args.base_path  # Dataset root provided by the caller.
case_name = args.case_name  # Target case directory inside base_path.
TEXT_PROMPT = args.TEXT_PROMPT  # Text description feeding GroundingDINO + SAM2.
camera_num = 3  # Expect exactly three calibrated cameras in the dataset layout.
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num  # Validate dataset completeness before processing.
print(f"Processing {case_name}")  # Emit status so multi-case batch logs remain readable.

# Iterate over each camera stream, generate masks, and discard temporary frame dumps per iteration.
for camera_idx in range(camera_num):
    print(f"Processing {case_name} camera {camera_idx}")  # Clarify which camera is currently being segmented.
    os.system(
        f"python ./data_process/segment_util_video.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}"
    )  # Delegate the heavy lifting to the SAM2 + GroundingDINO video utility.
    os.system(f"rm -rf {base_path}/{case_name}/tmp_data")  # Clean temporary frame extraction cache to save disk space.
