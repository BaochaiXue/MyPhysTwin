"""Segment a single-camera video using a GroundingDINO prompt and SAM2 propagation.

The script extracts frames from the requested video, uses GroundingDINO to obtain an initial
bounding box driven by a natural-language prompt, and then propagates the segmentation across
all frames with SAM2's video predictor. Results are saved as per-frame binary masks alongside
metadata describing which numeric identifier corresponds to each textual label.
"""

import os  # Launches helper scripts, checks for file existence, and manages folder creation.
import torch  # Provides tensor utilities, GPU introspection, and mixed-precision inference helpers.
import numpy as np  # Supplies vectorised array operations for mask post-processing.
import supervision as sv  # Convenience utilities for decoding/encoding video frames.
from torchvision.ops import box_convert  # Transforms bounding boxes between coordinate conventions.
from pathlib import Path  # Filesystem path helper for cross-platform directory creation.
from tqdm import tqdm  # Progress bars for frame export and SAM2 propagation.
from PIL import Image  # Persists binary masks as PNG files.
from sam2.build_sam import build_sam2_video_predictor, build_sam2  # Factories for constructing SAM2 models.
from sam2.sam2_image_predictor import SAM2ImagePredictor  # High-level API for single-image SAM2 inference.
from groundingdino.util.inference import load_model, load_image, predict  # GroundingDINO inference helpers.
import json  # Serialises the class-id to label mapping for downstream use.
from argparse import ArgumentParser  # Parses command-line arguments configuring the run.


parser = ArgumentParser(description="Video segmentation utility driven by text prompts.")
parser.add_argument(
    "--base_path",
    type=str,
    default="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect",
    help="Root directory that stores all captured cases. Defaults to the development dataset path.",
)
parser.add_argument("--case_name", type=str, help="Identifier for the sequence being segmented.")
parser.add_argument(
    "--TEXT_PROMPT",
    type=str,
    help="Natural-language description of the object we want SAM2 to isolate.",
)
parser.add_argument(
    "--camera_idx",
    type=int,
    help="Zero-based index of the camera whose video stream will be processed.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="NONE",
    help="Optional override for where segmentation artefacts should be written.",
)
args = parser.parse_args()

base_path = args.base_path  # Dataset root directory containing the case folders.
case_name = args.case_name  # Folder name for the current capture.
TEXT_PROMPT = args.TEXT_PROMPT  # Natural-language grounding string guiding GroundingDINO.
camera_idx = args.camera_idx  # Camera stream index to segment.
if args.output_path == "NONE":
    output_path = f"{base_path}/{case_name}"  # Default to writing results inside the case directory.
else:
    output_path = args.output_path  # Honour the caller-provided override path.


def existDir(dir_path: str) -> None:
    """Create ``dir_path`` (including parents) if it does not already exist.

    Args:
        dir_path (str): Absolute or relative directory that should be present before writing files.

    Returns:
        None: The function performs in-place filesystem mutations and leaves discovery to callers.
    """

    if not os.path.exists(dir_path):  # Avoid redundant directory creation.
        os.makedirs(dir_path)  # Recursively build the directory tree.


GROUNDING_DINO_CONFIG = "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"  # Model configuration describing network architecture.
GROUNDING_DINO_CHECKPOINT = "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"  # Weights tuned for open-vocabulary detection.
BOX_THRESHOLD = 0.35  # Minimum box confidence required to treat a detection as a valid prompt.
TEXT_THRESHOLD = 0.25  # Minimum text matching score accepted from GroundingDINO.
PROMPT_TYPE_FOR_VIDEO = "box"  # SAM2 video predictor accepts "box", "mask", or "point" prompts.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Prefer GPU inference when available for speed.

VIDEO_PATH = f"{base_path}/{case_name}/color/{camera_idx}.mp4"  # Path to the colour video captured by the requested camera.
existDir(f"{base_path}/{case_name}/tmp_data")  # Root cache directory for extracted frames.
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}")  # Namespace cache by case to avoid collisions during batch jobs.
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}")  # Separate per-camera frame dumps.

SOURCE_VIDEO_FRAME_DIR = f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}"  # Directory storing colour frames that SAM2 will consume.

# -----------------------------------------------------------------------------
# Step 1: Environment settings and model initialization for GroundingDINO + SAM2
# -----------------------------------------------------------------------------
# Build the GroundingDINO detector used to obtain an initial bounding-box prompt guided by the text query.
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)

# Initialise single-image and video SAM2 predictors so we can sample a prompt on the key frame and propagate masks.
sam2_checkpoint = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"  # Path to SAM2 checkpoint tuned for hierarchical large model.
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Configuration file describing the SAM2 hierarchy.

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)  # Handles temporal propagation across frames.
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)  # Vanilla SAM2 model that processes single images.
image_predictor = SAM2ImagePredictor(sam2_image_model)  # High-level wrapper providing ``predict`` with bounding-box prompts.

# Gather metadata about the video so we know how many frames to expect.
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # Read resolution, FPS, and frame count from the source video.
print(video_info)  # Log the video summary to make debugging easier.
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)  # Iterate over every frame sequentially.

# Persist the frames to disk so SAM2 can operate on image directories.
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)  # Wrap target directory in ``Path`` for ergonomic creation.
source_frames.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists prior to writing frames.

with sv.ImageSink(
    target_dir_path=source_frames,
    overwrite=True,
    image_name_pattern="{:05d}.jpg",
) as sink:  # Write each frame using a zero-padded numeric filename for deterministic ordering.
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):  # Iterate through video frames with a progress bar.
        sink.save_image(frame)  # Flush the current frame to disk.

# Scan all saved frame names so we can keep a deterministic ordering when feeding the video predictor.
frame_names = [
    p
    for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))  # Sort numerically to match chronological frame order.

# Initialise SAM2's video predictor state machine using the extracted frames.
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # Reference frame index where we will sample the manual prompt.

# -----------------------------------------------------------------------------
# Step 2: Prompt GroundingDINO on the reference frame to obtain bounding boxes
# -----------------------------------------------------------------------------
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])  # Path to the frame chosen for prompting.
image_source, image = load_image(img_path)  # Load the frame in the format expected by GroundingDINO.

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)  # Perform open-vocabulary detection guided by the user supplied text prompt.

# Convert relative (cx, cy, w, h) boxes into pixel-space ``xyxy`` format for SAM2.
h, w, _ = image_source.shape  # Extract spatial dimensions from the source image.
boxes = boxes * torch.Tensor([w, h, w, h])  # Scale normalised coordinates back into pixel units.
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()  # Reformat boxes to match SAM2's expected input.
confidences = confidences.numpy().tolist()  # Convert detector confidences into Python floats for logging.
class_names = labels  # Keep the textual class names returned by GroundingDINO.

print(input_boxes)  # Expose the detected boxes for troubleshooting mis-grounded prompts.

# Feed the reference frame into SAM2's image predictor so we can sample an initial mask.
image_predictor.set_image(image_source)
OBJECTS = class_names  # Record the class ordering; each entry will map to a numeric mask identifier.
print(OBJECTS)  # Helpful debug line that mirrors the textual labels used downstream.

# -----------------------------------------------------------------------------
# Step 3: Register the prompt with the SAM2 video predictor
# -----------------------------------------------------------------------------
assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompts"

# Enable mixed precision where possible to reduce memory pressure without sacrificing quality.
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # TensorFloat-32 significantly accelerates matmul/convolution on Ampere+ GPUs while keeping accuracy acceptable.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes)):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )  # Register each detected object as a separate track inside the video predictor.
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

# -----------------------------------------------------------------------------
# Step 4: Propagate the segmentation through the entire video
# -----------------------------------------------------------------------------
video_segments = {}  # Maps frame indices to dictionaries of {object_id: binary mask}.
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
    inference_state
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }  # Threshold mask logits at 0 to obtain boolean masks per object.

# -----------------------------------------------------------------------------
# Step 5: Persist masks and class metadata to disk
# -----------------------------------------------------------------------------
existDir(f"{output_path}/mask/")  # Root directory for per-camera semantic masks.
existDir(f"{output_path}/mask/{camera_idx}")  # Camera-specific directory that hosts per-object folders.

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}  # Map numeric IDs to human-readable labels.

with open(f"{output_path}/mask/mask_info_{camera_idx}.json", "w") as f:
    json.dump(ID_TO_OBJECTS, f)  # Log which identifier represents the controller vs. object for downstream scripts.

for frame_idx, masks in video_segments.items():  # Iterate over each frame returned by SAM2.
    for obj_id, mask in masks.items():  # Persist every object's mask independently.
        existDir(f"{output_path}/mask/{camera_idx}/{obj_id}")  # Create directory for the current object ID if missing.
        Image.fromarray((mask[0] * 255).astype(np.uint8)).save(
            f"{output_path}/mask/{camera_idx}/{obj_id}/{frame_idx}.png"
        )  # SAM2 outputs HxW arrays; convert to 8-bit and save as PNG for compatibility with downstream code.
