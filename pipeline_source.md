## process_data.py

```python
"""Top-level orchestration script for the MyPhysTwin data processing pipeline.

The module parses command-line arguments describing the dataset root, case name, and segmentation prompt,
then executes each downstream processing stage (segmentation, tracking, reconstruction, alignment, sampling)
while logging timings via a shared context manager. Individual stages can be toggled via module-level flags
to support incremental debugging or reprocessing specific artefacts.
"""

import os  # Standard library module for filesystem inspection and launching subprocesses.
from argparse import ArgumentParser  # Utility for parsing command-line arguments that configure the pipeline.
import time  # Provides wall-clock timing used by the profiling Timer context manager.
import logging  # Enables structured logging to both console and persistent files.
import json  # Supports reading mask metadata and writing dataset splits in JSON form.
import glob  # Supplies fast pattern matching for counting generated frames.

# Construct the command-line parser so callers can tailor where data lives and which features to run.
parser = ArgumentParser()  # CLI parser describing the data processing pipeline entry point.
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
)  # Base directory that holds all captured cases.
parser.add_argument("--case_name", type=str, required=True)  # Identifier for the specific capture to process.
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)  # Text label used by GroundingDINO/SAM2 to find the manipulated object.
parser.add_argument("--shape_prior", action="store_true", default=False)  # Flag toggling generation and alignment of a shape prior asset.
args = parser.parse_args()  # Parse CLI arguments immediately so the rest of the module can access user preferences.

# Configure developer toggles so individual stages can be skipped during debugging without altering CLI usage.
PROCESS_SEG = True  # Controls whether raw videos are segmented into controller/object masks.
PROCESS_SHAPE_PRIOR = True  # Controls running the upscaling + TRELLIS shape-prior generation chain.
PROCESS_TRACK = True  # Controls Co-Tracker post-processing for dense correspondences.
PROCESS_3D = True  # Controls lifting RGB-D streams into fused world-coordinate point clouds.
PROCESS_ALIGN = True  # Controls alignment of the generated shape prior against observations.
PROCESS_FINAL = True  # Controls final sampling and dataset split export.

# Cache argument values in module-level variables for convenience throughout the script.
base_path = args.base_path  # Root directory of the current dataset collection.
case_name = args.case_name  # Specific case folder name under base_path.
category = args.category  # Natural-language description of the manipulated object.
TEXT_PROMPT = f"{category}.hand"  # Prompt presented to the segmentation models (object category + controller keyword).
CONTROLLER_NAME = "hand"  # Semantic label used to distinguish the controller from the object in mask metadata.
SHAPE_PRIOR = args.shape_prior  # Boolean controlling whether to build and align a shape prior for this case.

logger = None  # Module-scoped logger instance shared across all stages; created lazily.


def setup_logger(log_file="timer.log"):
    """Initialise the global logger so stage timings land in both stdout and a persistent log file.

    Args:
        log_file (str): Destination file used to persist timing information across pipeline runs.

    Returns:
        logging.Logger: Shared logger instance configured with both console and file handlers.
    """
    global logger  # Allow updates to the module-level logger reference.

    if logger is None:  # Only build handlers if the logger has not been configured yet.
        logger = logging.getLogger("GlobalLogger")  # Create a dedicated logger namespace for the pipeline.
        logger.setLevel(logging.INFO)  # Emit high-level status updates without overwhelming verbosity.

        if not logger.handlers:  # Ensure duplicate handlers are not attached when script reruns in the same process.
            file_handler = logging.FileHandler(log_file)  # Persist timing information to disk for later inspection.
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))  # Standardise file log message format.

            console_handler = logging.StreamHandler()  # Mirror log output to the console for immediate feedback.
            console_handler.setFormatter(logging.Formatter("%(message)s"))  # Keep console messages compact.

            logger.addHandler(file_handler)  # Attach file handler to global logger.
            logger.addHandler(console_handler)  # Attach console handler to global logger.


setup_logger()  # Instantiate the logger as soon as the script is imported/run so subsequent stages can report progress.


def existDir(dir_path):
    """Create ``dir_path`` if it does not already exist to avoid race conditions later.

    Args:
        dir_path (str): Absolute or relative directory path that must exist before subsequent file writes.

    Returns:
        None: The function mutates filesystem state in-place and intentionally returns nothing.
    """
    if not os.path.exists(dir_path):  # Skip creation when directory already present.
        os.makedirs(dir_path)  # Recursively create the requested directory structure.


class Timer:
    """Context manager that records and logs how long a processing stage takes.

    The helper wraps each major pipeline step with consistent logging so operators can audit performance and
    quickly identify bottlenecks. It emits a start banner upon entering the context and a timing summary when
    exiting, regardless of whether the enclosed block succeeds or raises.

    Args:
        task_name (str): Human-readable label for the stage being timed. The name appears verbatim in log output.

    Attributes:
        task_name (str): Cached stage description used in the log banners.
        start_time (float): UNIX timestamp captured on ``__enter__`` and consumed during ``__exit__``.
    """

    def __init__(self, task_name):
        self.task_name = task_name  # Human-readable stage name displayed in logs.

    def __enter__(self):
        """Capture the wall-clock start time and emit a banner announcing the stage.

        Returns:
            None: Context managers optionally return a value, but this implementation only performs side effects.
        """
        self.start_time = time.time()  # Capture the wall-clock start time upon entering the context.
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )  # Emit a banner indicating the stage has begun.

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Compute duration and log the elapsed wall-clock time on context exit.

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type raised inside the context, if any.
            exc_val (Optional[BaseException]): Concrete exception instance propagated from the context body.
            exc_tb (Optional[TracebackType]): Python traceback describing where the exception occurred.

        Returns:
            None: The method logs timing information and allows any exception to propagate naturally.
        """
        elapsed_time = time.time() - self.start_time  # Compute elapsed seconds when leaving the context.
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )  # Log the duration regardless of success/failure for performance tracking.


if PROCESS_SEG:  # Execute video segmentation unless the developer explicitly disabled this stage.
    # Launch the multi-camera segmentation script that produces per-frame SAM2 masks for controller and object.
    with Timer("Video Segmentation"):
        os.system(
            f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT}"
        )  # Delegate segmentation to the specialised helper so this orchestrator stays lightweight.


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:  # Only build a shape prior when both the debug flag and CLI request are true.
    # Inspect the mask metadata to locate the object instance index for subsequent cropping.
    with open(f"{base_path}/{case_name}/mask/mask_info_{0}.json", "r") as f:
        data = json.load(f)  # Load the mapping from mask IDs to semantic labels produced during segmentation.
    obj_idx = None  # Accumulate the single non-controller object ID; assumption is exactly one object per sequence.
    for key, value in data.items():  # Iterate over all mask identifiers registered for the reference camera.
        if value != CONTROLLER_NAME:  # Only record the entry that corresponds to the manipulated object.
            if obj_idx is not None:  # Detect unexpected situations with multiple object labels.
                raise ValueError("More than one object detected.")  # Fail fast to avoid mixing masks from different objects.
            obj_idx = int(key)  # Store the canonical object mask index for later file lookups.
    mask_path = f"{base_path}/{case_name}/mask/0/{obj_idx}/0.png"  # Compose the path to the first-frame object mask.

    existDir(f"{base_path}/{case_name}/shape")  # Ensure the shape prior output directory exists before writing assets.
    # Generate a high-resolution reference crop to feed the TRELLIS generator; skip work if cached from an earlier run.
    with Timer("Image Upscale"):
        if not os.path.isfile(f"{base_path}/{case_name}/shape/high_resolution.png"):
            os.system(
                f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/0/0.png --mask_path {mask_path} --output_path {base_path}/{case_name}/shape/high_resolution.png --category {category}"
            )  # Run the diffusion-based upscaler over the masked region to obtain a clean high-res crop.

    # Re-segment the upscaled still image to isolate the object with an alpha channel for TRELLIS consumption.
    with Timer("Image Segmentation"):
        os.system(
            f"python ./data_process/segment_util_image.py --img_path {base_path}/{case_name}/shape/high_resolution.png --TEXT_PROMPT {category} --output_path {base_path}/{case_name}/shape/masked_image.png"
        )  # Produce a clean RGBA crop (background removed) aligned with the diffusion output.

    # Invoke TRELLIS to synthesise a textured mesh and gaussian representation from the masked crop.
    with Timer("Shape Prior Generation"):
        os.system(
            f"python ./data_process/shape_prior.py --img_path {base_path}/{case_name}/shape/masked_image.png --output_dir {base_path}/{case_name}/shape"
        )  # Save TRELLIS outputs (GLB, PLY, renders) to the shape directory for downstream alignment.

if PROCESS_TRACK:  # Optional stage for dense tracking across frames.
    # Run the Co-Tracker-based tracking pipeline to produce 2D correspondences for object/controller pixels.
    with Timer("Dense Tracking"):
        os.system(
            f"python ./data_process/dense_track.py --base_path {base_path} --case_name {case_name}"
        )  # Creates per-camera .npz files with tracked pixel coordinates and visibilities.

if PROCESS_3D:  # Execute world-coordinate reconstruction steps when enabled.
    # Fuse multi-view RGB-D frames into per-frame world-coordinate point clouds.
    with Timer("Lift to 3D"):
        os.system(
            f"python ./data_process/data_process_pcd.py --base_path {base_path} --case_name {case_name}"
        )  # Generates {points, colors, masks} tensors for each frame by merging calibrated depth streams.

    # Clean the raw masks using 3D outlier rejection and propagate results back to image space.
    with Timer("Mask Post-Processing"):
        os.system(
            f"python ./data_process/data_process_mask.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )  # Produces refined binary masks that exclude isolated depth noise.

    # Filter and restructure tracking outputs into world-space trajectories.
    with Timer("Data Tracking"):
        os.system(
            f"python ./data_process/data_process_track.py --base_path {base_path} --case_name {case_name}"
        )  # Writes consolidated object/controller trajectories and motion statistics.

if PROCESS_ALIGN and SHAPE_PRIOR:  # Align the generated mesh to observations only when a prior is available.
    # Optimise the TRELLIS mesh pose using multi-view correspondences and ARAP deformation.
    with Timer("Alignment"):
        os.system(
            f"python ./data_process/align.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )  # Produces a refined mesh registered to the observed scene and writes diagnostic visualisations.

if PROCESS_FINAL:  # Always export the final samples unless developer debugging disables the stage.
    # Sample fused point clouds for training/inference, optionally conditioning on the aligned shape prior.
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:  # Use the shape-prior aware sampler when a prior was generated/aligned.
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name} --shape_prior"
            )  # Exports model-ready samples with prior-specific metadata and filtering.
        else:  # Fall back to vanilla sampling when no prior is requested.
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name}"
            )  # Emits the same sample structure without prior-specific channels.

    # Record train/test splits so downstream training scripts know which frames to use for each partition.
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))  # Count the total number of reconstructed frames.
    split = {}  # Dictionary structure that mirrors expected JSON schema.
    split["frame_len"] = frame_len  # Store total frame count for reference.
    split["train"] = [0, int(frame_len * 0.7)]  # Use the first 70% of frames for training data.
    split["test"] = [int(frame_len * 0.7), frame_len]  # Reserve the remaining frames for evaluation.
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)  # Persist the split summary for other scripts to consume.
```

## data_process/segment.py

```python
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
```

## data_process/segment_util_video.py

```python
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
```

## data_process/image_upscale.py

```python
"""Upscale the reference-frame crop that will seed shape-prior generation.

This standalone utility receives the low-resolution frame containing the target object along with an
optional binary mask that isolates the object. It crops around the mask to remove background clutter,
feeds the crop into the Stable Diffusion x4 upscaler, and saves the high-resolution result that will
later be segmented again for TRELLIS. Assumes the caller provides GPU access and that the diffusion
weights are available locally or via the Hugging Face hub.

Inputs:
    --img_path (str): Path to the RGB frame captured from the reference camera.
    --mask_path (Optional[str]): Binary mask locating the object; improves cropping accuracy.
    --output_path (str): Destination for the 4x upscaled crop written as PNG.
    --category (str): Text description (e.g. "banana") injected into the upscaling prompt to steer appearance.

Outputs:
    A PNG image at ``output_path`` containing an enlarged crop around the object with additional context.
    The script mutates no other state beyond downloading/initialising the diffusion weights when required.
"""

from PIL import Image  # Handles image loading/saving and cropping operations.
from diffusers import StableDiffusionUpscalePipeline  # Provides the diffusion upscaler model for 4x enhancement.
import torch  # Supplies tensor dtype/device utilities for the diffusion pipeline.
from argparse import ArgumentParser  # Parses command-line inputs specifying paths and prompts.
import cv2  # Used for reading binary masks and further image manipulation.
import numpy as np  # Enables numeric operations such as bounding-box calculations.

# Expose command-line arguments so the upscaling utility can be reused in other scripts.
parser = ArgumentParser()  # CLI parser describing the upscaling inputs.
parser.add_argument(
    "--img_path",
    type=str,
)  # Low-resolution RGB image to enhance.
parser.add_argument("--mask_path", type=str, default=None)  # Optional binary mask isolating the object of interest.
parser.add_argument("--output_path", type=str)  # Destination path for the generated high-resolution crop.
parser.add_argument("--category", type=str)  # Semantic category used inside the diffusion prompt for better fidelity.
args = parser.parse_args()  # Parse provided arguments immediately.

# Cache argument values for readability.
img_path = args.img_path  # Source image path (should be 512x512 RGB captured frame).
mask_path = args.mask_path  # Optional mask path used to crop around the object.
output_path = args.output_path  # Target location for the enhanced image.
category = args.category  # Text snippet describing the object (e.g., "banana").


# Load the pretrained diffusion upscaler. Half precision keeps VRAM usage in check without sacrificing quality.
model_id = "stabilityai/stable-diffusion-x4-upscaler"  # Name of pretrained diffusion upscaler hosted on HuggingFace.
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)  # Instantiate the pipeline in half precision to balance speed and memory.
pipeline = pipeline.to("cuda")  # Move the model to GPU for faster inference.

# Read the low-resolution frame and optionally crop it around the mask to focus the diffusion model.
low_res_img = Image.open(img_path).convert("RGB")  # Load the low-res input and guarantee RGB ordering.
if mask_path is not None:  # Apply mask-guided cropping when a mask is supplied.
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the binary mask (white pixels denote the object).
    bbox = np.argwhere(mask > 0.8 * 255)  # Collect coordinates of confident foreground pixels.
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])  # Derive tight bounding box (x_min, y_min, x_max, y_max).
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Compute object centre to build a square crop.
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])  # Use the largest dimension to keep the object fully framed.
    size = int(size * 1.2)  # Expand the crop by 20% to include context around the object.
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2  # Convert to a padded bounding box expressed as (x0, y0, x1, y1).
    low_res_img = low_res_img.crop(bbox)  # type: ignore  # Crop the PIL image to focus the upscaler on the object.

prompt = f"Hand manipulates a {category}."  # Keep context about the controller to preserve joint boundaries in the upscale.

# Execute the diffusion pipeline; ``images[0]`` is the enhanced crop returned as a PIL image.
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save(output_path)  # Persist the enhanced crop for downstream shape-prior generation.
```

## data_process/segment_util_image.py

```python
"""Single-image segmentation helper combining GroundingDINO prompts with SAM2 masking.

The script isolates the target object inside a still image by first requesting bounding boxes from
GroundingDINO given a natural-language prompt, then feeding those boxes into SAM2 to obtain a high-quality
mask. The result is saved as an RGBA image where the alpha channel retains only the prompted object.
"""

import cv2  # Provides image I/O and manipulation utilities for mask generation.
import torch  # Supplies tensor operations and device detection for model inference.
import numpy as np  # Handles numerical calculations such as bounding-box transforms.
from torchvision.ops import box_convert  # Converts bounding boxes between coordinate conventions.
from sam2.build_sam import build_sam2  # Factory for SAM2 image model construction.
from sam2.sam2_image_predictor import SAM2ImagePredictor  # High-level API exposing SAM2 image segmentation.
from groundingdino.util.inference import load_model, load_image, predict  # GroundingDINO helpers for detection-style prompting.
from argparse import ArgumentParser  # Parses command-line arguments controlling inputs and prompts.

"""
Hyper parameters
"""

# Set up CLI arguments so this utility can be invoked from other scripts.
parser = ArgumentParser()  # Parser describing required segmentation inputs.
parser.add_argument(
    "--img_path",
    type=str,
)  # Path to the RGB image that should be segmented.
parser.add_argument("--output_path", type=str)  # Destination path for the generated RGBA mask composite.
parser.add_argument("--TEXT_PROMPT", type=str)  # Natural-language prompt used to locate the object within the image.
args = parser.parse_args()  # Resolve CLI arguments immediately.

img_path = args.img_path  # Source image path captured earlier in the pipeline.
output_path = args.output_path  # Target file path for writing the RGBA mask result.
TEXT_PROMPT = args.TEXT_PROMPT  # Text description guiding GroundingDINO to the correct object.

SAM2_CHECKPOINT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"  # Local weight file for SAM2 (image mode).
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Configuration describing model architecture.
GROUNDING_DINO_CONFIG = (
    "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
)  # Config file referencing the Swin-T version of GroundingDINO.
GROUNDING_DINO_CHECKPOINT = (
    "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
)  # Checkpoint file containing pretrained weights for object grounding.
BOX_THRESHOLD = 0.35  # Minimum box score from GroundingDINO to treat detections as valid prompts.
TEXT_THRESHOLD = 0.25  # Minimum text score ensuring the textual match is confident.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Select GPU when available for faster inference.

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT  # Alias for clarity when instantiating SAM2.
model_cfg = SAM2_MODEL_CONFIG  # Configuration path passed into builder.
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)  # Load SAM2 weights on the chosen device.
sam2_predictor = SAM2ImagePredictor(sam2_model)  # Wrap the raw model in a predictor API exposing `.predict`.

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)  # Instantiate GroundingDINO to propose bounding boxes based on the text prompt.


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT  # Use the provided prompt directly (caller responsible for formatting guidance).

image_source, image = load_image(img_path)  # Read the image in both numpy and tensor-friendly formats expected by GroundingDINO.

sam2_predictor.set_image(image_source)  # Provide the RGB frame to SAM2 so future predictions reuse shared features.

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)  # Run GroundingDINO to obtain candidate bounding boxes for the described object.

# process the box prompt for SAM 2
h, w, _ = image_source.shape  # Grab original resolution for scaling bounding boxes.
boxes = boxes * torch.Tensor([w, h, w, h])  # Convert relative coordinates (cx, cy, w, h) into pixel space.
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()  # Translate to (x_min, y_min, x_max, y_max) format for SAM2.

conf_values = (
    confidences.detach().cpu().numpy().tolist()
    if hasattr(confidences, "detach")
    else confidences
)  # Convert detection confidences into plain Python lists for logging/debugging.
print(
    f"[GroundingDINO Debug] boxes shape={input_boxes.shape}, confidences={conf_values}"
)  # Provide visibility into detection outcomes for troubleshooting.


# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()  # Enable mixed-precision inference to reduce GPU memory footprint.

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat-32 for faster matmuls on supported hardware.
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 inside cuDNN convolutions for additional speed.

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)  # Request SAM2 to produce the object mask guided solely by the bounding boxes.

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)  # Remove singleton dimension when SAM2 returns masks with explicit channel axis.


confidences = confidences.numpy().tolist()  # Convert confidences back to vanilla Python lists for downstream prints.
class_names = labels  # Capture the textual labels returned by GroundingDINO.

OBJECTS = class_names  # Keep track of the order in which objects were detected (likely length 1 for this pipeline).

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}  # Map integer indices to readable class names (debugging aid).

print(f"Detected {len(masks)} objects")  # Show how many masks SAM2 produced with the configured prompt.

raw_img = cv2.imread(img_path)  # Reload the image via OpenCV to extract raw pixel values for compositing.
mask_img = (masks[0] * 255).astype(np.uint8)  # Convert the first predicted mask into an 8-bit grayscale image.

ref_img = np.zeros((h, w, 4), dtype=np.uint8)  # Prepare an RGBA canvas initialised to transparent.
mask_bool = mask_img > 0  # Interpret non-zero mask pixels as foreground.
ref_img[mask_bool, :3] = raw_img[mask_bool]  # Copy RGB pixels from the original image wherever the mask is active.
ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255  # Set alpha channel: opaque for foreground, transparent otherwise.
cv2.imwrite(output_path, ref_img)  # Write the composited RGBA image for TRELLIS consumption.
```

## data_process/shape_prior.py

```python
"""Generate 3D shape priors from a single masked image using TRELLIS.

The script assumes ``image_upscale.py`` and ``segment_util_image.py`` have produced an RGBA crop of the
object with transparent background. It runs the TRELLIS image-to-3D pipeline to obtain gaussian splats,
textured meshes, and diagnostic turntable renderings. Outputs are written to the provided ``--output_dir``
and later consumed by the alignment stage.

Inputs:
    --img_path (str): Path to the high-resolution RGBA crop of the target object.
    --output_dir (str): Directory where TRELLIS assets (GLB, PLY, videos) should be saved.

Outputs:
    visualization.mp4 (diagnostic turntable), object.glb (textured mesh), object.ply (gaussian point cloud),
    plus any auxiliary files produced by TRELLIS.

Prerequisites:
    Environment variable ``SPCONV_ALGO`` is forced to "native" to avoid initial benchmarking overhead.
    The TRELLIS weights "JeffreyXiang/TRELLIS-image-large" must be accessible.
"""

import os  # Used to set environment variables affecting CUDA kernels and library behaviour.

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Force deterministic behaviour; "auto" benchmarks per run.

import imageio  # Saves diagnostic videos demonstrating shape-prior renderings.
from PIL import Image  # Loads RGBA crops used as TRELLIS input.
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline  # Generates 3D assets from single images.
from TRELLIS.trellis.utils import render_utils, postprocessing_utils  # Helper utilities for rendering and mesh export.
import numpy as np  # Validates alpha channels and drives array conversions.
from argparse import ArgumentParser  # Exposes CLI configuration for input/output paths.

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)  # RGBA object crop generated during upscaling + segmentation.
parser.add_argument("--output_dir", type=str)  # Destination directory for TRELLIS outputs (videos, meshes, point clouds).
args = parser.parse_args()

img_path = args.img_path  # Input path to the masked high-resolution crop.
output_dir = args.output_dir  # Directory in which to store TRELLIS artefacts.

# Instantiate the pre-trained TRELLIS pipeline. Single GPU usage is assumed for the pipeline stages.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")  # Instantiate the image-to-3D generative pipeline.
pipeline.cuda()  # Move the model to GPU for faster inference.

final_im = Image.open(img_path).convert("RGBA")  # Load the input crop ensuring an RGBA representation.
assert not np.all(
    np.array(final_im)[:, :, 3] == 255
), "Mask must contain transparency so TRELLIS can distinguish foreground."  # Guard against missing alpha masks.

# Run the pipeline
outputs = pipeline.run(
    final_im,
)  # Execute TRELLIS; returns dictionary with gaussian-based and mesh-based reconstructions.

# Render turntables for both gaussian splats and meshes so users can quickly judge reconstruction quality.
video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
video = [
    np.concatenate([frame_gs, frame_mesh], axis=1)
    for frame_gs, frame_mesh in zip(video_gs, video_mesh)
]  # Stitch gaussian + mesh frames side-by-side for debugging.
imageio.mimsave(f"{output_dir}/visualization.mp4", video, fps=30)  # Export the turntable as an MP4 diagnostic video.

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    # Optional parameters
    simplify=0.95,  # Ratio of triangles to remove in the simplification process
    texture_size=1024,  # Size of the texture used for the GLB
)  # Convert TRELLIS outputs into a textured GLB mesh through post-processing.
glb.export(f"{output_dir}/object.glb")  # Persist the GLB for external alignment tools.

# Save Gaussians as PLY files
outputs["gaussian"][0].save_ply(f"{output_dir}/object.ply")  # Dump gaussian splats in PLY format for potential point-cloud use.
```

## data_process/dense_track.py

```python
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
```

## data_process/data_process_pcd.py

```python
"""Fuse multi-view RGB-D frames into world-aligned point clouds with basic depth filtering."""

import numpy as np  # Fundamental numerical library used for matrix operations and point manipulations.
import open3d as o3d  # Handles point-cloud/mesh representations and visualisation utilities.
import json  # Reads metadata describing camera intrinsics and frame counts.
import pickle  # Loads calibration matrices (camera-to-world transforms).
import cv2  # Reads RGB/depth images from disk.
from tqdm import tqdm  # Provides progress bars for long-running frame loops.
import os  # Used for filesystem inspection and directory creation.
from argparse import ArgumentParser  # Parses command-line arguments specifying dataset paths.

# Configure CLI options so the script can be reused standalone or via orchestrators.
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)  # Root directory that stores all capture cases.
parser.add_argument("--case_name", type=str, required=True)  # Specific case folder to convert into point clouds.
args = parser.parse_args()  # Parse arguments immediately for convenience.

base_path = args.base_path  # Dataset root provided by the caller.
case_name = args.case_name  # Name of the case to process.


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
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
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)  # Build a local axis-aligned frame for the camera origin.
        camera.transform(transformation)  # Move the coordinate frame into the camera pose.
    else:
        camera = o3d.geometry.TriangleMesh()  # Fallback dummy mesh when coordinate frame is undesired.
    # Add origin and four corner points in image plane
    points = []  # Collect vertices describing the frustum wireframe.
    camera_origin = np.array([0, 0, 0, 1])  # Homogeneous origin used for transformation.
    points.append(np.dot(transformation, camera_origin)[0:3])  # Transform origin into world coordinates and record.
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5  # Normalising factor derived from intrinsics.
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]  # Flip image plane orientation when needed.
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]  # Default image plane coordinates.
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)  # Normalise and scale the plane point.
        temp_point = np.array(point + [1])  # Promote to homogeneous coordinates for transformation.
        points.append(np.dot(transformation, temp_point)[0:3])  # Transform each frustum corner into world space.
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]  # Indices describing frustum edges.
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )  # Construct a wireframe from the collected points.

    meshes = [camera, line_set]  # Base geometries included in the output list.

    if shoot:
        shoot_points = []  # Extra geometry showing the viewing ray when requested.
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])  # Start point at the camera origin.
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])  # End point along camera forward direction.
        shoot_lines = [[0, 1]]  # Single line segment connecting the two points.
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )  # Build the line geometry.
        shoot_line_set.paint_uniform_color(color)  # Colour the ray for visibility.
        meshes.append(shoot_line_set)  # Include the ray in the returned list.

    return meshes  # Caller receives all generated geometries.


def getPcdFromDepth(depth, intrinsic):
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
    points = np.stack([x, y, np.ones_like(x)], axis=1)  # Assemble homogeneous pixel coordinates.
    points = points * depth[:, None]  # Scale by depth to obtain un-normalised camera rays.
    points = points @ np.linalg.inv(intrinsic).T  # Apply inverse intrinsics to reach metric camera coordinates.
    points = points.reshape(H, W, 3)  # Reshape back to image grid layout.
    return points  # Return XYZ coordinates for every pixel.


def get_pcd_from_data(path, frame_idx, num_cam, intrinsics, c2ws):
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
    for i in range(num_cam):  # Iterate over all cameras contributing to the fused cloud.
        color = cv2.imread(f"{path}/color/{i}/{frame_idx}.png")  # Read the per-camera colour image.
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # Convert OpenCV's BGR ordering back to RGB.
        color = color.astype(np.float32) / 255.0  # Normalise colours to [0, 1] floats for Open3D compatibility.
        depth = np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0  # Load the depth map and convert millimetres to metres.

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsics[i],
        )  # Reconstruct camera-space XYZ coordinates per pixel.
        masks = np.logical_and(points[:, :, 2] > 0.2, points[:, :, 2] < 1.5)  # Keep depths within a plausible range to remove sensor noise.
        points_flat = points.reshape(-1, 3)  # Flatten for homogeneous transform application.
        # Transform points to world coordinates using homogeneous transformation
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )  # Append ones so 4x4 matrices can be applied.
        points_world = np.dot(c2ws[i], homogeneous_points.T).T[:, :3]  # Transform into world space using camera-to-world matrix.
        points_final = points_world.reshape(points.shape)  # Reshape back into image grid layout.
        total_points.append(points_final)  # Store world-space coordinates for this camera.
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
    total_points = np.asarray(total_points)  # Convert lists to numpy arrays for compact storage.
    total_colors = np.asarray(total_colors)  # Convert to array form to simplify downstream indexing.
    total_masks = np.asarray(total_masks)  # Convert mask list to boolean array.
    return total_points, total_colors, total_masks  # Provide the per-camera tensors to the caller.


def exist_dir(dir):
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
        data = json.load(f)  # Load dataset metadata including intrinsics and frame count.
    intrinsics = np.array(data["intrinsics"])  # Convert intrinsics list into numpy array for numeric operations.
    WH = data["WH"]  # Unused image shape info, retained for completeness/debugging.
    frame_num = data["frame_num"]  # Total number of frames captured per camera.
    print(data["serial_numbers"])  # Display camera serials to confirm ordering.

    num_cam = len(intrinsics)  # Determine how many cameras were calibrated.
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))  # Load camera-to-world transforms from disk.

    exist_dir(f"{base_path}/{case_name}/pcd")  # Ensure output directory for point clouds exists.

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

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)  # Global coordinate frame for reference.
    vis.add_geometry(coordinate)  # Visualise the world axes alongside the cameras.

    pcd = None  # Will hold the point cloud geometry reused across frames.
    for i in tqdm(range(frame_num)):  # Iterate over all frames with a progress bar.
        points, colors, masks = get_pcd_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws
        )  # Fetch per-camera world-space points, colours, and masks for the current frame.

        if i == 0:
            pcd = o3d.geometry.PointCloud()  # Create a fresh point cloud geometry for the first frame.
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )  # Flatten camera dimension and filter by mask before assigning XYZ points.
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )  # Apply matching colours to the same valid points.
            vis.add_geometry(pcd)  # Insert the merged cloud into the visualiser.
            # Adjust the viewpoint
            view_control = vis.get_view_control()  # Access camera control interface to orient the scene.
            view_control.set_front([1, 0, -2])  # Configure camera orientation for a clear initial view.
            view_control.set_up([0, 0, -1])  # Set upward direction relative to world axes.
            view_control.set_zoom(1)  # Zoom to a comfortable level for the dataset scale.
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
```

## data_process/data_process_mask.py

```python
"""Post-process SAM2 masks using 3D consistency checks and radius outlier filtering."""

import numpy as np  # Numerical operations for point filtering and mask updates.
import open3d as o3d  # Provides radius-based outlier removal and visualisation utilities.
import json  # Loads semantic ID metadata saved during segmentation.
from tqdm import tqdm  # Progress bar for long-running per-frame loops.
import os  # Handles filesystem operations and directory creation.
import glob  # Enumerates mask and point-cloud files to infer camera/frame counts.
import cv2  # Reads binary mask PNGs for processing.
import pickle  # Serialises the processed mask dictionary for reuse.
from argparse import ArgumentParser  # CLI configuration for locating dataset assets.

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name

processed_masks = {}


def exist_dir(dir):
    """Create ``dir`` if missing so subsequent writes succeed."""

    if not os.path.exists(dir):
        os.makedirs(dir)


def read_mask(mask_path):
    """Load a binary mask and convert it into a boolean numpy array.

    Args:
        mask_path (str): Path to the PNG mask file exported by SAM2.

    Returns:
        np.ndarray: Boolean mask with ``True`` for foreground pixels.
    """

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale intensities.
    mask = mask > 0  # Convert to boolean by treating any non-zero pixel as foreground.
    return mask


def process_pcd_mask(frame_idx, pcd_path, mask_path, mask_info, num_cam):
    """Combine depth-aware filtering and semantic masks to prune noisy detections.

    Args:
        frame_idx (int): Frame number under consideration.
        pcd_path (str): Directory containing per-frame fused point clouds ``{points, colors, masks}``.
        mask_path (str): Directory holding per-camera semantic mask PNGs.
        mask_info (Dict[int, Dict[str, Any]]): Mapping of camera index to object/controller mask IDs.
        num_cam (int): Total number of calibrated cameras contributing to the reconstruction.

    Returns:
        Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]: Filtered object and controller point clouds for visual inspection.
    """
    global processed_masks
    processed_masks[frame_idx] = {}  # Will store refined binary masks for each camera at the current frame.

    # Load the fused RGB-D tensors: one array per camera containing 3D points, colours, and validity masks.
    data = np.load(f"{pcd_path}/{frame_idx}.npz")
    points = data["points"]  # Shape: (num_cam, H, W, 3)
    colors = data["colors"]  # Shape: (num_cam, H, W, 3)
    masks = data["masks"]  # Shape: (num_cam, H, W), indicates valid depth samples.

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()

    for i in range(num_cam):
        # Extract raw object points for camera ``i`` by intersecting semantic masks with valid-depth pixels.
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        object_colors = colors[i][object_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd.colors = o3d.utility.Vector3dVector(object_colors)
        object_pcd += pcd

        # Merge all controller masks, since multiple indices may correspond to the hand/controller.
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        controller_colors = colors[i][controller_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points)
        pcd.colors = o3d.utility.Vector3dVector(controller_colors)
        controller_pcd += pcd

    # Apply radius outlier removal to prune isolated points caused by depth noise.
    cl, ind = object_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_object_points = np.asarray(
        object_pcd.select_by_index(ind, invert=True).points
    )
    object_pcd = object_pcd.select_by_index(ind)

    cl, ind = controller_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_controller_points = np.asarray(
        controller_pcd.select_by_index(ind, invert=True).points
    )
    controller_pcd = controller_pcd.select_by_index(ind)

    # controller_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])
    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()
    for i in range(num_cam):
        processed_masks[frame_idx][i] = {}
        # Reconstruct refined per-pixel masks by zeroing out projections of outlier 3D points.
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        indices = np.nonzero(object_mask)  # Retrieve (row, col) indices of surviving pixels.
        indices_list = list(zip(indices[0], indices[1]))  # Convert to a list so we can map back from flattened indices.
        # Locate all the object_points in the filtered points
        object_indices = []
        for j, point in enumerate(object_points):
            if tuple(point) in filtered_object_points:
                object_indices.append(j)
        original_indices = [indices_list[j] for j in object_indices]
        # Update the object mask
        for idx in original_indices:
            object_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["object"] = object_mask

        # Repeat the same outlier removal process for the controller mask.
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        indices = np.nonzero(controller_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the controller_points in the filtered points
        controller_indices = []
        for j, point in enumerate(controller_points):
            if tuple(point) in filtered_controller_points:
                controller_indices.append(j)
        original_indices = [indices_list[j] for j in controller_indices]
        # Update the controller mask
        for idx in original_indices:
            controller_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["controller"] = controller_mask

        # Build Open3D point clouds representing the refined masks for interactive inspection.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][object_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][object_mask])
        object_pcd += pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][controller_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][controller_mask])
        controller_pcd += pcd

    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])

    return object_pcd, controller_pcd


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"  # Directory containing per-frame fused point clouds.
    mask_path = f"{base_path}/{case_name}/mask"  # Directory of semantic masks exported per camera.

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))  # Determine the number of cameras from metadata files.
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))  # Count fused frames to know iteration length.
    # Load the mask metadata for each camera, collecting the object ID and all controller IDs.
    mask_info = {}
    for i in range(num_cam):
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        mask_info[i] = {}
        for key, value in data.items():
            # Each camera is expected to contain a single non-controller object; the pipeline raises if that assumption breaks.
            if value != CONTROLLER_NAME:
                if "object" in mask_info[i]:
                    # TODO: Handle the case when there are multiple objects
                    import pdb
                    pdb.set_trace()
                mask_info[i]["object"] = int(key)
            if value == CONTROLLER_NAME:
                if "controller" in mask_info[i]:
                    mask_info[i]["controller"].append(int(key))
                else:
                    mask_info[i]["controller"] = [int(key)]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    object_pcd = None
    controller_pcd = None
    for i in tqdm(range(frame_num)):
        temp_object_pcd, temp_controller_pcd = process_pcd_mask(
            i, pcd_path, mask_path, mask_info, num_cam
        )
        if i == 0:
            object_pcd = temp_object_pcd
            controller_pcd = temp_controller_pcd
            vis.add_geometry(object_pcd)
            vis.add_geometry(controller_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
            object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
            controller_pcd.points = o3d.utility.Vector3dVector(
                temp_controller_pcd.points
            )
            controller_pcd.colors = o3d.utility.Vector3dVector(
                temp_controller_pcd.colors
            )
            vis.update_geometry(object_pcd)
            vis.update_geometry(controller_pcd)
            vis.poll_events()
            vis.update_renderer()

    # Save the processed masks considering both depth filter, semantic filter and outlier filter
    with open(f"{base_path}/{case_name}/mask/processed_masks.pkl", "wb") as f:
        pickle.dump(processed_masks, f)

    # Deprecated for now
    # # Generate the videos with for masked objects and controllers
    # exist_dir(f"{base_path}/{case_name}/temp_mask")
    # for i in range(num_cam):
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/object")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/controller")
    #     object_idx = mask_info[i]["object"]
    #     for frame_idx in range(frame_num):
    #         object_mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
    #         img = cv2.imread(f"{base_path}/{case_name}/color/{i}/{frame_idx}.png")
    #         masked_object_img = cv2.bitwise_and(
    #             img, img, mask=object_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/object/{frame_idx}.png",
    #             masked_object_img,
    #         )

    #         controller_mask = np.zeros_like(object_mask)
    #         for controller_idx in mask_info[i]["controller"]:
    #             mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
    #             controller_mask = np.logical_or(controller_mask, mask)
    #         masked_controller_img = cv2.bitwise_and(
    #             img, img, mask=controller_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/controller/{frame_idx}.png",
    #             masked_controller_img,
    #         )

    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/object/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/object_{i}.mp4"
    #     )
    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/controller/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/controller_{i}.mp4"
    #     )
```

## data_process/data_process_track.py

```python
"""Post-process dense tracks to remove outliers and extract stable controller anchor points."""

import numpy as np  # Numerical backbone for array manipulation and vectorised filtering.
import open3d as o3d  # Offers point-cloud processing, KD-tree queries, and visualisation utilities.
from tqdm import tqdm  # Provides progress bars for per-frame iteration loops.
import os  # Manages filesystem operations such as directory creation.
import glob  # Counts files to infer numbers of cameras/frames.
import pickle  # Serialises/deserialises processed tracking data and masks.
import matplotlib.pyplot as plt  # Supplies colour maps for visualisation of motion quality.
from argparse import ArgumentParser  # Parses CLI arguments describing dataset locations.

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)  # Root folder containing the processed dataset.
parser.add_argument("--case_name", type=str, required=True)  # Specific case directory to process.
args = parser.parse_args()

base_path = args.base_path  # Dataset root path provided by user.
case_name = args.case_name  # Case identifier under the dataset root.


def exist_dir(dir):
    """Create ``dir`` if missing.

    Args:
        dir (str): Filesystem path that should exist prior to writing artefacts.

    Returns:
        None
    """
    if not os.path.exists(dir):  # Avoid redundant mkdir calls when directory already exists.
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
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)  # Build and position the primitive sphere.
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
        processed_masks = pickle.load(f)  # Refined per-frame masks computed during mask post-processing.

    # Filter out the points not valid in the first frame
    object_points = []  # Will store object trajectories (XYZ) concatenated across cameras.
    object_colors = []  # Corresponding RGB colours for object trajectories.
    object_visibilities = []  # Binary visibility flags indicating when each trajectory is present.
    controller_points = []  # Controller trajectories (XYZ).
    controller_colors = []  # Controller colours for visualisation.
    controller_visibilities = []  # Visibility mask for controller trajectories.
    for i in range(num_cam):  # Process each camera's tracking data independently before merging.
        current_track_data = np.load(f"{track_path}/{i}.npz")  # Load tracked pixel coordinates and visibility flags.
        # Filter out the track data
        tracks = current_track_data["tracks"]  # Shape: (frame_num, num_points, 2) storing pixel coordinates.
        tracks = np.round(tracks).astype(int)  # Round to nearest pixel indices so they can index mask arrays.
        visibility = current_track_data["visibility"]  # Binary matrix (frame_num, num_points) indicating tracker confidence.
        assert tracks.shape[0] == frame_num  # Sanity-check that track duration matches frame count.
        num_points = np.shape(tracks)[1]  # Total number of tracked points for this camera.

        # Locate the track points in the object mask of the first frame
        object_mask = processed_masks[0][i]["object"]  # Binary mask describing object pixels in frame 0.
        track_object_idx = np.zeros((num_points), dtype=int)  # Placeholder storing whether each track belongs to the object.
        for j in range(num_points):  # Evaluate every trajectory.
            if visibility[0, j] == 1:  # Only consider tracks visible in the first frame for classification.
                track_object_idx[j] = object_mask[tracks[0, j, 0], tracks[0, j, 1]]  # Mark if starting pixel lies inside object mask.
        # Locate the controller points in the controller mask of the first frame
        controller_mask = processed_masks[0][i]["controller"]  # Binary mask highlighting controller pixels in frame 0.
        track_controller_idx = np.zeros((num_points), dtype=int)  # Flag array tracking controller membership per trajectory.
        for j in range(num_points):
            if visibility[0, j] == 1:  # Only classify points visible in reference frame.
                track_controller_idx[j] = controller_mask[
                    tracks[0, j, 0], tracks[0, j, 1]
                ]  # Set flag if pixel begins within the controller mask.

        # Filter out bad tracking in other frames
        for frame_idx in range(1, frame_num):  # Inspect every subsequent frame to drop inconsistent tracks.
            # Filter based on object_mask
            object_mask = processed_masks[frame_idx][i]["object"]  # Object mask at current frame.
            for j in range(num_points):
                try:
                    if track_object_idx[j] == 1 and visibility[frame_idx, j] == 1:
                        if not object_mask[
                            tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                        ]:
                            visibility[frame_idx, j] = 0  # Invalidate track when projected pixel leaves object mask.
                except:
                    # Sometimes the track coordinate is out of image
                    visibility[frame_idx, j] = 0  # Drop coordinates that fall outside valid image bounds.
            # Filter based on controller_mask
            controller_mask = processed_masks[frame_idx][i]["controller"]  # Controller mask at current frame.
            for j in range(num_points):
                if track_controller_idx[j] == 1 and visibility[frame_idx, j] == 1:
                    if not controller_mask[
                        tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                    ]:
                        visibility[frame_idx, j] = 0  # Remove controller track when it drifts outside controller segmentation.

        # Get the track point cloud
        track_points = np.zeros((frame_num, num_points, 3))  # Placeholder for per-frame 3D points.
        track_colors = np.zeros((frame_num, num_points, 3))  # Placeholder for per-frame RGB colours.
        for frame_idx in range(frame_num):  # For each frame, gather corresponding 3D sample from fused PCD.
            data = np.load(f"{pcd_path}/{frame_idx}.npz")  # Load fused point cloud arrays for current frame.
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

        object_points.append(track_points[:, np.where(track_object_idx)[0], :])  # Collect only the trajectories identified as object points.
        object_colors.append(track_colors[:, np.where(track_object_idx)[0], :])  # Store their colours for visualisation.
        object_visibilities.append(visibility[:, np.where(track_object_idx)[0]])  # Retain visibility flags for the same subset.
        controller_points.append(track_points[:, np.where(track_controller_idx)[0], :])  # Extract controller-associated tracks.
        controller_colors.append(track_colors[:, np.where(track_controller_idx)[0], :])  # Save controller colours.
        controller_visibilities.append(visibility[:, np.where(track_controller_idx)[0]])  # Save controller visibility masks.

    object_points = np.concatenate(object_points, axis=1)  # Merge object tracks from all cameras along point dimension.
    object_colors = np.concatenate(object_colors, axis=1)  # Combine object colour arrays accordingly.
    object_visibilities = np.concatenate(object_visibilities, axis=1)  # Merge object visibility matrices.
    controller_points = np.concatenate(controller_points, axis=1)  # Merge controller tracks across cameras.
    controller_colors = np.concatenate(controller_colors, axis=1)  # Merge controller colours across cameras.
    controller_visibilities = np.concatenate(controller_visibilities, axis=1)  # Merge controller visibility flags.

    track_data = {}  # Collect filtered track payload in one dictionary.
    track_data["object_points"] = object_points  # World-space object trajectories.
    track_data["object_colors"] = object_colors  # Associated RGB colours for object tracks.
    track_data["object_visibilities"] = object_visibilities  # Frame-by-frame visibility of object tracks.
    track_data["controller_points"] = controller_points  # World-space controller trajectories.
    track_data["controller_colors"] = controller_colors  # Controller colour samples.
    track_data["controller_visibilities"] = controller_visibilities  # Controller visibility flags.

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
    object_points = track_data["object_points"]  # (num_frames, num_points, 3) object trajectory positions.
    object_colors = track_data["object_colors"]  # RGB colours for object tracks.
    object_visibilities = track_data["object_visibilities"]  # Visibility flags for each object track per frame.
    object_motions = np.zeros_like(object_points)  # Placeholder for per-frame motion vectors.
    object_motions[:-1] = object_points[1:] - object_points[:-1]  # Finite difference to approximate motion between consecutive frames.
    object_motions_valid = np.zeros_like(object_visibilities)  # Flags to indicate when both consecutive frames are valid.
    object_motions_valid[:-1] = np.logical_and(
        object_visibilities[:-1], object_visibilities[1:]
    )  # Mark motion as valid only when the point is visible in both frames being compared.

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])  # Determine y-range for colouring.
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)  # Normalise heights to [0, 1].
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]  # Assign rainbow colours based on vertical position.

    num_frames = object_points.shape[0]  # Total number of frames.
    num_points = object_points.shape[1]  # Total number of object tracks.

    vis = o3d.visualization.Visualizer()  # Create Open3D window for interactive pruning supervision.
    vis.create_window()  # Display the window.
    for i in tqdm(range(num_frames - 1)):  # Inspect each motion segment across frames (frame i -> i+1).
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points[i])  # Set positions for frame i.
        pcd.colors = o3d.utility.Vector3dVector(object_colors[i])  # Use RGB colours captured from fused PCD.
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)  # Precompute nearest neighbours for the current frame.
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
            neighbors = [index for index in idx if object_motions_valid[i, index] == 1]  # Keep neighbours with valid motion for comparison.
            if len(neighbors) < 5:
                object_motions_valid[i, j] = 0  # Reject trajectories with insufficient local support.
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])
            motion_diff = np.linalg.norm(
                object_motions[i, j] - object_motions[i, neighbors], axis=1
            )  # Compute deviation between point motion and its neighbours.
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                object_motions_valid[i, j] = 0  # Invalidate motion when it disagrees with the majority of nearby tracks.
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
            render_motion_pcd = motion_pcd  # Cache geometry pointer to update in-place across frames.
            # render_modified_pcd = modified_pcd
            # render_new_pcd = new_pcd
            vis.add_geometry(render_motion_pcd)  # Add filtered object motion cloud to the viewer.
            # vis.add_geometry(render_modified_pcd)
            # vis.add_geometry(render_new_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()  # Configure default view for easier inspection.
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)  # Update geometry with next frame's surviving points.
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)  # Update colours to match new frame.
            # render_modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
            # render_modified_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
            # )
            # render_new_pcd.points = o3d.utility.Vector3dVector(new_points)
            # render_new_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
            # )
            vis.update_geometry(render_motion_pcd)  # Trigger viewer refresh with latest selection.
            # vis.update_geometry(render_modified_pcd)
            # vis.update_geometry(render_new_pcd)
            vis.poll_events()  # Keep UI responsive while iterating.
            vis.update_renderer()  # Redraw with updated data.
        # modified_num = len(modified_points)
        # print(f"Object Frame {i}: {modified_num} points are modified")

    vis.destroy_window()  # Close the window once object motion filtering is complete.
    track_data["object_motions_valid"] = object_motions_valid  # Persist validity mask for later use/visualisation.

    controller_points = track_data["controller_points"]  # Controller trajectories.
    controller_colors = track_data["controller_colors"]  # Controller colour samples.
    controller_visibilities = track_data["controller_visibilities"]  # Controller visibility flags.
    controller_motions = np.zeros_like(controller_points)  # Placeholder for controller motion vectors.
    controller_motions[:-1] = controller_points[1:] - controller_points[:-1]  # Compute motion between consecutive frames.
    controller_motions_valid = np.zeros_like(controller_visibilities)  # Flags to track when controller motion is reliable.
    controller_motions_valid[:-1] = np.logical_and(
        controller_visibilities[:-1], controller_visibilities[1:]
    )  # Motion valid only if point visible in adjacent frames.
    num_points = controller_points.shape[1]  # Number of controller trajectories.
    # Filter all points that disappear in the sequence
    mask = np.prod(controller_visibilities, axis=0)  # Identify controller points visible in every frame (product = 1 when always visible).

    y_min, y_max = np.min(controller_points[0, :, 1]), np.max(
        controller_points[0, :, 1]
    )  # Determine y-range for controller colouring.
    y_normalized = (controller_points[0, :, 1] - y_min) / (y_max - y_min)  # Normalise heights.
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]  # Precompute rainbow colours for persistent controller tracks.

    vis = o3d.visualization.Visualizer()  # New viewer for controller filtering.
    vis.create_window()

    for i in tqdm(range(num_frames - 1)):  # Iterate over controller motion segments.
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points[i])  # Set controller point positions.
        pcd.colors = o3d.utility.Vector3dVector(controller_colors[i])  # Use their RGB colours.
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)  # Prepare neighbour queries.
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):  # Evaluate each controller track.
            if mask[j] == 0:
                controller_motions_valid[i, j] = 0  # Immediately drop tracks that are not visible in all frames.
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
                controller_motions_valid[i, j] = 0  # Drop track when motion deviates from neighbours.
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

    track_data["controller_mask"] = mask  # Store binary indicator of controller tracks that survived filtering.
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
    object_points = track_data["object_points"]  # Object trajectories for potential future use/visualisation.
    object_colors = track_data["object_colors"]  # Colours for the object trajectories.
    object_visibilities = track_data["object_visibilities"]  # Visibility flags for object points.
    object_motions_valid = track_data["object_motions_valid"]  # Motion validity mask computed earlier.
    controller_points = track_data["controller_points"]  # Controller trajectories retained from motion filtering.
    mask = track_data["controller_mask"]  # Boolean mask selecting controller tracks valid across sequence.

    new_controller_points = controller_points[:, np.where(mask)[0], :]  # Keep only globally valid controller tracks.
    assert len(new_controller_points[0]) >= 30  # Sanity-check that enough points survived for farthest-point sampling.
    # Do farthest point sampling on the valid controller points to select the final controller points
    valid_indices = np.arange(len(new_controller_points[0]))  # Candidate indices among surviving tracks.
    points_map = {}  # Map from 3D coordinate tuples to index for quick lookup.
    sample_points = []  # List of points used for FPS input geometry.
    for i in valid_indices:
        points_map[tuple(new_controller_points[0, i])] = i  # Remember which index owns each point.
        sample_points.append(new_controller_points[0, i])  # Collect points for FPS.
    sample_points = np.array(sample_points)  # Convert to numpy array for Open3D operations.
    sample_pcd = o3d.geometry.PointCloud()  # Build Open3D point cloud from candidate points.
    sample_pcd.points = o3d.utility.Vector3dVector(sample_points)
    fps_pcd = sample_pcd.farthest_point_down_sample(30)  # Select 30 representative controller points using FPS.
    final_indices = []  # Indices of points selected by FPS in the original array ordering.
    for point in fps_pcd.points:
        final_indices.append(points_map[tuple(point)])  # Map each sampled point back to its index.

    print(f"Controller Point Number: {len(final_indices)}")  # Report how many controller anchors remain.

    # Get the nearest controller points and their colors
    nearest_controller_points = new_controller_points[:, final_indices]  # Keep trajectories for the selected controller anchors.

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

    track_data.pop("controller_points")  # Remove original (larger) controller set to avoid confusion.
    track_data.pop("controller_colors")  # Remove colours aligned with removed points.
    track_data.pop("controller_visibilities")  # Remove visibility info for removed points.
    track_data["controller_points"] = nearest_controller_points  # Replace with compact controller subset.

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
    object_visibilities = track_data["object_visibilities"]  # Visibility flags for object trajectories.
    object_motions_valid = track_data["object_motions_valid"]  # Mask of object motions surviving filtering.
    controller_points = track_data["controller_points"]  # Final controller anchor trajectories.

    frame_num = object_points.shape[0]  # Number of frames to visualise.

    vis = o3d.visualization.Visualizer()
    vis.create_window()  # Launch viewer window.
    controller_meshes = []  # Will hold sphere geometry per controller anchor.
    prev_center = []  # Track previous positions to update spheres efficiently.

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])  # Determine y-range for consistent colouring.
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)  # Normalise to [0, 1].
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]  # Colour look-up for object trajectories.

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
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)  # Update point positions.
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)  # Update colours.
            vis.update_geometry(render_object_pcd)  # Notify Open3D of updates.
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]  # New controller position in current frame.
                controller_meshes[j].translate(origin - prev_center[j])  # Move sphere to new location relative to previous frame.
                vis.update_geometry(controller_meshes[j])  # Refresh geometry in viewer.
                prev_center[j] = origin  # Cache position for next iteration.
            vis.poll_events()  # Process UI events while animating.
            vis.update_renderer()  # Redraw scene for current frame.


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"  # Directory containing fused point clouds per frame.
    mask_path = f"{base_path}/{case_name}/mask"  # Directory containing processed masks.
    track_path = f"{base_path}/{case_name}/cotracker"  # Directory with Co-Tracker raw outputs.

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))  # Infer number of cameras from mask metadata files.
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))  # Count number of fused point-cloud frames.

    # Filter the track data using the semantic mask of object and controller
    track_data = filter_track(track_path, pcd_path, mask_path, frame_num, num_cam)  # Remove inconsistent trajectories using segmentation masks.
    # Filter motion
    track_data = filter_motion(track_data)  # Further prune tracks with aberrant motion patterns.
    # # Save the filtered track data
    # with open(f"test2.pkl", "wb") as f:
    #     pickle.dump(track_data, f)

    # with open(f"test2.pkl", "rb") as f:
    #     track_data = pickle.load(f)

    track_data = get_final_track_data(track_data)  # Reduce controller tracks to a representative subset via FPS.

    with open(f"{base_path}/{case_name}/track_process_data.pkl", "wb") as f:
        pickle.dump(track_data, f)  # Persist filtered trajectories for downstream optimisation.

    visualize_track(track_data)  # Launch interactive playback so users can confirm track quality.
```

## data_process/align.py

```python
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

VIS = True  # Global flag controlling whether intermediate visualisations are generated.
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


def existDir(dir_path):
    """Create ``dir_path`` if it does not exist so downstream file writes succeed.

    Args:
        dir_path (str): Absolute or relative directory path used to store alignment artefacts.

    Returns:
        None
    """
    if not os.path.exists(dir_path):  # Avoid re-creating directories unnecessarily.
        os.makedirs(dir_path)  # Recursively create the requested directory.


def pose_selection_render_superglue(
    raw_img, fov, mesh_path, mesh, crop_img, output_dir
):
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


def registration_pnp(mesh_matching_points, raw_matching_points, intrinsic):
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


def registration_scale(mesh_matching_points_cam, matching_points_cam):
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


def deform_ARAP(initial_mesh_world, mesh_matching_points_world, matching_points):
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
    mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
):
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
    deform_kp_mesh_world,
    obs_points_world,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    mesh_points_indices,
    matching_points,
):
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


def line_point_distance(p, points):
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
```

## data_process/data_process_sample.py

```python
"""Sample training data from filtered trajectories and optional shape priors."""

import numpy as np  # Numerical backbone for point manipulations.
import open3d as o3d  # Provides point-cloud operations and rendering utilities.
import pickle  # Serialises processed data back to disk.
import matplotlib.pyplot as plt  # Supplies colour maps for visualising trajectories.
import trimesh  # Loads and samples meshes produced by TRELLIS.
import cv2  # Writes diagnostic videos showcasing the final data.
from utils.align_util import as_mesh  # Converts heterogeneous trimesh objects into a canonical mesh.
from argparse import ArgumentParser  # Parses CLI arguments selecting dataset and options.

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument("--num_surface_points", type=int, default=1024)
parser.add_argument("--volume_sample_size", type=float, default=0.005)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Used to judge if using the shape prior
SHAPE_PRIOR = args.shape_prior
num_surface_points = args.num_surface_points
volume_sample_size = args.volume_sample_size


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    """Create an Open3D sphere used to mark controller anchor points in visualisations.

    Args:
        center (Iterable[float]): XYZ coordinates of the sphere centre.
        radius (float, optional): Radius of the sphere marker. Defaults to 0.1.
        color (Iterable[float], optional): RGB colour for the marker surface. Defaults to black.

    Returns:
        o3d.geometry.TriangleMesh: Sphere mesh translated to ``center``.
    """

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def process_unique_points(track_data):
    """Remove duplicate object tracks, optionally blend in a shape prior, and export diagnostics.

    Args:
        track_data (Dict[str, np.ndarray]): Filtered trajectories generated by ``data_process_track``.

    Returns:
        Dict[str, np.ndarray]: Updated dictionary with deduplicated object points and optional prior samples.
    """

    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    # Compute a unique index set so duplicated pixels (common across camera overlap) are removed.
    first_object_points = object_points[0]
    unique_idx = np.unique(first_object_points, axis=0, return_index=True)[1]
    object_points = object_points[:, unique_idx, :]
    object_colors = object_colors[:, unique_idx, :]
    object_visibilities = object_visibilities[:, unique_idx]
    object_motions_valid = object_motions_valid[:, unique_idx]

    # Clamp any point predicted below the table plane back onto z = 0 to avoid thin tails in the sample grid.
    object_points[object_points[..., 2] > 0, 2] = 0

    if SHAPE_PRIOR:
        shape_mesh_path = f"{base_path}/{case_name}/shape/matching/final_mesh.glb"
        trimesh_mesh = trimesh.load(shape_mesh_path, force="mesh")
        trimesh_mesh = as_mesh(trimesh_mesh)
        # Sample both surface and interior points from the aligned mesh to densify the object volume.
        surface_points, _ = trimesh.sample.sample_surface(
            trimesh_mesh, num_surface_points
        )
        interior_points = trimesh.sample.volume_mesh(trimesh_mesh, 10000)

    # Build a voxel grid keyed by ``volume_sample_size`` to prioritise object points and avoid redundant samples.
    if SHAPE_PRIOR:
        all_points = np.concatenate(
            [surface_points, interior_points, object_points[0]], axis=0
        )
    else:
        all_points = object_points[0]
    min_bound = np.min(all_points, axis=0)
    index = []  # Indices of object points that occupy new voxels.
    grid_flag = {}  # Tracks which voxels have already been filled.
    for i in range(object_points.shape[1]):
        grid_index = tuple(
            np.floor((object_points[0, i] - min_bound) / volume_sample_size).astype(int)
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            index.append(i)

    if SHAPE_PRIOR:
        final_surface_points = []
        for i in range(surface_points.shape[0]):
            grid_index = tuple(
                np.floor((surface_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_surface_points.append(surface_points[i])
        final_interior_points = []
        for i in range(interior_points.shape[0]):
            grid_index = tuple(
                np.floor((interior_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_interior_points.append(interior_points[i])
        all_points = np.concatenate(
            [final_surface_points, final_interior_points, object_points[0][index]],
            axis=0,
        )
    else:
        all_points = object_points[0][index]

    # Render a turntable video of the densified point cloud so users can validate sampling quality.
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(all_points)
    coorindate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_pcd.mp4", fourcc, 30, (width, height)
    )

    vis.add_geometry(all_pcd)
    # vis.add_geometry(coorindate)
    view_control = vis.get_view_control()
    for j in range(360):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    vis.destroy_window()

    # Replace the object entries with the deduplicated subset retained by ``index``.
    track_data.pop("object_points")
    track_data.pop("object_colors")
    track_data.pop("object_visibilities")
    track_data.pop("object_motions_valid")
    track_data["object_points"] = object_points[:, index, :]
    track_data["object_colors"] = object_colors[:, index, :]
    track_data["object_visibilities"] = object_visibilities[:, index]
    track_data["object_motions_valid"] = object_motions_valid[:, index]
    if SHAPE_PRIOR:
        track_data["surface_points"] = np.array(final_surface_points)
        track_data["interior_points"] = np.array(final_interior_points)
    else:
        track_data["surface_points"] = np.zeros((0, 3))
        track_data["interior_points"] = np.zeros((0, 3))

    return track_data


def visualize_track(track_data):
    """Render an animated preview of the filtered object and controller trajectories.

    Args:
        track_data (Dict[str, np.ndarray]): Dictionary produced by :func:`process_unique_points`.

    Returns:
        None
    """

    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_data.mp4", fourcc, 30, (width, height)
    )

    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_visibilities[i])[0], :]
        )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_visibilities[i])[0]]
        )  # Colour encode points by vertical position for easier reading.

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere meshes to visualise each controller anchor trajectory.
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint to a canonical angle for all turntable renders.
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
        track_data = pickle.load(f)  # Load motion-filtered trajectories produced by the tracking pipeline.

    track_data = process_unique_points(track_data)  # Deduplicate, voxel-sample, and integrate optional shape prior points.

    with open(f"{base_path}/{case_name}/final_data.pkl", "wb") as f:
        pickle.dump(track_data, f)  # Persist the final dataset consumed by training/inference scripts.

    visualize_track(track_data)  # Export a diagnostic video that visualises both object and controller trajectories.
```

