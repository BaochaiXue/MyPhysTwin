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
