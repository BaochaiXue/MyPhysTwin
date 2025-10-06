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
