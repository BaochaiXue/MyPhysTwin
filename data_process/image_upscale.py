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
