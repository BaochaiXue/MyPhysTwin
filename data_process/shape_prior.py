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
