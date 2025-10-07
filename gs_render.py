#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Stage 4—Gaussian Splatting Rendering Glue
# Role: Batch-render Gaussian checkpoints, run culling heuristics, and deliver assets
#       to the interactive playground and evaluation scripts.
# Inputs: Trained Gaussians (`model_path`/iteration folders), camera rigs (Scene),
#         pipeline config, optional filtering flags.
# Outputs: Saved PNG renders per view, filtered GaussianModel instances for runtime.
# Key in-house deps: `gaussian_splatting.scene.Scene`, `GaussianModel`, `gs_render`
#                    consumers in trainer module.
# Side effects: Creates render directories under checkpoints, mutates Gaussian models
#               in memory when filtering, writes imagery to disk.
# Assumptions: CUDA available for rendering kernels; dataset folders follow Gaussian
#              Splatting layout; required metadata (cameras, masks) already stored.

import torch
from gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_splatting.gaussian_renderer import render
import torchvision
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import numpy as np
from kornia import create_meshgrid
import copy
import pytorch3d
import pytorch3d.ops as ops


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, disable_sh=False):
    """Render a list of camera views and persist RGB/GT frames for inspection.

    Parameters
    ----------
    model_path : str
        Root directory of the Gaussian checkpoint.
    name : str
        Subfolder identifier (e.g., 'train' or 'test').
    iteration : int
        Checkpoint iteration number used to differentiate render batches.
    views : Sequence[Camera]
        Camera descriptors to render from.
    gaussians : GaussianModel
        Gaussian radiance field to rasterise.
    pipeline : PipelineParams
        Rendering configuration (shading, background blending).
    background : torch.Tensor
        RGB background colour, typically `[1, 1, 1]` or `[0, 0, 0]`.
    train_test_exp : bool
        Flag controlling whether to render half-resolution crops for stereo training.
    separate_sh : bool
        Toggle for using per-Gaussian spherical harmonics coefficients.
    disable_sh : bool, optional
        When True, bypass trained SH and fall back to DC components only.

    Side Effects
    ------------
    * Creates render/GT directories and writes PNG files for each view.
    """
    # --- Prepare output directories ------------------------------------------
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # TODO: temporary debug for demo
    # scene_name = model_path.split('/')[-2]
    # render_path = os.path.join('./output_tmp_for_sydney', scene_name, "renders")
    # gts_path = os.path.join('./output_tmp_for_sydney', scene_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # --- Iterate over cameras and rasterise -----------------------------------
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if disable_sh:
            override_color = gaussians.get_features_dc.squeeze()
            results = render(view, gaussians, pipeline, background, override_color=override_color, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        else:
            results = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}.png".format(idx)))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}.png".format(idx)))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, separate_sh: bool, remove_gaussians: bool = False):
    """Load Gaussian checkpoints and render train/test splits, applying optional filters.

    Parameters
    ----------
    dataset : ModelParams
        Parsed model arguments describing dataset layout and background settings.
    iteration : int
        Checkpoint iteration to render; `-1` selects the latest.
    pipeline : PipelineParams
        Rendering configuration (shading, upsampling, etc.).
    skip_train, skip_test : bool
        Flags to avoid rendering respective splits.
    separate_sh : bool
        Whether to treat spherical harmonics coefficients per Gaussian.
    remove_gaussians : bool, optional
        Apply visibility-based culling before rendering.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # remove gaussians that are outside the mask
        if remove_gaussians:
            gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())

        # remove gaussians that are low opacity
        gaussians = remove_gaussians_with_low_opacity(gaussians)

        # TODO: quick demo purpose (remove later)
        # # sub-sample the gaussians
        # n_subsample = 1000
        # idx = torch.randperm(gaussians._xyz.size(0))[:n_subsample]
        # gaussians._xyz = gaussians._xyz[idx]
        # gaussians._features_dc = gaussians._features_dc[idx]
        # gaussians._features_rest = gaussians._features_rest[idx]
        # gaussians._scaling = gaussians._scaling[idx]
        # gaussians._rotation = gaussians._rotation[idx]
        # gaussians._opacity = gaussians._opacity[idx]
        # # set the scale of the gaussians
        # scale = 0.01
        # gaussians._scaling = gaussians.scaling_inverse_activation(torch.ones_like(gaussians._scaling) * scale)

        # remove gaussians that are far from the mesh
        # gaussians = remove_gaussians_with_point_mesh_distance(gaussians, scene.mesh_sampled_points, dist_threshold=0.01)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, disable_sh=dataset.disable_sh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, disable_sh=dataset.disable_sh)


def get_ray_directions(H, W, K, device='cuda', random=False, return_uv=False, flatten=True, anti_aliasing_factor=1.0):
    """Generate camera-space ray directions for every pixel given intrinsics.

    Parameters
    ----------
    H, W : int
        Image height and width in pixels.
    K : torch.Tensor or np.ndarray
        3x3 camera intrinsics matrix (focal lengths and principal point).
    device : str
        Desired device for the generated tensors (default `'cuda'`).
    random : bool
        If True, stratify rays by sampling random offsets within each pixel footprint.
    return_uv : bool
        If True, also return pixel coordinates alongside ray directions.
    flatten : bool
        When True, collapse `(H, W, 3)` into `(H*W, 3)` for downstream rasterization.
    anti_aliasing_factor : float
        Isotropic scaling to supersample the grid when anti-aliasing (AA) is enabled.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Ray directions of shape `(H*W, 3)` (or `(H, W, 3)` when `flatten=False`). If
        `return_uv` is True, also returns pixel coordinates of shape `(H*W, 2)`.
    """
    if anti_aliasing_factor > 1.0:
        H = int(H * anti_aliasing_factor) 
        W = int(W * anti_aliasing_factor) 
        K *= anti_aliasing_factor
        K[2, 2] = 1
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)
    if return_uv:
        return directions, grid
    return directions


def remove_gaussians_with_mask(gaussians, views):
    """Cull Gaussians that rarely appear within alpha masks across the training views.

    Parameters
    ----------
    gaussians : GaussianModel
        Source Gaussian model to filter (not modified in place).
    views : Sequence[Camera]
        Training cameras providing alpha masks for visibility statistics.

    Returns
    -------
    GaussianModel
        Deep copy containing only Gaussians that pass the visibility threshold.
    """
    gaussians_xyz = gaussians._xyz.detach()
    gaussians_view_counter = torch.zeros(gaussians_xyz.shape[0], dtype=torch.int32, device='cuda')
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            H, W = view.image_height, view.image_width
            K = view.K
            R, T = view.R, view.T

            # Create the World-to-Camera transformation matrix
            W2C = np.zeros((4, 4))
            W2C[:3, :3] = R.transpose()
            W2C[:3, 3] = T
            W2C[3, 3] = 1.0
            W2C = torch.tensor(W2C, dtype=torch.float32, device='cuda')

            # --- Transform Gaussians into camera space ------------------------
            xyz = torch.cat([gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1, device='cuda')], dim=1)
            xyz = torch.matmul(xyz, W2C.T)
            xyz = xyz[:, :3]
            xyz = xyz / xyz[:, 2].unsqueeze(1)  # Normalize by z-coordinate

            # --- Project to image plane and accumulate mask coverage ----------
            uv = torch.matmul(xyz, torch.FloatTensor(K).to("cuda").T)
            uv = uv[:, :2].round().long()   # Convert to integer pixel coordinates

            # Check if (u, v) coordinates are within the image bounds
            alpha_mask = view.alpha_mask.squeeze(0)    # Assuming mask is a 2D tensor on CUDA with shape [H, W]
            valid_uv = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

            # Filter valid coordinates and check mask values
            for i, (u, v) in enumerate(uv):
                if valid_uv[i] and alpha_mask[v, u] > 0:  # Mask value > 0 implies it lies within the mask region
                    gaussians_view_counter[i] += 1
        
        # Remove the gaussians that are visible in a frequency of less than 50% of the views
        VIEW_THRESHOLD = 1.0
        mask3d = gaussians_view_counter >= len(views) * VIEW_THRESHOLD
        print(f"Removing {len(mask3d) - mask3d.sum()} gaussians not visible in {VIEW_THRESHOLD * 100}% of the views")
        new_gaussians = copy.deepcopy(gaussians)
        new_gaussians._xyz = gaussians._xyz[mask3d]
        new_gaussians._features_dc = gaussians._features_dc[mask3d]
        new_gaussians._features_rest = gaussians._features_rest[mask3d]
        new_gaussians._scaling = gaussians._scaling[mask3d]
        new_gaussians._rotation = gaussians._rotation[mask3d]
        new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


def remove_gaussians_with_low_opacity(gaussians, opacity_threshold=0.1):
    """Discard Gaussians whose decoded opacity falls below a transparency threshold.

    Parameters
    ----------
    gaussians : GaussianModel
        Source Gaussian model to filter (deep-copied internally).
    opacity_threshold : float, optional
        Minimum opacity for Gaussians to survive the filter.

    Returns
    -------
    GaussianModel
        Filtered Gaussian model containing only high-opacity splats.
    """

    opacity = gaussians.get_opacity.squeeze(-1)
    mask3d = opacity > opacity_threshold
    print(f"Removing {len(mask3d) - mask3d.sum()} gaussians with opacity < 0.1")

    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = gaussians._xyz[mask3d]
    new_gaussians._features_dc = gaussians._features_dc[mask3d]
    new_gaussians._features_rest = gaussians._features_rest[mask3d]
    new_gaussians._scaling = gaussians._scaling[mask3d]
    new_gaussians._rotation = gaussians._rotation[mask3d]
    new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


def remove_gaussians_with_point_mesh_distance(gaussians, mesh_sampled_points, dist_threshold=0.1):
    """Filter Gaussians by proximity to a reference mesh, using PyTorch3D ball queries.

    Parameters
    ----------
    gaussians : GaussianModel
        Gaussian model to filter.
    mesh_sampled_points : torch.Tensor
        `(M, 3)` points sampled from the reference mesh surface.
    dist_threshold : float, optional
        Maximum allowed distance between a Gaussian centre and the mesh sample.

    Returns
    -------
    GaussianModel
        Filtered Gaussian model with distant splats removed.
    """

    gaussians_xyz = gaussians._xyz.detach()
    # dists_knn = ops.knn_points(gaussians_xyz.unsqueeze(0), mesh_sampled_points.unsqueeze(0), K=1, norm=2)
    dists_bq = ops.ball_query(gaussians_xyz.unsqueeze(0), mesh_sampled_points.unsqueeze(0), K=1, radius=dist_threshold)
    mask3d = (dists_bq[1].squeeze(0) != -1).squeeze(-1)
    print(f"Removing {len(mask3d) - mask3d.sum()} gaussians with distance < {dist_threshold}")

    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = gaussians._xyz[mask3d]
    new_gaussians._features_dc = gaussians._features_dc[mask3d]
    new_gaussians._features_rest = gaussians._features_rest[mask3d]
    new_gaussians._scaling = gaussians._scaling[mask3d]
    new_gaussians._rotation = gaussians._rotation[mask3d]
    new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.remove_gaussians)
