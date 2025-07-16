# Data Processing Modules

This folder contains utilities for pre-processing RGB-D sequences and generating meshes or point clouds used throughout the project.

## Module Overview

- `data_process_pcd.py` – Generates colored point clouds from depth images and camera poses.
- `data_process_mask.py` – Filters segmentation masks and removes outliers in point clouds.
- `match_pairs.py` – Performs image feature matching using SuperPoint and SuperGlue.
- `shape_prior.py` – Example script demonstrating TRELLIS based mesh generation.
- `utils/` – Helper functions for rendering, alignment and visualisation.

Other scripts are used for tracking and segmentation and follow similar conventions.

## Usage

Most scripts expect a prepared dataset directory containing `color/`, `depth/` and `mask/` folders as well as metadata such as intrinsics and calibration files. Example command for point cloud generation:

```bash
python data_process/data_process_pcd.py --base_path PATH_TO_DATASET --case_name CASE
```

Feature matching on a set of images can be invoked via:

```bash
python data_process/match_pairs.py --help
```

### Dependencies

The code relies on `open3d`, `numpy`, `opencv-python`, `torch` and `tqdm` among others. Please install the requirements of the repository before running the scripts.

### Example

```
python data_process/shape_prior.py --img_path input.png --output_dir output
```

This command produces a mesh and gaussian representation of the input image using the TRELLIS pipeline.

## Notes

- Paths in arguments are assumed to be relative to the project root.
- Some scripts may require CUDA capable hardware.
