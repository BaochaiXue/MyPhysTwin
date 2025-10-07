# Stage 2—Real Dataset Adapter
# Role: Deserialize capture pickles and map their arrays into GPU tensors aligned with
#       the Warp simulator's expectations; optionally export diagnostic videos.
# Inputs: `cfg.data_path` (`final_data.pkl`), device spec `cfg.device`, experiment
#         directory `cfg.base_dir`, config flags for visualisation.
# Outputs: Attributes on `RealData` (point clouds, visibilities, colours, controller
#          trajectories) plus optional `gt.mp4` rendered via `visualize_pc`.
# Key in-house deps: `qqtt.utils.cfg` (device/base_dir), `qqtt.utils.logger`,
#                    `qqtt.utils.visualize_pc` for visual diagnostics.
# Side effects: Reads pickle into memory, writes `cfg.base_dir/gt.mp4` when `save_gt`
#               is True, logs progress messages.
# Assumptions: Pickle keys include `object_points`, `controller_points`, etc.; CUDA
#              device available and `cfg.base_dir` already set by the trainer.

import numpy as np
import torch
import pickle
from qqtt.utils import logger, visualize_pc, cfg
import matplotlib.pyplot as plt


class RealData:
    """Wrap real capture data and expose tensors aligned with simulator expectations.

    The loader normalises heterogeneous pickle contents (object surface/interior points,
    controller trajectories, visibility masks) into Torch tensors stored on the device
    declared in `cfg.device`. Attributes are later consumed by the Warp-based trainer
    during loss computation and rendering.
    """

    def __init__(self, visualize=False, save_gt=True):
        """Load, preprocess, and optionally visualise recorded point-cloud trajectories.

        Parameters
        ----------
        visualize : bool, optional
            When True, spawns the interactive viewer provided by `visualize_pc` to
            inspect the loaded trajectory on screen.
        save_gt : bool, optional
            When True, saves a ground-truth reference video to `cfg.base_dir/gt.mp4`
            using `visualize_pc` in non-interactive mode.

        Side Effects
        ------------
        * Reads the pickle file located at `cfg.data_path`.
        * May write a ground-truth video alongside experiment outputs.
        """

        logger.info(f"[DATA]: loading data from {cfg.data_path}")
        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        # --- Load pickle produced by the preprocessing pipeline ----------------
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        # Extract structured arrays describing object geometry, colour, and controllers.
        object_points = data["object_points"]
        object_colors = data["object_colors"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        controller_points = data["controller_points"]
        other_surface_points = data["surface_points"]
        interior_points = data["interior_points"]

        # --- Build per-point colour coding for qualitative overlays --------------
        # Map vertical position to a perceptually uniform rainbow palette for clarity
        # when visualising cloth layers and controller influence.
        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

        # Track counts for subsequent graph construction and physics parameterisation.
        self.num_original_points = object_points.shape[1]
        self.num_surface_points = (
            self.num_original_points + other_surface_points.shape[0]
        )
        self.num_all_points = self.num_surface_points + interior_points.shape[0]

        # --- Concatenate structural layers for spring-graph initialisation --------
        self.structure_points = np.concatenate(
            [object_points[0], other_surface_points, interior_points], axis=0
        )
        self.structure_points = torch.tensor(
            self.structure_points, dtype=torch.float32, device=cfg.device
        )

        # --- Materialise tensors on the configured device ------------------------
        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=cfg.device
        )
        self.original_object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=cfg.device
        )
        rainbow_colors = torch.tensor(
            rainbow_colors, dtype=torch.float32, device=cfg.device
        )
        self.object_colors = rainbow_colors.repeat(self.object_points.shape[0], 1, 1)

        # Persist masks indicating per-point visibility and motion validity.
        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=cfg.device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=cfg.device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        self.frame_len = self.object_points.shape[0]
        self.visualize_data(visualize=visualize, save_gt=save_gt)

    def visualize_data(self, visualize=False, save_gt=True):
        """Render point-cloud trajectories for debugging or export ground-truth video.

        Parameters
        ----------
        visualize : bool, optional
            If True, pop up an interactive viewer window to scrub through frames.
        save_gt : bool, optional
            If True, save a `.mp4` video capturing the ground-truth trajectory to disk.

        Side Effects
        ------------
        * Calls `visualize_pc`, which may open GUI windows and write `gt.mp4`.
        """

        if visualize:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=True,
            )
        if save_gt:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=False,
                save_video=True,
                save_path=f"{self.base_dir}/gt.mp4",
            )
