# Stage 1—CLI Entrypoint & Runtime Orchestrator
# Role: Parse playground CLI flags, hydrate configuration state, and hand off to
#       `InvPhyTrainerWarp.interactive_playground` for the live simulator/render loop.
# Inputs: CLI args (`--base_path`, `--gaussian_path`, `--case_name`, etc.),
#         per-case artifacts (`final_data.pkl`, `optimal_params.pkl`,
#         `calibrate.pkl`, `metadata.json`), Gaussian checkpoint folder.
# Outputs: Launches interactive session, writes logs under
#          `./temp_experiments/<case_name>/inference_log.log`, triggers trainer to
#          render/playback outputs in the chosen experiment directory.
# Key in-house deps: `qqtt.InvPhyTrainerWarp`, `qqtt.utils.cfg`, `qqtt.utils.logger`.
# Side effects: Seeds global RNGs (including CUDA), mutates `cfg`, reads pickles/JSON,
#               creates directories via logger, and opens GUI windows via the trainer.
# Assumptions: Offline optimisation already produced the case folders; CUDA present
#              for PyTorch seeding; filesystem layout matches repo conventions.

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed):
    """Synchronise pseudo-random generators across Python/NumPy/PyTorch.

    Parameters
    ----------
    seed : int
        Global random seed propagated to Python's `random`, NumPy, and PyTorch RNGs
        (CPU + CUDA) so that dataset sampling, controller shuffles, and simulator
        initial conditions remain deterministic across launches.

    Side Effects
    ------------
    * Sets PyTorch CUDA RNGs on all devices and toggles cuDNN into deterministic mode.
    * Disables cuDNN benchmarking to avoid autotuner-induced nondeterminism between runs.
    """

    # Reseed Python's built-in RNG so downstream modules that rely on `random` (e.g.
    # sampling controller IDs) produce reproducible sequences.
    random.seed(seed)
    # Align NumPy's RNG because many preprocessing steps (point-cloud shuffling,
    # geometric augmentation) leverage NumPy utilities.
    np.random.seed(seed)
    # Seed PyTorch's CPU generator to stabilise dataset shuffling and tensor init.
    torch.manual_seed(seed)
    # Seed the default CUDA RNG so GPU kernels that sample (e.g. simulator noise) repeat.
    torch.cuda.manual_seed(seed)
    # Seed all CUDA devices when executing in multi-GPU contexts to avoid divergent
    # behaviour if tensors get replicated across devices by the trainer.
    torch.cuda.manual_seed_all(seed)
    # Force cuDNN to choose deterministic algorithms, preventing run-to-run variance.
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN's auto-tuner because it may swap kernels between runs, breaking
    # determinism that is critical for debugging inverse-physics optimisation.
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    # --- CLI argument parsing --------------------------------------------------
    # The parser exposes the minimal knobs required to point the playground at
    # different capture sessions and control setups collected offline.
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
        help=(
            "Root directory containing case folders (each with calibrations, metadata,"
            " and final trajectory pickles)."
        ),
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
        help=(
            "Directory produced by Gaussian Splatting training; expected to contain"
            " per-case subfolders with reconstructed point-cloud checkpoints."
        ),
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.png",
        help="Background image blended behind the rendered Gaussian splat."
    )
    parser.add_argument(
        "--case_name", type=str, default="double_lift_cloth_3",
        help="Identifier for the capture session; used to locate all per-case assets."
    )
    parser.add_argument(
        "--n_ctrl_parts", type=int, default=2,
        help="Number of virtual controllers (1 or 2) exposed in the UI overlay."
    )
    parser.add_argument(
        "--inv_ctrl", action="store_true",
        help="Invert horizontal controller motion to match mirrored camera setups."
    )
    parser.add_argument(
        "--virtual_key_input", action="store_true",
        help="Drive the simulator with a synthetic keyboard event stream instead of"
             " live hardware inputs; useful for demos/replays."
    )
    # Parse CLI args once; ArgumentParser validates type conversions and falls back to
    # curated defaults if the user omits flags.
    args = parser.parse_args()

    # --- Global configuration selection ---------------------------------------
    # Store frequently reused CLI arguments to avoid repeated dotted lookups below.
    base_path = args.base_path
    case_name = args.case_name

    # Select cloth-vs-rigid configuration presets. The YAML files override defaults in
    # `cfg` and encode constitutive-model assumptions specific to soft objects.
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    # Derive the experiment output directory where logs and inference videos will live.
    # Compose the per-case experiment directory where logs and media will be written.
    base_dir = f"./temp_experiments/{case_name}"

    # --- Offline optimisation artefact loading --------------------------------
    # Ingest the upstream optimisation results; these hold calibrated parameters (e.g.
    # material stiffness) that cannot be differentiated during live interaction.
    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    # Log the resolved path for traceability when debugging missing artefacts.
    logger.info(f"Load optimal parameters from: {optimal_path}")
    # Fail fast if offline optimisation has not been executed for this case.
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    # Unpickle the parameter dictionary (NumPy arrays / scalars) residing on disk.
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    # Push the static parameters into the global config so downstream modules read the
    # same tuned values (spring stiffness, damping, etc.).
    cfg.set_optimal_params(optimal_params)

    # --- Camera calibration + rendering metadata -------------------------------
    # Load calibrated camera extrinsics (c2w) for every frame; invert to get w2c so the
    # renderer can project simulator vertices correctly.
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)  # Store homogeneous camera-to-world transforms.
    cfg.w2cs = np.array(w2cs)  # Store inverse transforms for convenient lookups.
    # Camera intrinsics and viewport dimensions drive the Gaussian renderer's frusta.
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path  # Record background image path for blending.

    # --- Locate Gaussian Splatting checkpoint ----------------------------------
    # Gaussian Splatting checkpoints follow a deterministic naming pattern based on
    # training hyperparameters; we currently hard-code the exp_name to match the best
    # performing configuration from upstream experiments.
    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = (
        f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/"
        "point_cloud.ply"
    )

    # --- Trainer bootstrap + interactive loop ----------------------------------
    # Route runtime logs to the case-specific directory so multiple launches do not
    # overwrite each other. The logger internally handles directory creation.
    logger.set_log_file(path=base_dir, name="inference_log")
    # Instantiate the trainer in inference-only mode: disables gradient-based updates
    # but still wires up the Warp-based simulator and rendering subsystems.
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    # Grab the best-performing checkpoint from the training runs; the naming convention
    # includes the iteration number to distinguish between multiple potential bests.
    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    # Delegate to the trainer’s interactive playground loop, which spawns the renderer,
    # keyboard listener, and simulation update cycle using the configured assets.
    trainer.interactive_playground(
        best_model_path,
        gaussians_path,
        args.n_ctrl_parts,
        args.inv_ctrl,
        virtual_key_input=args.virtual_key_input,
    )
