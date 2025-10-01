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
import re


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    expected_springs = trainer.simulator.n_springs

    def _extract_epoch(path: str) -> int:
        match = re.search(r"best_(\\d+)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    candidate_paths = sorted(glob.glob(f"{base_dir}/train/best_*.pth"))
    assert candidate_paths, f"No checkpoint found under {base_dir}/train"

    matching_models = []
    for path in candidate_paths:
        checkpoint = torch.load(path, map_location=cfg.device)
        spring_len = len(checkpoint.get("spring_Y", []))
        if spring_len == expected_springs:
            matching_models.append((_extract_epoch(path), path))
        else:
            logger.warning(
                f"Skip {path}: checkpoint has {spring_len} springs, expected {expected_springs}"
            )

    assert (
        matching_models
    ), "No checkpoint matches current topology. Check experiments directory or regenerate models."
    # Pick the checkpoint with the highest epoch among the matches.
    matching_models.sort(reverse=True)
    best_model_path = matching_models[0][1]
    trainer.test(best_model_path)
