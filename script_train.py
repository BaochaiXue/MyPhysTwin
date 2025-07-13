"""Utility script to run training for all scenes in ``base_path``."""

from __future__ import annotations

import glob
import os
import json

base_path: str = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    # Execute the training script for the selected frame
    os.system(
        f"python train_warp.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
    )
