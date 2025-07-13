"""Run inference on all trained experiment folders."""

from __future__ import annotations

import glob
import os
import json

base_path: str = "./data/different_types"
dir_names = glob.glob("experiments/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    # Launch the inference script for the given case
    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
    )
