"""Run training for each scene found in ``base_path``.

The script scans all subfolders under ``base_path`` and launches
``train_warp.py`` for the specified training frame in every scene's
``split.json`` configuration.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List


def run_training(base_path: str) -> None:
    """Execute training for all available scenes."""

    dir_names: List[str] = glob.glob(f"{base_path}/*")
    for dir_name in dir_names:
        case_name: str = os.path.basename(dir_name)

        # Load the frame split information for this case.
        with open(f"{base_path}/{case_name}/split.json", "r", encoding="utf-8") as f:
            split: dict[str, list[int]] = json.load(f)

        train_frame: int = split["train"][1]

        # Invoke the actual training script for the selected frame.
        cmd: str = (
            f"python train_warp.py --base_path {base_path} "
            f"--case_name {case_name} --train_frame {train_frame}"
        )
        os.system(cmd)


if __name__ == "__main__":
    run_training("./data/different_types")

