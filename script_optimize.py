"""Utility to launch CMA‑ES optimization for every dataset.

This module iterates over all case folders inside ``base_path`` and invokes
``optimize_cma.py`` on the second frame of the training split defined in each
``split.json`` file.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List


def run_optimization(base_path: str) -> None:
    """Run CMA-ES optimization for all scenes under ``base_path``.

    Parameters
    ----------
    base_path:
        Root directory that contains subdirectories for each scene. Each scene
        must provide a ``split.json`` file describing the train/test split.
    """

    # Collect all case directories within ``base_path``.
    dir_names: List[str] = glob.glob(f"{base_path}/*")
    for dir_name in dir_names:
        # Each directory name becomes the case name.
        case_name: str = os.path.basename(dir_name)

        # Read the training/testing split information.
        with open(f"{base_path}/{case_name}/split.json", "r", encoding="utf-8") as f:
            split: dict[str, list[int]] = json.load(f)

        # ``train`` is assumed to be a two-element list; we use the second
        # element as the frame index for optimization.
        train_frame: int = split["train"][1]

        # Execute CMA-ES optimization on the selected frame.
        cmd: str = (
            f"python optimize_cma.py --base_path {base_path} "
            f"--case_name {case_name} --train_frame {train_frame}"
        )
        os.system(cmd)


if __name__ == "__main__":
    run_optimization("./data/different_types")
