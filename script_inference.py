"""Run inference for every experiment folder.

Each subdirectory inside ``experiments`` is treated as a trained case.  The
script invokes ``inference_warp.py`` with the case name so that the final
trajectory can be generated for each scene.
"""

from __future__ import annotations

import glob
import os
from typing import List


def run_inference(base_path: str, exp_dir: str = "experiments") -> None:
    """Launch inference on all experiments.

    Parameters
    ----------
    base_path:
        Root path of the data used during training/inference.
    exp_dir:
        Directory containing all experiment subfolders.
    """

    dir_names: List[str] = glob.glob(f"{exp_dir}/*")
    for dir_name in dir_names:
        case_name: str = os.path.basename(dir_name)

        # Launch the inference script for the given case.
        cmd: str = (
            f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
        )
        os.system(cmd)


if __name__ == "__main__":
    run_inference("./data/different_types")

