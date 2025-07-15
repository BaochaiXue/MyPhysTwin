"""Batch data processing based on ``data_config.csv``.

This script iterates over rows in ``data_config.csv`` and calls
``process_data.py`` for each case.  If ``shape_prior`` is set to ``true`` for a
case, the ``--shape_prior`` flag is included when invoking ``process_data.py``.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable


def process_all_cases(base_path: str, config_file: str = "data_config.csv") -> None:
    """Read ``config_file`` and invoke ``process_data.py`` for each case."""

    # Remove previous timer log if it exists.
    os.system("rm -f timer.log")

    with open(config_file, newline="", encoding="utf-8") as csvfile:
        reader: Iterable[list[str]] = csv.reader(csvfile)
        for row in reader:
            case_name: str = row[0]
            category: str = row[1]
            shape_prior: str = row[2]

            if not os.path.exists(f"{base_path}/{case_name}"):
                continue

            if shape_prior.lower() == "true":
                cmd: str = (
                    f"python process_data.py --base_path {base_path} "
                    f"--case_name {case_name} --category {category} --shape_prior"
                )
            else:
                cmd = (
                    f"python process_data.py --base_path {base_path} "
                    f"--case_name {case_name} --category {category}"
                )
            os.system(cmd)


if __name__ == "__main__":
    process_all_cases("./data/different_types")

