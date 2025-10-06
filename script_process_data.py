"""Batch driver that executes ``process_data.py`` for every entry in ``data_config.csv``."""

import os  # Used to launch subprocesses and inspect dataset directories.
import csv  # Parses the configuration file listing cases, categories, and shape-prior flags.

base_path = "./data/different_types"  # Root directory where all captured cases are stored.

os.system("rm -f timer.log")  # Reset the timing log so each batch run starts with a clean slate.

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue  # Skip entries whose folders have not been created yet.

        if shape_prior.lower() == "true":
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category} --shape_prior"
            )  # Enable shape-prior generation for cases flagged in the CSV file.
        else:
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category}"
            )  # Run the pipeline without creating or aligning a shape prior.
