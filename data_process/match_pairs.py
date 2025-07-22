#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
"""Utilities for matching keypoints between images using SuperPoint and SuperGlue."""

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from models.matching import Matching
from models.utils import (
    make_matching_plot,
    AverageTimer,
    read_image,
)

torch.set_grad_enabled(False)


def image_pair_matching(
    input_images: Sequence[np.ndarray],
    ref_image: np.ndarray,
    output_dir: str,
    resize: Sequence[int] | None = [-1],
    resize_float: bool = False,
    superglue: str = "indoor",
    max_keypoints: int = 1024,
    keypoint_threshold: float = 0.005,
    nms_radius: int = 4,
    sinkhorn_iterations: int = 20,
    match_threshold: float = 0.2,
    viz: bool = False,
    fast_viz: bool = False,
    cache: bool = True,
    show_keypoints: bool = False,
    viz_extension: str = "png",
    save: bool = False,
    viz_best: bool = True,
) -> tuple[int, dict]:
    """Match features between input images and a reference image using SuperGlue.

    Args:
        input_images: List of image arrays to match.
        ref_image: The reference image used for matching.
        output_dir: Directory to save intermediate results.
        resize: Optional resize setting for images.
        resize_float: If True, resize using float precision.
        superglue: Pretrained SuperGlue weights to use.
        max_keypoints: Maximum keypoints detected per image.
        keypoint_threshold: Detector threshold.
        nms_radius: NMS radius for keypoints.
        sinkhorn_iterations: Sinkhorn iterations in SuperGlue.
        match_threshold: Matching threshold.
        viz: Whether to save matching visualisations.
        fast_viz: Use fast visualisation.
        cache: Reuse cached matches if available.
        show_keypoints: Display keypoints in visualisations.
        viz_extension: File format for visualisation images.
        save: Save match result arrays.
        viz_best: Visualise the best matching pair.

    Returns:
        Tuple of index of the best pose and match dictionary for that pose.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Running inference on device "{}"'.format(device))
    config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": superglue,
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }
    matching = Matching(config).eval().to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory "{}"'.format(output_dir))
    if viz:
        print('`Will writ`e visualization images to directory "{}"'.format(output_dir))

    timer = AverageTimer(newline=True)
    match_nums = []
    match_result = []

    best_match = {}
    best_match_num = -1

    for i, image in enumerate(input_images):
        matches_path = output_dir / "matches_{}.npz".format(i)
        viz_path = output_dir / "matches_{}.{}".format(i, viz_extension)

        do_match = True
        do_viz = viz
        if cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError("Cannot load matches .npz file: %s" % matches_path)

                kpts0, kpts1 = results["keypoints0"], results["keypoints1"]
                matches, conf = results["matches"], results["match_confidence"]
                do_match = False
            if viz and viz_path.exists():
                do_viz = False
            timer.update("load_cache")

        rot0, rot1 = 0, 0
        image0, inp0, scales0 = read_image(image, device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(
            ref_image, device, resize, rot1, resize_float
        )
        if image0 is None or image1 is None:
            print("Problem reading image pair: {} and ref".format(i))
            exit(1)
        timer.update("load_image")

        if do_match:
            pred = matching({"image0": inp0, "image1": inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            matches, conf = pred["matches0"], pred["matching_scores0"]
            timer.update("matcher")

            out_matches = {
                "keypoints0": kpts0,
                "keypoints1": kpts1,
                "matches": matches,
                "match_confidence": conf,
            }
            match_result.append(out_matches)
            if save:
                np.savez(str(matches_path), **out_matches)
        else:
            match_result.append(results)

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        match_nums.append(len(mkpts0))

        if len(mkpts0) > best_match_num:
            best_match_num = len(mkpts0)
            best_match["image0"] = image0
            best_match["image1"] = image1
            best_match["kpts0"] = kpts0
            best_match["kpts1"] = kpts1
            best_match["mkpts0"] = mkpts0
            best_match["mkpts1"] = mkpts1
            best_match["mconf"] = mconf

        if do_viz:
            color = cm.jet(mconf)
            text = [
                "SuperGlue",
                "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
                "Matches: {}".format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append("Rotation: {}:{}".format(rot0, rot1))

            k_thresh = matching.superpoint.config["keypoint_threshold"]
            m_thresh = matching.superglue.config["match_threshold"]
            small_text = [
                "Keypoint Threshold: {:.4f}".format(k_thresh),
                "Match Threshold: {:.2f}".format(m_thresh),
                "Image Pair: {} : ref".format(i),
            ]

            make_matching_plot(
                image0,
                image1,
                kpts0,
                kpts1,
                mkpts0,
                mkpts1,
                color,
                text,
                viz_path,
                show_keypoints,
                fast_viz,
                small_text,
            )

            timer.update("viz_match")
    best_pose = match_nums.index(max(match_nums))

    if viz_best:
        viz_path = f"{output_dir}/best_match.{viz_extension}"
        color = cm.jet(best_match["mconf"])
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(
                len(best_match["kpts0"]), len(best_match["kpts1"])
            ),
            "Matches: {}".format(len(best_match["mkpts0"])),
        ]

        make_matching_plot(
            best_match["image0"],
            best_match["image1"],
            best_match["kpts0"],
            best_match["kpts1"],
            best_match["mkpts0"],
            best_match["mkpts1"],
            color,
            text,
            viz_path,
            show_keypoints,
            fast_viz,
        )

        timer.update("viz_match")
    return best_pose, match_result[best_pose]
