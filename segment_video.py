import logging
import subprocess
import time
from kitman.snv2 import MatchPaths
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from pixellib.torchbackend.instance import instanceSegmentation as PointRend
from tqdm import tqdm

from segmentation import segment_directory
from util import load_log_configuration


if __name__ == "__main__":

    parser = ArgumentParser(description="PointRend Segmentation for Players")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-s",
        "--single_video",
        help="Filepath for the video to process",
        default=None,
        type=lambda p: Path(p),
    )
    group.add_argument(
        "-v",
        "--videos",
        help="Path for files containing a video list to process",
        default=None,
        type=lambda p: Path(p),
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        help="PointRend batch size (default: 8)",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--weights",
        required=False,
        help='Weights of PointRend to load (default: "weights/pointrend_resnet50.pkl")',
        default="weights/pointrend_resnet50.pkl",
        type=lambda p: Path(p),
    )
    parser.add_argument(
        "--logs_dir",
        required=False,
        help="Path for segmentation logging directory (default: segmentation_logs)",
        default="segmentation_logs",
        type=lambda p: Path(p),
    )
    parser.add_argument(
        "--all_frames",
        required=False,
        help="Extract and segment all frames from video (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--log_config",
        required=False,
        help="Logging configuration file (default: config/log_config.yml)",
        default="config/log_config.yml",
        type=lambda p: Path(p),
    )
    args = parser.parse_args()

    load_log_configuration(args.log_config, args.logs_dir)

    point_rend = PointRend()
    point_rend.load_model(str(args.weights))
    point_rend.predictor.model.cuda()

    if args.videos:
        with args.videos.open() as f:
            videos = [Path(strip(line)) for line in f.readlines()]
    else:
        videos = [args.single_video]

    for video in tqdm(videos, desc="Overall Progress", leave=True, position=0):

        half = int(video.stem[0]) - 1

        match_paths = MatchPaths(video.parent)
        if match_paths.segmentations[half].exists:
            print(video)
            continue

        frames_dir = match_paths.frames[half].path
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Extracting frames into {frames_dir}")
            if args.all_frames:
                subprocess.check_call(
                    f'./scripts/extract_frames.sh -a "{str(video)}" "{frames_dir}"',
                    shell=True,
                )
            else:
                subprocess.check_call(
                    f'./scripts/extract_frames.sh "{str(video)}" "{frames_dir}"',
                    shell=True,
                )

        start = time.time()
        semantic_seg = segment_directory(point_rend, match_paths.frames[half], args.batch_size)
        np.save(match_paths.segmentations[half], semantic_seg)
        logging.info(f"Video processing time is {time.time() - start} seconds")
