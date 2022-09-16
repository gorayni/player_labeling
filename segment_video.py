import logging
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from pixellib.torchbackend.instance import instanceSegmentation
from skimage.io import imread
from tqdm import tqdm

from segmentation import preprocess_batch
from segmentation import segment
from util import images_size
from util import load_log_configuration


def segment_video(point_rend, frames_dir, num_frames, batch_size):
    height, width = images_size(frames_dir)
    results = []
    for i in tqdm(range(0, num_frames, batch_size)):
        frames = [imread(frames_dir.joinpath(f'{j + 1:05}.jpg')) for j in range(i, min(i + batch_size, num_frames))]
        with torch.no_grad():
            inputs = preprocess_batch(point_rend, frames, height, width)
            predictions = segment(point_rend, inputs)
            torch.cuda.empty_cache()

            results.extend(predictions)
            del inputs
    return results


if __name__ == '__main__':

    parser = ArgumentParser(description='PointRend Segmentation for Players')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--single_video',
                       help='Filepath for the video to process',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-v', '--videos',
                       help='Path for files containing a video list to process',
                       default=None, type=lambda p: Path(p))
    parser.add_argument('--batch_size', required=False,
                        help='PointRend batch size (default: 8)',
                        default=8, type=int)
    parser.add_argument('--weights', required=False,
                        help='Weights of PointRend to load (default: "weights/pointrend_resnet50.pkl")',
                        default='weights/pointrend_resnet50.pkl', type=lambda p: Path(p))
    parser.add_argument('--logs_dir', required=False,
                        help='Path for segmentation logging directory (default: segmentation_logs)',
                        default="segmentation_logs", type=lambda p: Path(p))
    parser.add_argument('--all_frames', required=False,
                        help="Extract and segment all frames from video (default: False)",
                        action='store_true')
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    args = parser.parse_args()

    load_log_configuration(args.log_config, args.logs_dir)

    point_rend = instanceSegmentation()
    point_rend.load_model(str(args.weights))
    point_rend.predictor.model.cuda()

    if args.videos:
        with args.videos.open() as f:
            videos = [Path(line) for line in f.readlines()]
    else:
        videos = [args.single_video]

    all_frames_prefix = 'all_' if args.all_frames else ''
    for video in tqdm(videos, desc='Overall Progress', leave=True, position=0):
        half = int(video.stem[0]) - 1

        match_path = video.parent
        segmentation_results_fpath = match_path.joinpath(f'{all_frames_prefix}segmentation_results_{half + 1}_HQ.npy')
        if segmentation_results_fpath.exists():
            print(video)
            continue

        frames_dir = match_path.joinpath(f'{half + 1}_HQ', f'{all_frames_prefix}frames')
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f'Extracting frames into {frames_dir}')
            if args.all_frames:
                subprocess.check_call(f'./scripts/extract_frames.sh -a "{str(video).strip()}" "{frames_dir}"', shell=True)
            else:
                subprocess.check_call(f'./scripts/extract_frames.sh "{str(video).strip()}" "{frames_dir}"', shell=True)

        num_frames = len(list(frames_dir.glob('*.jpg')))
        logging.info(f'Number of frames to segment {num_frames}')

        start = time.time()
        semantic_seg = segment_video(point_rend, frames_dir, num_frames, args.batch_size)
        np.save(segmentation_results_fpath, semantic_seg)
        logging.info(f'Video processing time is {time.time() - start} seconds')
