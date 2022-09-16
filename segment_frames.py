import logging
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pixellib.torchbackend.instance import instanceSegmentation
from skimage.io import imread
from tqdm import tqdm

from segmentation import preprocess_batch
from segmentation import segment
from util import load_log_configuration


def segment_directory(point_rend, frames_dir, batch_size):
    image_paths = [f for f in frames_dir.glob('*.png')]
    num_frames = len(image_paths)

    im = Image.open(image_paths[0])
    width, height = im.size

    results = []
    for i in tqdm(range(0, num_frames, batch_size)):
        frames = [imread(image_paths[j]) for j in range(i, min(i + batch_size, num_frames))]
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
    group.add_argument('-s', '--single_match',
                       help='Filepath for the match to process',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-m', '--matches',
                       help='Path for files containing a matches list to process',
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
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    args = parser.parse_args()

    load_log_configuration(args.log_config, args.logs_dir)

    point_rend = instanceSegmentation()
    point_rend.load_model(str(args.weights))
    point_rend.predictor.model.cuda()

    if args.matches:
        with args.matches.open() as f:
            matches = [Path(line) for line in f.readlines()]
    else:
        matches = [args.single_match]

    for match_path in tqdm(matches, desc='Overall Progress', leave=True, position=0):

        segmentation_results_fpath = match_path.joinpath('segmentation_results_v3.npy')
        if segmentation_results_fpath.exists():
            print(match_path)
            continue

        frames_dir = match_path.joinpath('frames_v3')
        start = time.time()
        semantic_seg = segment_directory(point_rend, frames_dir, args.batch_size)
        np.save(segmentation_results_fpath, semantic_seg)
        logging.info(f'Video processing time is {time.time() - start} seconds')
