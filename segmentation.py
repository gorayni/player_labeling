import logging
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from imantics import Polygons
from pixellib.torchbackend.instance import instanceSegmentation
from skimage.io import imread
from tqdm import tqdm

from util import images_size
from util import load_log_configuration


def preprocess_batch(point_rend, frames, height, width):
    inputs = []
    for i in range(len(frames)):
        image = point_rend.predictor.aug.get_transform(frames[i]).apply_image(frames[i])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs.append({"image": image, "height": height, "width": width})
    return inputs


def cut_masks(bboxes, masks, copy=False):
    masks_ = list()
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        if copy:
            masks_.append(np.copy(masks[i][y1:y2, x1:x2]))
        else:
            masks_.append(masks[i][y1:y2, x1:x2])
    return masks_


def to_polygons(masks):
    if isinstance(masks, list):
        return [Polygons.from_mask(m).points for m in masks]
    return Polygons.from_mask(masks).points


def segment(point_rend, inputs, polygonal_masks=True):
    predictions = point_rend.predictor.model(inputs)

    outputs = list()
    for p in predictions:
        masks = p["instances"].pred_masks
        scores = p["instances"].scores
        class_ids = p["instances"].pred_classes

        boxes = p["instances"].pred_boxes.tensor
        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        boxes = boxes.cpu().numpy().astype(np.int16)

        if torch.cuda.is_available():
            class_ids = class_ids.cpu().numpy().astype(np.int16)
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()
        else:
            class_ids = class_ids.numpy().astype(np.int16)
            masks = masks.numpy()
            scores = scores.numpy()

        if polygonal_masks:
            masks = to_polygons(cut_masks(boxes, masks))
        else:
            masks = cut_masks(boxes, masks, True)

        outputs.append({"boxes": boxes,
                        "class_ids": class_ids,
                        "scores": scores,
                        "masks": masks})
    return outputs


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

    for video in tqdm(videos, desc='Overall Progress', leave=True, position=0):
        half = int(video.stem[0]) - 1

        match_path = video.parent
        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        if segmentation_results_fpath.exists():
            continue

        frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'frames')
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f'Extracting frames into {frames_dir}')
            subprocess.check_call(f'./scripts/extract_frames.sh "{str(video).strip()}" "{frames_dir}"', shell=True)

        num_frames = len(list(frames_dir.glob('*.jpg')))
        logging.info(f'Number of frames to segment {num_frames}')

        start = time.time()
        semantic_seg = segment_video(point_rend, frames_dir, num_frames, args.batch_size)
        np.save(segmentation_results_fpath, semantic_seg)
        logging.info(f'Video processing time is {time.time() - start} seconds')
