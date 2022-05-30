import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from IO import load_bboxes
from IO import load_calibration
from IO import load_sampling_aspect_ratios
from regions import area
from regions import iou


def parse_args():
    parser = argparse.ArgumentParser(description='Find wrong groundtruth data')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--single_match',
                       help='Directory path for the match to process (default: None)',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-m', '--matches',
                       help='Path for a file containing a matches list to process',
                       default=None, type=lambda p: Path(p))
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset (default: data/soccernet)',
                        default="data/soccernet", type=lambda p: Path(p))
    parser.add_argument('--min_segmentation_score',
                        help='Minimum segmentation score for filtering players (default: 0.65)',
                        default=0.65, type=float)
    parser.add_argument('--seed', help='random seed (default: 42)',
                        default=42, type=int)
    return parser.parse_args()


def filter_small_players(match_path, min_calibration_confidence=0.85, max_area=40000):
    small_players = {0: {}, 1: {}}
    for half in range(2):
        half_match_detections = load_bboxes(match_path, half)
        half_match_calibration = load_calibration(match_path, half)
        idx, num_frames = -1, len(half_match_detections)

        while (idx := idx + 1) < num_frames:
            calibration_confidence = half_match_calibration[idx][0]["confidence"]
            if calibration_confidence < min_calibration_confidence:
                continue
            bboxes, onfield = map(half_match_detections[idx].get, ['bboxes', 'onfield'])
            bboxes = [bb for bb, on in zip(bboxes, onfield) if on]
            if len(bboxes) < 20 or any(area(bb) > max_area for bb in bboxes):
                continue

            small_players[half][idx] = bboxes
    return small_players


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(args.seed)

    if args.matches:
        with args.matches.open() as f:
            match_paths = [Path(line) for line in f.read().splitlines()]
    else:
        match_paths = [args.single_match]

    sampling_aspect_ratios = load_sampling_aspect_ratios(args.dataset_path)

    half = 0
    PERSON_ID = 0
    for match_path in tqdm(match_paths, desc='Overall Progress', leave=True, position=0):
        half_match_detections = load_bboxes(match_path, half)
        num_frames = len(half_match_detections)

        bboxes = filter_small_players(match_path)
        indices = list(bboxes[half].keys())

        if len(indices) == 0:
            print('-', match_path)
            continue
        idx = indices[int(np.random.choice(len(indices), 1, replace=False)[0])]
        bboxes = bboxes[half][idx]

        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        ss_frame = [bb for bb, class_id, score in zip(semantic_seg[idx]['boxes'],
                                                      semantic_seg[idx]['class_ids'],
                                                      semantic_seg[idx]['scores']) if
                    class_id == PERSON_ID and score > args.min_segmentation_score]

        sar = sampling_aspect_ratios[match_path]
        if sar > 1:
            bboxes = [[sar * bb[0], bb[1], sar * bb[2], bb[3]] for bb in bboxes]

        num_bboxes, num_objects = len(bboxes), len(ss_frame)

        iou_scores = np.asarray([[iou(ss_frame[j], bboxes[i]) for j in range(num_objects)] for i in range(num_bboxes)])

        if num_bboxes <= num_objects:
            correspondence = np.argmax(iou_scores, axis=1).tolist()
            correspondences = [(bboxes[i], ss_frame[c]) for i, c in enumerate(correspondence) if iou_scores[i, c] > 0.3]
        else:
            correspondence = np.argmax(iou_scores, axis=0).tolist()
            correspondences = [(bboxes[c], ss_frame[i]) for i, c in enumerate(correspondence) if iou_scores[c, i] > 0.3]

        if len(correspondences) / min(num_bboxes, num_objects) < 0.8:
            print(match_path, idx, len(correspondences) / min(num_bboxes, num_objects))
