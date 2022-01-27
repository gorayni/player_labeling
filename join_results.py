from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from addict import Dict
from tqdm import tqdm

from IO import load_bboxes


def get_segmented_people(semantic_seg):
    bboxes, scores = [], []
    for bb, score, class_id in zip(semantic_seg['boxes'], semantic_seg['scores'], semantic_seg['class_ids']):
        if class_id != 0: # Person class ID is 0
            continue
        bboxes.append(bb)
        scores.append(score)

    return bboxes, scores


def main(match_path):
    velocity_results = np.load(match_path.joinpath('velocity_results.npy'), allow_pickle=True).item()
    team_classification_results = np.load(match_path.joinpath(f'team_classification_results.npy'),
                                          allow_pickle=True).item()

    people = {}
    for half in range(0, 2):
        num_rgb_frames = len(load_bboxes(match_path, half))

        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        people[half] = {}
        for frame_idx in range(num_rgb_frames):
            bboxes, scores = get_segmented_people(semantic_seg[frame_idx])
            if len(bboxes) == 0:
                people[half][frame_idx] = None
                continue

            people[half][frame_idx] = {'bboxes': bboxes,
                                       'scores': scores,
                                       'velocities': velocity_results[half][frame_idx]['velocity'],
                                       'teams': team_classification_results[half][frame_idx]}
    return people


def parse_args():
    parser = ArgumentParser(description='Segmentation, velocity and team classification results joiner')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--single_match',
                       help='Directory path for the match to process (default: None)',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-m', '--matches',
                       help='Path for a file containing a matches list to process',
                       default=None, type=lambda p: Path(p))
    args = parser.parse_args()

    if args.matches:
        with args.matches.open() as f:
            match_paths = [Path(m) for m in f.read().splitlines()]
    else:
        match_paths = [args.single_match]

    return Dict({'match_paths': match_paths})


if __name__ == '__main__':
    args = parse_args()

    for match_path in tqdm(args.match_paths, desc='Overall Progress', leave=True, position=0):
        joined_results_fpath = match_path.joinpath(f'player_velocity_team_results.npy')
        if joined_results_fpath.exists():
            continue
        results = main(match_path)
        np.save(joined_results_fpath, results)
