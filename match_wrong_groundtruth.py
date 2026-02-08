import shutil
from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from IO import load_bboxes
from IO import load_sampling_aspect_ratios
from kitman.regions import iou


PERSON_ID = 0


def load_semantic_segmentation(match_path, half, min_segmentation_score):
    segmentation_results_fpath = match_path.joinpath(f'all_segmentation_results_{half + 1}_HQ.npy')
    semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)
    num_frames = len(semantic_seg)
    return [[bb for bb, class_id, score in zip(semantic_seg[idx]['boxes'],
                                               semantic_seg[idx]['class_ids'],
                                               semantic_seg[idx]['scores']) if
             class_id == PERSON_ID and score > min_segmentation_score] for idx in range(num_frames)]


def load_groundtruth_bboxes(match_path, half, sar):
    half_match_detections = load_bboxes(match_path, half)
    num_frames = len(half_match_detections)
    bboxes_in_frames = [half_match_detections[idx].get('bboxes') for idx in range(num_frames)]
    if sar <= 1:
        return bboxes_in_frames
    return [[[int(sar * bb[0]), bb[1], int(sar * bb[2]), bb[3]] for bb in bboxes] for bboxes in bboxes_in_frames]


def calc_mean_iou_scores(bboxes, semantic_seg):
    num_bboxes, num_semantic_seg = len(bboxes), len(semantic_seg)
    if num_bboxes == 0 or num_semantic_seg == 0:
        return -1

    iou_scores = np.asarray(
        [[iou(semantic_seg[j], bboxes[i]) for j in range(num_semantic_seg)] for i in range(num_bboxes)])

    if num_bboxes <= num_semantic_seg:
        correspondence = np.argmax(iou_scores, axis=1).tolist()
        correspondences = [iou_scores[i, c] for i, c in enumerate(correspondence)]
    else:
        correspondence = np.argmax(iou_scores, axis=0).tolist()
        correspondences = [iou_scores[c, i] for i, c in enumerate(correspondence)]
    return np.mean(correspondences)


def match_video(match_path, half, sar, min_segmentation_score):
    semantic_seg = load_semantic_segmentation(match_path, half, min_segmentation_score)
    num_frames = len(semantic_seg)

    bboxes = load_groundtruth_bboxes(match_path, half, sar)
    num_detected_frames = len(bboxes)

    sampling_freq = int(2 * num_frames / num_detected_frames)

    match_indices, mean_iou_scores = [], []
    last_idx, start_idx, end_idx = 0, 0, min(sampling_freq, num_frames)
    for i in tqdm(range(num_detected_frames), desc=f'{match_path} {half+1}', leave=True, position=0):
        if len(bboxes[i]) != 0:
            max_mean_iou_score, idx = -np.inf, -1
            tested_indices = set()

            n = 30
            for j in range(n):
                indices = sorted(set(np.arange(start_idx, end_idx)) - tested_indices)
                if len(indices) == 0:
                    break
                mean_iou_scores_ = [calc_mean_iou_scores(bboxes[i], semantic_seg[j]) for j in indices]

                max_id = np.argmax(mean_iou_scores_)
                if mean_iou_scores_[max_id] > max_mean_iou_score:
                    max_mean_iou_score = mean_iou_scores_[max_id]
                    idx = indices[max_id]

                if max_mean_iou_score < 0.5:
                    start_idx = max(start_idx - sampling_freq//2, max(last_idx-sampling_freq, 0))
                    end_idx = min(end_idx + sampling_freq//2, len(semantic_seg))
                    tested_indices = tested_indices.union(set(indices))
                else:
                    break

            mean_iou_scores.append(max_mean_iou_score)
            match_indices.append(idx)

            start_idx = idx + sampling_freq // 4
            end_idx = min(idx + ceil(4 * sampling_freq / 5), len(semantic_seg))
        else:
            mean_iou_scores.append(3)
            match_indices.append(start_idx + sampling_freq // 2)
            end_idx = min(start_idx + 10 * sampling_freq, len(semantic_seg))

    return np.asarray(match_indices), np.asarray(mean_iou_scores)


if __name__ == '__main__':
    parser = ArgumentParser(description='Match wrong groundtruth indices')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--single_match',
                       help='Directory path for the match to process (default: None)',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-m', '--matches',
                       help='Path for a file containing a matches list to process',
                       default=None, type=lambda p: Path(p))
    parser.add_argument('-half',
                        help='Match half to process (default: None)',
                        default=None, type=int)
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset (default: data/soccernet)',
                        default="data/soccernet", type=lambda p: Path(p))
    parser.add_argument('--min_segmentation_score',
                        help='Minimum segmentation score for filtering players (default: 0.65)',
                        default=0.65, type=float)
    parser.add_argument('--no_fix_files', required=False,
                        help="Don't fix files (default: False)",
                        action='store_true')
    parser.add_argument('--rematch', required=False,
                        help="Recalculate matching indices (default: False)",
                        action='store_true')
    args = parser.parse_args()

    if args.matches:
        with args.matches.open() as f:
            match_paths = [Path(line) for line in f.read().splitlines()]
    else:
        match_paths = [args.single_match]

    if args.half is not None:
        halves = [args.half]
    else:
        halves = [0, 1]

    sampling_aspect_ratios = load_sampling_aspect_ratios(args.dataset_path)
    for match_path in tqdm(match_paths, desc='Overall Progress', leave=True, position=0):
        sar = sampling_aspect_ratios[match_path]
        for half in halves:

            all_frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'all_frames')
            if not all_frames_dir.exists():
                continue

            fixed_indices_fpath = match_path.joinpath(f'fixed_indices_results_{half + 1}.npz')
            if not fixed_indices_fpath.exists() or args.rematch:
                match_indices, mean_iou_scores = match_video(match_path, half, sar, args.min_segmentation_score)
                np.savez(fixed_indices_fpath, match_indices=match_indices, mean_iou_scores=mean_iou_scores)
            else:
                data = np.load(fixed_indices_fpath)
                match_indices, mean_iou_scores = map(data.get, ['match_indices', 'mean_iou_scores'])
            
            if not args.no_fix_files:
                frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'frames')
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)
                frames_dir.mkdir(parents=True, exist_ok=True)

                all_segmentation_results_fpath = match_path.joinpath(f'all_segmentation_results_{half + 1}_HQ.npy')
                all_semantic_seg = np.load(all_segmentation_results_fpath, allow_pickle=True)

                segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
                if segmentation_results_fpath.exists():
                    if segmentation_results_fpath.is_symlink():
                        segmentation_results_fpath.resolve().unlink()
                    segmentation_results_fpath.unlink()
                elif segmentation_results_fpath.is_symlink():
                    segmentation_results_fpath.unlink()

                semantic_seg = []
                all_frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'all_frames')
                for i, idx in enumerate(match_indices):
                    idx = int(idx)
                    original_frame = all_frames_dir.joinpath(f'{idx + 1:05}.jpg')
                    dest_frame = frames_dir.joinpath(f'{i + 1:05}.jpg')
                    shutil.copyfile(original_frame, dest_frame)
                    semantic_seg.append(all_semantic_seg[idx])
                np.save(segmentation_results_fpath, semantic_seg)

            fig, ax = plt.subplots(figsize=(15, 10))
            plt.plot(np.diff(match_indices))
            ax.set_xlim([0, len(match_indices)])
            plt.xlabel("Frame", fontsize=20)
            plt.ylabel("Consecutive frame index difference", fontsize=20)
            plt.savefig(match_path.joinpath(f'indices_diff_{half + 1}.jpg'))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(15, 10))
            plt.plot(mean_iou_scores)
            ax.set_xlim([0, len(match_indices)])
            plt.xlabel("Frame", fontsize=20)
            plt.ylabel("Mean IoU", fontsize=20)
            plt.savefig(match_path.joinpath(f'mean_iou_scores_{half + 1}.jpg'))
            plt.close(fig)
