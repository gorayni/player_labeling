import argparse
import logging
import pickle
import shutil
from collections import defaultdict
from itertools import product
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage import io
from skimage.morphology import binary_erosion
from skimage.morphology import disk
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from IO import Players
from IO import load_bboxes
from IO import load_calibration
from IO import load_sampling_aspect_ratios
from cluster_midfielders import label_midfielders
from field_calibration import GOAL_CENTERS
from field_calibration import calculate_dist_from_goals
from field_calibration import calculate_radar_position
from field_calibration import load_homography
from regions import area
from regions import calculate_patch_hist
from regions import get_masked_patch
from regions import get_patch
from regions import iou
from regions import bhattacharyya_distance
from regions import rgb2lab
from regions import to_mask
from util import load_log_configuration


class Blob:
    def __init__(self, bb, mask_bb, mask_cnt, hist, is_dark=False, category=None):
        self.bb = bb
        self.mask_bb = mask_bb
        self.mask_cnt = mask_cnt
        self.hist = hist
        self.is_dark = is_dark
        self.category = category


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
            if len(bboxes) == 0 or any(area(bb) > max_area for bb in bboxes):
                continue

            small_players[half][idx] = bboxes
    return small_players


def match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx, min_segmentation_score=0.65, sar=1):
    PERSON_ID = 0
    ss_frame = [(bb, mask_cnts, score) for bb, mask_cnts, class_id, score in zip(semantic_seg[idx]['boxes'],
                                                                                 semantic_seg[idx]['masks'],
                                                                                 semantic_seg[idx]['class_ids'],
                                                                                 semantic_seg[idx]['scores']) if
                class_id == PERSON_ID and score > min_segmentation_score]

    if sar > 1:
        bboxes = [[sar * bb[0], bb[1], sar * bb[2], bb[3]] for bb in bboxes]

    num_bboxes, num_objects = len(bboxes), len(ss_frame)
    iou_scores = np.asarray(
        [[iou(ss_frame[j][0], bboxes[i]) for j in range(num_objects)] for i in range(num_bboxes)])

    if num_bboxes <= num_objects:
        correspondence = np.argmax(iou_scores, axis=1).tolist()
        return [(bboxes[i], ss_frame[c][0], ss_frame[c][1], ss_frame[c][2]) for i, c in enumerate(correspondence) if
                iou_scores[i, c] > 0.1]
    else:
        correspondence = np.argmax(iou_scores, axis=0).tolist()
        return [(bboxes[c], ss_frame[i][0], ss_frame[i][1], ss_frame[i][2]) for i, c in enumerate(correspondence) if
                iou_scores[c, i] > 0.1]


def calculate_masked_patch_hist(frame, mask_bb, mask_cnt, erosion_disk_radius=2, hist_type='rgb'):
    patch = get_patch(frame, mask_bb, False)
    mask = to_mask(mask_bb, mask_cnt)
    if erosion_disk_radius > 0:
        eroded_mask = np.zeros(mask.shape, dtype=np.uint8)
        binary_erosion(mask, disk(erosion_disk_radius), out=eroded_mask)
        mask = eroded_mask
    return calculate_patch_hist(patch, mask, hist_type)


def filter_bench_people(homography, bboxes, bench_dist):
    return [bb for bb in bboxes if calculate_radar_position(bb, homography)[1] < bench_dist]


def extract_midfielders_blobs(match_path, players_bboxes, sar, min_segmentation_score=0.65, min_dist_to_goals=20,
                              min_player_bb_area=1000, erosion_disk_radius=2, hist_type='rgb', bench_dist=29):
    players_blobs = {0: {}, 1: {}}
    for half in range(2):
        half_match_calibration = load_calibration(match_path, half)
        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')

        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        for idx, bboxes in players_bboxes[half].items():

            homography = load_homography(half_match_calibration[idx][0]["homography"])

            distances = [calculate_dist_from_goals(bb, homography) for bb in bboxes]
            if any(d < min_dist_to_goals for d in distances):
                continue

            if bench_dist:
                bboxes = filter_bench_people(homography, bboxes, bench_dist)

            if len(bboxes) == 0:
                continue

            bboxes_masks = match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx, min_segmentation_score, sar)

            players_blobs[half][idx] = list()
            frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
            f = io.imread(frame_path)

            dark_regions_mask = calculate_dark_regions_mask(f)
            dark_contours = cv.findContours(dark_regions_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

            for bb, mask_bb, mask_cnt, _ in bboxes_masks:
                if area(mask_bb) < min_player_bb_area:
                    continue

                is_dark = False
                if len(dark_contours) > 0:
                    bb_centroid = ((bb[2] + bb[0]) // 2, (bb[3] + bb[1]) // 2)
                    is_dark = in_dark_region(bb_centroid, dark_contours)

                hist = calculate_masked_patch_hist(f, mask_bb, mask_cnt, erosion_disk_radius, hist_type)

                players_blobs[half][idx].append(Blob(bb, mask_bb, mask_cnt, hist, is_dark))
    return players_blobs


def filter_bboxes_by_size(midfielders_blobs):
    bboxes = [blob.mask_bb for half in range(2) for _, frame in midfielders_blobs[half].items() for blob in frame]

    bb_sizes = np.asarray([[bb[2] - bb[0], bb[3] - bb[1]] for bb in bboxes])
    gmm = GaussianMixture(n_components=20, random_state=0).fit(bb_sizes)

    scores = gmm.score_samples(bb_sizes)
    scores -= scores.min()
    scores /= scores.max()

    filtered_blobs, k = {}, 0
    for half, players_blobs in midfielders_blobs.items():
        filtered_blobs[half] = {}
        for frame_idx, blobs in players_blobs.items():
            filtered_blobs[half][frame_idx] = []
            for blob in blobs:
                if scores[k] > 0.5:
                    filtered_blobs[half][frame_idx].append(blob)
                k += 1
    return filtered_blobs


def calculate_dark_regions_mask(img, brightness_threshold=100, opening_disk_radius=20, min_area_proportion=0.05):
    mask = rgb2lab(img)[:, :, 0] < brightness_threshold
    mask = 255 * mask.astype(np.uint8)

    # Removing small regions and connecting big ones
    disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (opening_disk_radius, opening_disk_radius))
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, disk)

    # Filtering regions by area
    min_field_area = np.power(min_area_proportion * np.max(opened_mask.shape[:2]), 2)

    filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
    contours = cv.findContours(opened_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        if cv.contourArea(cnt) >= min_field_area:
            # cv.drawContours(filtered_mask, [cnt], 0, 255, -1)
            hull = cv.convexHull(cnt)
            cv.drawContours(filtered_mask, [hull], 0, 255, -1)
    return filtered_mask


def fix_dark_regions(img, brightness_correction=60, brightness_threshold=100, opening_disk_radius=20, min_area_proportion=0.1):
    mask = calculate_dark_regions_mask(img, brightness_threshold, opening_disk_radius, min_area_proportion)
    mask = np.repeat(mask[:, :, np.newaxis]//255, 3, axis=2)
    return cv.add(img, brightness_correction * mask)


def in_dark_region(point, contours):
    for cnt in contours:
        if cv.pointPolygonTest(cnt, point, True) > 0:
            return True
    return False


def extract_goalkeeper(match_path, half, semantic_seg, sar, goal_center, category, players_bboxes, labels,
                       min_segmentation_score=0.65, min_goalkeeper_and_goal_dist=5, min_dist_to_goalkeeper=2.5,
                       min_player_bb_area=1000, erosion_disk_radius=2, hist_type='rgb'):
    half_match_calibration = load_calibration(match_path, half)
    frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')

    for idx, bboxes in players_bboxes[half].items():

        homography = load_homography(half_match_calibration[idx][0]["homography"])

        bboxes_masks = match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx, min_segmentation_score, sar)
        if len(bboxes_masks) == 0:
            continue

        positions = np.asarray([calculate_radar_position(bb, homography) for bb, _, _, _ in bboxes_masks])
        distances_to_goal = np.linalg.norm(positions - goal_center, axis=1)
        gk_idx = np.argmin(distances_to_goal)
        if distances_to_goal[gk_idx] > min_goalkeeper_and_goal_dist:
            continue

        others_idx = np.ones(len(bboxes_masks), bool)
        others_idx[gk_idx] = False

        if np.sum(others_idx) != 0:
            distances_to_goalkeeper = np.linalg.norm(positions[others_idx] - positions[gk_idx], axis=1)
            if np.min(distances_to_goalkeeper) < min_dist_to_goalkeeper:
                continue

        bb, mask_bb, mask_cnt, _ = bboxes_masks[gk_idx]
        if area(mask_bb) < min_player_bb_area:
            continue
        frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
        f = io.imread(frame_path)

        dark_regions_mask = calculate_dark_regions_mask(f)
        dark_contours = cv.findContours(dark_regions_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        is_dark = False
        if len(dark_contours) > 0:
            bb_centroid = ((bb[2] + bb[0]) // 2, (bb[3] + bb[1]) // 2)
            is_dark = in_dark_region(bb_centroid, dark_contours)

        hist = calculate_masked_patch_hist(f, mask_bb, mask_cnt, erosion_disk_radius, hist_type)
        labels[half][idx] = [Blob(bb, mask_bb, mask_cnt, hist, is_dark, category)]


def extract_goalkeepers(match_path, players_bboxes, sar=1, min_segmentation_score=0.65, min_goalkeeper_and_goal_dist=5,
                        min_dist_to_goalkeeper=2.5, min_player_bb_area=1000, erosion_disk_radius=2, hist_type='rgb'):
    labels = {0: {}, 1: {}}
    for half in range(2):
        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        for i, goalkeeper in enumerate([Players.GOALKEEPER_1, Players.GOALKEEPER_2]):
            goal_center = GOAL_CENTERS[(half + i) % 2, :]
            extract_goalkeeper(match_path, half, semantic_seg, sar, goal_center, goalkeeper.value, players_bboxes,
                               labels, min_segmentation_score, min_goalkeeper_and_goal_dist, min_dist_to_goalkeeper,
                               min_player_bb_area, erosion_disk_radius, hist_type)
    return labels


def calculate_hist_medians(labels, categories=None):
    if categories is None:
        categories = [0, 1, 2, 3, 4]

    hists = {c: [] for c in categories}
    for half, bboxes_by_idx in labels.items():
        for idx, blobs in bboxes_by_idx.items():
            for b in blobs:
                if b.category in categories:
                    hists[b.category].append(b.hist)
    hists = {i: np.squeeze(np.asarray(h, dtype=np.float32)) for i, h in hists.items()}
    medians = [np.median(hists[c], axis=0) for c in categories]
    return medians if len(categories) > 1 else medians[0]


def count_instances(labels, categories=None):
    if categories is None:
        categories = [0, 1, 2, 3, 4]

    count = {c: 0 for c in categories}
    for half, blobs_by_frame_idx in labels.items():
        for _, blobs in blobs_by_frame_idx.items():
            for b in blobs:
                if b.category in categories:
                    count[b.category] += 1
    return count


def remove_referees_from_goalkeepers(labels):
    referee_median = calculate_hist_medians(labels, [0])

    filtered_labels = {0: {}, 1: {}}
    for half, blobs_by_frame_idx in labels.items():
        for frame_idx, blobs in blobs_by_frame_idx.items():
            filtered_blobs = []
            for b in blobs:
                if b.category not in (3, 4):
                    filtered_blobs.append(b)
                    continue

                if bhattacharyya_distance(referee_median, b.hist.T) < args.color_hist_threshold:
                    continue
                filtered_blobs.append(b)

            if len(filtered_blobs) > 0:
                filtered_labels[half][frame_idx] = filtered_blobs
    return filtered_labels


def filter_class_instances(labels, color_hist_threshold=0.4):
    medians = calculate_hist_medians(labels)

    filtered_labels = {0: {}, 1: {}}
    for half, blobs_by_frame_idx in labels.items():
        for frame_idx, blobs in blobs_by_frame_idx.items():
            filtered_blobs = []
            for b in blobs:
                if bhattacharyya_distance(medians[b.category], b.hist.T) > color_hist_threshold:
                    continue
                filtered_blobs.append(b)

            if len(filtered_blobs) > 0:
                filtered_labels[half][frame_idx] = filtered_blobs
    return filtered_labels


def extract_preliminary_labels(match_path, players_bboxes, sar, min_segmentation_score,
                               min_dist_to_goals, min_player_bb_area, erosion_disk_radius, hist_type,
                               min_goalkeeper_and_goal_dist, min_dist_to_goalkeeper, color_hist_threshold, bench_dist,
                               save=True):
    midfielders_blobs = extract_midfielders_blobs(match_path,
                                                  players_bboxes,
                                                  sar,
                                                  min_segmentation_score,
                                                  min_dist_to_goals,
                                                  min_player_bb_area,
                                                  erosion_disk_radius,
                                                  hist_type,
                                                  bench_dist)
    midfielders_blobs = filter_bboxes_by_size(midfielders_blobs)

    midfielders = label_midfielders(midfielders_blobs)

    goalkeepers = extract_goalkeepers(match_path,
                                      players_bboxes,
                                      sar,
                                      min_segmentation_score,
                                      min_goalkeeper_and_goal_dist,
                                      min_dist_to_goalkeeper,
                                      min_player_bb_area,
                                      erosion_disk_radius,
                                      hist_type)

    labels = {h: midfielders[h] | goalkeepers[h] for h in range(2)}
    filtered_labels = filter_class_instances(labels, color_hist_threshold)
    
    goalkeepers_count = count_instances(filtered_labels, [3, 4])
    if goalkeepers_count[3] == 0 or goalkeepers_count[4] == 0:
        filtered_labels = remove_referees_from_goalkeepers(labels)
        filtered_labels = filter_class_instances(filtered_labels, args.color_hist_threshold)

    labels = filtered_labels

    if save:
        labels_path = match_path.joinpath('player_labeling', 'labels.pkl')
        with labels_path.open(mode='wb') as fid:
            pickle.dump(labels, fid)
    return labels


def count_bboxes(labels: dict):
    num_bboxes = defaultdict(lambda: 0)
    for _, blobs_by_frame in labels.items():
        for _, blobs in blobs_by_frame.items():
            for b in blobs:
                num_bboxes[b.category] += 1
    return num_bboxes


def create_training_splits(match_path, labels, validation_proportion=0.1):
    num_bboxes = count_bboxes(labels)
    is_train = dict()
    for label, count in num_bboxes.items():
        is_train[label] = np.zeros(count, dtype=int)

        num_validation_samples = np.round(validation_proportion * count).astype(int)
        if count > 1:
            num_validation_samples = max(num_validation_samples, 1)
        indices = np.random.choice(count, num_validation_samples, replace=False)
        is_train[label][indices] = 1

    splits_dir = match_path.joinpath('player_labeling', 'data')
    train_dir = splits_dir.joinpath('train')
    valid_dir = splits_dir.joinpath('valid')

    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    split_dirs = [train_dir, valid_dir]
    category_dirs = {i: {} for i in range(2)}
    for i, j in product(range(2), range(5)):
        category_dir = split_dirs[i].joinpath(f'{j}')
        category_dir.mkdir(parents=True, exist_ok=True)
        category_dirs[i][j] = category_dir
    img_path_count = {i: {j: 0 for j in range(5)} for i in range(2)}
    count = {i: 0 for i in range(5)}

    for half in range(2):
        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')
        for frame_idx in sorted(labels[half].keys()):
            labeled_blobs = labels[half][frame_idx]

            frame_path = frames_dir.joinpath(f'{frame_idx + 1:05}.jpg')
            f = io.imread(frame_path)
            for b in labeled_blobs:
                masked_patch = get_masked_patch(f, b.mask_bb, b.mask_cnt)
                split_idx = is_train[b.category][count[b.category]]

                img_fname = f'{img_path_count[split_idx][b.category] + 1:05}.jpg'
                img_fpath = category_dirs[split_idx][b.category].joinpath(img_fname)
                io.imsave(img_fpath, masked_patch)

                img_path_count[split_idx][b.category] += 1
                count[b.category] += 1


def main(args):
    np.random.seed(args.seed)

    if args.matches:
        with args.matches.open() as f:
            match_paths = [Path(line) for line in f.read().splitlines()]
    else:
        match_paths = [args.single_match]

    sampling_aspect_ratios = load_sampling_aspect_ratios(args.dataset_path)

    for match_path in tqdm(match_paths, desc='Overall Progress', leave=True, position=0):

        player_labeling_dir = match_path.joinpath('player_labeling')
        player_labeling_dir.mkdir(parents=True, exist_ok=True)

        labels_path = player_labeling_dir.joinpath('labels.pkl')

        # try:
        players_bboxes = filter_small_players(match_path,
                                              args.min_calibration_confidence,
                                              args.max_player_area)
        if not labels_path.exists():
            labels = extract_preliminary_labels(match_path,
                                                players_bboxes,
                                                sampling_aspect_ratios[match_path],
                                                args.min_segmentation_score,
                                                args.min_dist_to_goals,
                                                args.min_player_bb_area,
                                                args.erosion_disk_radius,
                                                args.hist_type,
                                                args.min_goalkeeper_and_goal_dist,
                                                args.min_dist_to_goalkeeper,
                                                args.color_hist_threshold,
                                                args.bench_dist)
        else:
            logging.info(f'Labels.pkl exists for match {match_path}')
            with labels_path.open(mode='rb') as fid:
                labels = pickle.load(fid)

        create_training_splits(match_path, labels, args.validation_proportion)
        # except Exception as e:
        #     logging.info(f'An exception occurred {e} for match {match_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Preprocessing for Player Classification')
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
    parser.add_argument('--min_calibration_confidence',
                        help='Minimum calibration confidence for filtering players (default: 0.85)',
                        default=0.75, type=float)
    parser.add_argument('--max_player_area',
                        help='Maximum player area for filtering players (default: 40,000)',
                        default=40000, type=float)
    parser.add_argument('--min_segmentation_score',
                        help='Minimum segmentation score for filtering players (default: 0.65)',
                        default=0.65, type=float)
    parser.add_argument('--min_gmm_prob_density',
                        help='Minimum GMM probability density for filtering midfield players (default: 1.0)',
                        default=1., type=float)
    parser.add_argument('--min_dist_to_goals',
                        help='Minimum distance to goals for all midfielders [1-32] (default: 20)',
                        default=20., type=float)
    parser.add_argument('--min_player_bb_area',
                        help="Minimum bounding box area of detected player to consider (default: 1000)",
                        default=1000, type=int)
    parser.add_argument('--erosion_disk_radius',
                        help="Erosion disk radius for player's masks (default: 2)",
                        default=2, type=int)
    parser.add_argument('--hist_type', choices=['rgb', 'hsv', 'hs', 'lab', 'ab'],
                        help='Histogram type for midfielders clustering  (default: rgb)',
                        default='rgb', type=str)
    parser.add_argument('--min_goalkeeper_and_goal_dist',
                        help='Min distance between goalkeeper and goal [0-64] (default: 5)',
                        default=5, type=float)
    parser.add_argument('--min_dist_to_goalkeeper',
                        help='Min distance between goalkeeper and other players [0-64] (default: 2.5)',
                        default=2.5, type=float)
    parser.add_argument('--color_hist_threshold',
                        help='Color histogram threshold between an instance and the class median [0.0-1.0] (default: '
                             '0.4)',
                        default=0.4, type=float)
    parser.add_argument('--validation_proportion',
                        help='Validation proportion for the training/validation splits [0.0-1.0] (default: 0.1)',
                        default=0.1, type=float)
    parser.add_argument('--seed', help='random seed (default: 42)',
                        default=42, type=int)
    parser.add_argument('--bench_dist', help='Distance to the bench  (default: 29)',
                        default=29, type=int)
    parser.add_argument('--logs_dir', required=False,
                        help='Path for logging directory (default: velocity_logs)',
                        default="velocity_logs", type=lambda p: Path(p))
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    load_log_configuration(args.log_config, args.logs_dir)
    main(args)
