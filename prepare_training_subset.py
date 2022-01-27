import argparse
import pickle
import shutil
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
from field_calibration import GOAL_CENTERS
from field_calibration import calculate_dist_from_goals
from field_calibration import calculate_radar_position
from field_calibration import load_homography
from regions import area
from regions import calculate_patch_hist
from regions import get_masked_patch
from regions import get_patch
from regions import iou
from regions import to_mask


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


def match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx, min_segmentation_score=0.65):
    PERSON_ID = 0
    ss_frame = [(bb, mask_cnts, score) for bb, mask_cnts, class_id, score in zip(semantic_seg[idx]['boxes'],
                                                                                 semantic_seg[idx]['masks'],
                                                                                 semantic_seg[idx]['class_ids'],
                                                                                 semantic_seg[idx]['scores']) if
                class_id == PERSON_ID and score > min_segmentation_score]

    num_bboxes, num_objects = len(bboxes), len(ss_frame)
    iou_scores = np.asarray(
        [[iou(ss_frame[j][0], bboxes[i]) for j in range(num_objects)] for i in range(num_bboxes)])

    if num_bboxes <= num_objects:
        correspondence = np.argmax(iou_scores, axis=1).tolist()
        return [(bboxes[i], ss_frame[c][0], ss_frame[c][1], ss_frame[c][2]) for i, c in enumerate(correspondence)]
    else:
        correspondence = np.argmax(iou_scores, axis=0).tolist()
        return [(bboxes[c], ss_frame[i][0], ss_frame[i][1], ss_frame[i][2]) for i, c in enumerate(correspondence)]


def calculate_masked_patch_hist(frame, mask_bb, mask_cnt, erosion_disk_radius=2, hist_type='rgb'):
    patch = get_patch(frame, mask_bb, False)
    mask = to_mask(mask_bb, mask_cnt)
    if erosion_disk_radius > 0:
        eroded_mask = np.zeros(mask.shape, dtype=np.uint8)
        binary_erosion(mask, disk(erosion_disk_radius), out=eroded_mask)
        mask = eroded_mask
    return calculate_patch_hist(patch, mask, hist_type)


def extract_midfielders_blobs(match_path, player_bboxes, min_segmentation_score=0.65, min_dist_to_goals=20,
                              min_player_bb_area=1000, erosion_disk_radius=2, hist_type='rgb'):
    players_blobs = {0: {}, 1: {}}
    for half in range(2):
        half_match_calibration = load_calibration(match_path, half)
        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')

        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        for idx, bboxes in player_bboxes[half].items():

            homography = load_homography(half_match_calibration[idx][0]["homography"])
            distances = [calculate_dist_from_goals(bb, homography) for bb in bboxes]
            if any(d < min_dist_to_goals for d in distances):
                continue

            bboxes_masks = match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx, min_segmentation_score)

            players_blobs[half][idx] = list()
            frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
            f = io.imread(frame_path)
            for bb, mask_bb, mask_cnt, _ in bboxes_masks:
                if area(mask_bb) < min_player_bb_area:
                    continue
                hist = calculate_masked_patch_hist(f, mask_bb, mask_cnt, erosion_disk_radius, hist_type)
                players_blobs[half][idx].append((bb, mask_bb, mask_cnt, hist))
    return players_blobs


def reorder_labels(labels, labels_count):
    referee_class = int(np.argmin(labels_count))
    classes = [0, 1, 2]
    classes.remove(referee_class)

    correct_labels = {referee_class: Players.REFEREE,
                      classes[0]: Players.TEAM_A,
                      classes[1]: Players.TEAM_B}

    new_labels = {0: {}, 1: {}}
    for half, labels_by_frame in labels.items():
        for idx, labeled_bboxes in labels_by_frame.items():
            new_labels[half][idx] = [(bb, mask_bb, mask_cnts, hist, correct_labels[l].value) for bb, mask_bb,
                                                                                                 mask_cnts, hist, l in
                                     labeled_bboxes]
    return new_labels


def label_midfielders(players_blobs_by_half, min_prob_density=1.):
    hists = [h[-1] for half in range(2) for _, frame in players_blobs_by_half[half].items() for h in frame]
    hists = np.squeeze(np.asarray(hists, dtype=np.float32))
    gmm = GaussianMixture(n_components=3, random_state=0).fit(hists)

    labels = dict()
    labels_count = [0, 0, 0]
    for half, players_blobs_ in players_blobs_by_half.items():
        labels[half] = dict()
        for idx, bboxes_hists in players_blobs_.items():
            labels[half][idx] = list()
            for bb, mask_bb, mask_cnts, hist in bboxes_hists:
                p = gmm.predict_proba(hist)
                if min_prob_density is not None and p.max() < min_prob_density:
                    continue
                c = np.argmax(p)
                labels[half][idx].append((bb, mask_bb, mask_cnts, hist, c))
                labels_count[c] += 1
    return reorder_labels(labels, labels_count)


def extract_goalkeeper(match_path, half, semantic_seg, goal_center, category, player_bboxes, labels,
                       min_segmentation_score=0.65, min_goalkeeper_and_goal_dist=5, min_dist_to_goalkeeper=2.5,
                       min_player_bb_area=1000, erosion_disk_radius=2, hist_type='rgb'):
    half_match_calibration = load_calibration(match_path, half)
    frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')

    for idx, bboxes in player_bboxes[half].items():

        homography = load_homography(half_match_calibration[idx][0]["homography"])

        bboxes_masks = match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx, min_segmentation_score)
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
        hist = calculate_masked_patch_hist(f, mask_bb, mask_cnt, erosion_disk_radius, hist_type)
        labels[half][idx] = [(bb, mask_bb, mask_cnt, hist, category)]


def extract_goalkeepers(match_path, player_bboxes, min_segmentation_score=0.65, min_goalkeeper_and_goal_dist=5,
                        min_dist_to_goalkeeper=2.5, min_player_bb_area=1000, erosion_disk_radius=2, hist_type='rgb'):
    labels = {0: {}, 1: {}}
    for half in range(2):
        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        for i, goalkeeper in enumerate([Players.GOALKEEPER_1, Players.GOALKEEPER_2]):
            goal_center = GOAL_CENTERS[(half + i) % 2, :]
            extract_goalkeeper(match_path, half, semantic_seg, goal_center, goalkeeper.value, player_bboxes,
                               labels, min_segmentation_score, min_goalkeeper_and_goal_dist, min_dist_to_goalkeeper,
                               min_player_bb_area, erosion_disk_radius, hist_type)
    return labels


def filter_class_instances(labels, color_hist_threshold=0.4):
    hists = [[], [], [], [], []]
    for half, bboxes_by_idx in labels.items():
        for idx, bboxes in bboxes_by_idx.items():
            for bb, mask_bb, mask_cnt, hist, c in bboxes:
                hists[c].append(hist)
    hists = [np.squeeze(np.asarray(h, dtype=np.float32)) for h in hists]
    medians = [np.median(h, axis=0) for h in hists]

    filtered_labels = {0: {}, 1: {}}
    for half, bboxes_by_idx in labels.items():
        for idx, bboxes in bboxes_by_idx.items():
            filtered_bboxes = []
            for bb, mask_bb, mask_cnt, hist, c in bboxes:
                if cv.compareHist(medians[c], hist.T, cv.HISTCMP_BHATTACHARYYA) > color_hist_threshold:
                    continue
                filtered_bboxes.append((bb, mask_bb, mask_cnt, c))

            if len(filtered_bboxes) > 0:
                filtered_labels[half][idx] = filtered_bboxes
    return filtered_labels


def extract_preliminary_labels(match_path, player_bboxes, min_segmentation_score, min_gmm_prob_density,
                               min_dist_to_goals, min_player_bb_area, erosion_disk_radius, hist_type,
                               min_goalkeeper_and_goal_dist, min_dist_to_goalkeeper, color_hist_threshold, save=True):
    midfielders_blobs = extract_midfielders_blobs(match_path,
                                                  player_bboxes,
                                                  min_segmentation_score,
                                                  min_dist_to_goals,
                                                  min_player_bb_area,
                                                  erosion_disk_radius,
                                                  hist_type)

    midfielders = label_midfielders(midfielders_blobs, min_gmm_prob_density)

    goalkeepers = extract_goalkeepers(match_path,
                                      player_bboxes,
                                      min_segmentation_score,
                                      min_goalkeeper_and_goal_dist,
                                      min_dist_to_goalkeeper,
                                      min_player_bb_area,
                                      erosion_disk_radius,
                                      hist_type)

    labels = {h: midfielders[h] | goalkeepers[h] for h in range(2)}
    labels = filter_class_instances(labels, color_hist_threshold)

    if save:
        labels_path = match_path.joinpath('player_labeling', 'labels.pkl')
        with labels_path.open(mode='wb') as fid:
            pickle.dump(labels, fid)
    return labels


def count_bboxes(labels: dict):
    num_bboxes = 0
    for _, labels_by_frame in labels.items():
        for _, labeled_bboxes in labels_by_frame.items():
            num_bboxes += len(labeled_bboxes)
    return num_bboxes


def create_training_splits(match_path, labels, validation_proportion=0.1):
    num_bboxes = count_bboxes(labels)
    is_train = np.random.binomial(1, validation_proportion, num_bboxes).astype(int)

    splits_dir = match_path.joinpath('player_labeling', 'data')
    train_dir = splits_dir.joinpath('train')
    valid_dir = splits_dir.joinpath('valid')

    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    split_dirs = [train_dir, valid_dir]
    category_dirs = {i: {} for i in range(2)}
    count = {i: {j: 0 for j in range(5)} for i in range(2)}
    for i, j in product(range(2), range(5)):
        category_dir = split_dirs[i].joinpath(f'{j}')
        category_dir.mkdir(parents=True, exist_ok=True)
        category_dirs[i][j] = category_dir

    i = 0
    for half in range(2):
        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')
        for idx in sorted(labels[half].keys()):
            labeled_bboxes = labels[half][idx]

            frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
            f = io.imread(frame_path)
            for _, mask_bb, mask_cnt, l in labeled_bboxes:
                masked_patch = get_masked_patch(f, mask_bb, mask_cnt)
                split_idx = is_train[i]
                img_fpath = category_dirs[split_idx][l].joinpath(f'{count[is_train[split_idx]][l] + 1:05}.jpg')
                io.imsave(img_fpath, masked_patch)
                count[split_idx][l] += 1
                i += 1


def main(args):
    np.random.seed(args.seed)

    if args.matches:
        with args.matches.open() as f:
            match_paths = [Path(line) for line in f.read().splitlines()]
    else:
        match_paths = [args.single_match]

    for match_path in tqdm(match_paths, desc='Overall Progress', leave=True, position=0):
        labels_path = match_path.joinpath('player_labeling', 'labels.pkl')

        players_bboxes = filter_small_players(match_path,
                                              args.min_calibration_confidence,
                                              args.max_player_area)
        if not labels_path.exists():
            labels = extract_preliminary_labels(match_path,
                                                players_bboxes,
                                                args.min_segmentation_score,
                                                args.min_gmm_prob_density,
                                                args.min_dist_to_goals,
                                                args.min_player_bb_area,
                                                args.erosion_disk_radius,
                                                args.hist_type,
                                                args.min_goalkeeper_and_goal_dist,
                                                args.min_dist_to_goalkeeper,
                                                args.color_hist_threshold)
        else:
            with labels_path.open(mode='rb') as fid:
                labels = pickle.load(fid)

        create_training_splits(match_path, labels, args.validation_proportion)


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Preprocessing for Player Classification')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--single_match',
                       help='Directory path for the match to process (default: None)',
                       default=None, type=lambda p: Path(p))
    group.add_argument('-m', '--matches',
                       help='Path for a file containing a matches list to process',
                       default=None, type=lambda p: Path(p))
    parser.add_argument('--min_calibration_confidence',
                        help='Minimum calibration confidence for filtering players (default: 0.85)',
                        default=0.85, type=float)
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
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
