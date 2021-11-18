import argparse
import pickle
from itertools import product
from pathlib import Path

import numpy as np
from skimage import io
from sklearn.mixture import GaussianMixture

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
    ss_frame = [(bb, mask_cnts) for bb, mask_cnts, class_id, score in zip(semantic_seg[idx]['boxes'],
                                                                          semantic_seg[idx]['masks'],
                                                                          semantic_seg[idx]['class_ids'],
                                                                          semantic_seg[idx]['scores']) if
                class_id == PERSON_ID and score > min_segmentation_score]

    num_bboxes, num_objects = len(bboxes), len(ss_frame)
    iou_scores = np.asarray(
        [[iou(ss_frame[j][0], bboxes[i]) for j in range(num_objects)] for i in range(num_bboxes)])

    if num_bboxes <= num_objects:
        correspondence = np.argmax(iou_scores, axis=1).tolist()
        return [(bboxes[i], ss_frame[c][0], ss_frame[c][1]) for i, c in enumerate(correspondence)]
    else:
        correspondence = np.argmax(iou_scores, axis=0).tolist()
        return [(bboxes[c], ss_frame[i][0], ss_frame[i][1]) for i, c in enumerate(correspondence)]


def extract_midfielders_blobs(match_path, player_bboxes, min_segmentation_score=0.65, min_dist_to_goals=20,
                              hist_type='rgb'):
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
            for bb, mask_bb, mask_cnts in bboxes_masks:
                patch = get_patch(f, mask_bb, False)
                mask = to_mask(mask_bb, mask_cnts)
                hist = calculate_patch_hist(patch, mask, hist_type)
                players_blobs[half][idx].append((bb, mask_bb, mask_cnts, hist))
    return players_blobs


def fix_labels(labels, labels_count):
    referee_class = int(np.argmin(labels_count))
    classes = [0, 1, 2]
    classes.remove(referee_class)

    correct_labels = {referee_class: Players.REFEREE,
                      classes[0]: Players.TEAM_A,
                      classes[1]: Players.TEAM_B}

    new_labels = {0: {}, 1: {}}
    for half, labels_by_frame in labels.items():
        for idx, labeled_bboxes in labels_by_frame.items():
            new_labels[half][idx] = [(bb, mask_bb, mask_cnt, correct_labels[l].value) for bb, mask_bb, mask_cnt, l in
                                     labeled_bboxes]
    return new_labels


def label_midfielders(players_blobs_by_half, gmm, min_prob_density=1.):
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
                labels[half][idx].append((bb, mask_bb, mask_cnts, c))
                labels_count[c] += 1
    return fix_labels(labels, labels_count)


def extract_goalkeeper(match_path, half, semantic_seg, goal_center, category, player_bboxes, labels,
                       min_segmentation_score=0.65, min_goalkeeper_and_goal_dist=5, min_dist_to_goalkeeper=2.5):
    half_match_calibration = load_calibration(match_path, half)

    for idx, bboxes in player_bboxes[half].items():

        homography = load_homography(half_match_calibration[idx][0]["homography"])

        bboxes_masks = match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx, min_segmentation_score)
        if len(bboxes_masks) == 0:
            continue

        positions = np.asarray([calculate_radar_position(bb, homography) for bb, _, _ in bboxes_masks])
        distances_to_goal = np.linalg.norm(positions - goal_center, axis=1)
        gk_idx = np.argmin(distances_to_goal)
        if distances_to_goal[gk_idx] > min_goalkeeper_and_goal_dist:
            continue
        
        others_idx = np.ones(len(bboxes_masks), bool)
        others_idx[gk_idx] = False

        if np.sum(others_idx) == 0:
            bb, mask_bb, mask_cnt = bboxes_masks[gk_idx]
            labels[half][idx] = [(bb, mask_bb, mask_cnt, category)]
            continue

        distances_to_goalkeeper = np.linalg.norm(positions[others_idx] - positions[gk_idx], axis=1)
        if np.min(distances_to_goalkeeper) < min_dist_to_goalkeeper:
            continue
        bb, mask_bb, mask_cnt = bboxes_masks[gk_idx]
        labels[half][idx] = [(bb, mask_bb, mask_cnt, category)]


def extract_goalkeepers(match_path, player_bboxes, min_segmentation_score=0.65, min_goalkeeper_and_goal_dist=5,
                        min_dist_to_goalkeeper=2.5):
    labels = {0: {}, 1: {}}
    for half in range(2):
        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        for i, goalkeeper in enumerate([Players.GOALKEEPER_1, Players.GOALKEEPER_2]):
            goal_center = GOAL_CENTERS[(half + i) % 2, :]
            extract_goalkeeper(match_path, half, semantic_seg, goal_center, goalkeeper.value, player_bboxes,
                               labels, min_segmentation_score, min_goalkeeper_and_goal_dist, min_dist_to_goalkeeper)
    return labels


def extract_preliminary_labels(match_path, player_bboxes, min_segmentation_score, min_gmm_prob_density,
                               min_dist_to_goals, hist_type, min_goalkeeper_and_goal_dist, min_dist_to_goalkeeper,
                               save=True):
    midfielders_blobs = extract_midfielders_blobs(match_path,
                                                  player_bboxes,
                                                  min_segmentation_score,
                                                  min_dist_to_goals,
                                                  hist_type)

    hists = [h[-1] for half in range(2) for _, frame in midfielders_blobs[half].items() for h in frame]
    hists = np.squeeze(np.asarray(hists, dtype=np.float32))
    gmm = GaussianMixture(n_components=3, random_state=0).fit(hists)

    midfielders = label_midfielders(midfielders_blobs, gmm, min_gmm_prob_density)
    goalkeepers = extract_goalkeepers(match_path,
                                      player_bboxes,
                                      min_segmentation_score,
                                      min_goalkeeper_and_goal_dist,
                                      min_dist_to_goalkeeper)
    labels = {h: midfielders[h] | goalkeepers[h] for h in range(2)}

    if save:
        labels_path = match_path.joinpath('labels.pkl')
        with labels_path.open(mode='wb') as fid:
            pickle.dump(labels, fid)
    return labels


def count_bboxes(labels: dict):
    num_bboxes = 0
    for _, labels_by_frame in labels.items():
        for _, labeled_bboxes in labels_by_frame.items():
            num_bboxes += len(labeled_bboxes)
    return num_bboxes


def create_init_training_splits(match_path, labels):
    num_bboxes = count_bboxes(labels)
    is_train = np.random.binomial(1, 0.1, num_bboxes).astype(int)

    train_dir = Path('data/train')
    valid_dir = Path('data/valid')

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


def create_training_patches(match_path, players_bboxes):
    data_dir, i = Path('data/unlabeled/0'), 0
    data_dir.mkdir(parents=True, exist_ok=True)
    for half in range(2):
        segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')
        for idx, bboxes in players_bboxes[half].items():
            frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
            f = io.imread(frame_path)

            bboxes_masks = match_semantic_segmentation_bboxes(bboxes, semantic_seg, idx)
            for bb, mask_bb, mask_cnt in bboxes_masks:
                masked_patch = get_masked_patch(f, mask_bb, mask_cnt)
                img_fpath = data_dir.joinpath(f'{i + 1:06}.jpg')
                io.imsave(img_fpath, masked_patch)
                i += 1


def main(args):
    labels_path = args.match_path.joinpath('labels.pkl')
    np.random.seed(args.seed)

    players_bboxes = filter_small_players(args.match_path,
                                          args.min_calibration_confidence,
                                          args.max_player_area)
    if not labels_path.exists():
        labels = extract_preliminary_labels(args.match_path,
                                            players_bboxes,
                                            args.min_segmentation_score,
                                            args.min_gmm_prob_density,
                                            args.min_dist_to_goals,
                                            args.hist_type,
                                            args.min_goalkeeper_and_goal_dist,
                                            args.min_dist_to_goalkeeper)
    else:
        with labels_path.open(mode='rb') as fid:
            labels = pickle.load(fid)

    create_init_training_splits(args.match_path, labels)
    create_training_patches(args.match_path, players_bboxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Preprocessing for Deep Clustering')

    parser.add_argument('-m', '--match_path', required=True,
                        help='Path for SoccerNet dataset (default: None)',
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
    parser.add_argument('--hist_type', choices=['rgb', 'hsv', 'hs', 'lab', 'ab'],
                        help='Histogram type for midfielders clustering  (default: rgb)',
                        default='rgb', type=str)
    parser.add_argument('--min_goalkeeper_and_goal_dist',
                        help='Min distance between goalkeeper and goal [0-64] (default: 5)',
                        default=5, type=float)
    parser.add_argument('--min_dist_to_goalkeeper',
                        help='Min distance between goalkeeper and other players [0-64] (default: 2.5)',
                        default=2.5, type=float)
    parser.add_argument('--seed', help='random seed (default: 42)',
                        default=42, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
