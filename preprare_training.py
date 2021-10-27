import pickle
from itertools import product
from pathlib import Path

import numpy as np
from pixellib.torchbackend.instance import instanceSegmentation
from skimage import io
from sklearn.cluster import KMeans
import argparse
from IO import Players
from IO import load_bboxes
from IO import load_calibration
from field_calibration import GOAL_CENTERS
from field_calibration import calculate_dist_from_goals
from field_calibration import calculate_radar_position
from field_calibration import load_homography
from regions import area
from regions import calculate_patch_hist
from regions import get_patch
from regions import iou
from segmentation import segment_video


def load_segmentation_results(match_path, half, num_frames, pointrend_weights):
    segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}.npy')
    if not segmentation_results_fpath.exists():
        # Loading PointRend
        point_rend = instanceSegmentation()
        point_rend.load_model(pointrend_weights)
        point_rend.predictor.model.cuda()

        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')
        semantic_seg = segment_video(point_rend, frames_dir, num_frames, 10)
        np.save(segmentation_results_fpath, semantic_seg)
    else:
        semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)
    return semantic_seg


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


def extract_midfielders_blobs(match_path, player_bboxes, pointrend_weights, min_dist_to_goals=20, hist_type='rgb'):
    PERSON_ID = 0
    players_blobs = {0: {}, 1: {}}
    for half in range(2):
        half_match_detections = load_bboxes(match_path, half)
        half_match_calibration = load_calibration(match_path, half)
        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')

        num_frames = len(half_match_detections)
        semantic_seg = load_segmentation_results(match_path, half, num_frames, pointrend_weights)

        for idx, bboxes in player_bboxes[half].items():

            homography = load_homography(half_match_calibration[idx][0]["homography"])
            distances = [calculate_dist_from_goals(bb, homography) for bb in bboxes]
            if any(d < min_dist_to_goals for d in distances):
                continue

            ss_frame = [(bb, mask) for bb, mask, class_id in zip(semantic_seg[idx]['boxes'],
                                                                 semantic_seg[idx]['masks'],
                                                                 semantic_seg[idx]['class_ids']) if
                        class_id == PERSON_ID]

            num_bboxes, num_objects = len(bboxes), len(ss_frame)
            iou_scores = np.asarray(
                [[iou(ss_frame[j][0], bboxes[i]) for j in range(num_objects)] for i in range(num_bboxes)])

            if num_bboxes <= num_objects:
                correspondence = np.argmax(iou_scores, axis=1).tolist()
                bboxes_masks = [(bboxes[i], ss_frame[c][0], ss_frame[c][1]) for i, c in enumerate(correspondence)]
            else:
                correspondence = np.argmax(iou_scores, axis=0).tolist()
                bboxes_masks = [(bboxes[c], ss_frame[i][0], ss_frame[i][1]) for i, c in enumerate(correspondence)]

            players_blobs[half][idx] = list()
            frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
            f = io.imread(frame_path)
            for bb, mask_bb, mask in bboxes_masks:
                patch = get_patch(f, mask_bb, False)
                mask = 255 * mask.astype(np.uint8)
                hist = calculate_patch_hist(patch, mask, hist_type)
                players_blobs[half][idx].append((bb, hist))
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
            new_labels[half][idx] = [(bb, correct_labels[l].value) for bb, l in labeled_bboxes]
    return new_labels


def label_midfielders(players_blobs_by_half, kmeans):
    labels = dict()
    labels_count = [0, 0, 0]
    for half, players_blobs_ in players_blobs_by_half.items():
        labels[half] = dict()
        for idx, bboxes_hists in players_blobs_.items():
            labels[half][idx] = list()
            for bb, hist in bboxes_hists:
                c = int(kmeans.predict(hist)[0])
                labels[half][idx].append((bb, c))
                labels_count[c] += 1
    return fix_labels(labels, labels_count)


def extract_goalkeeper(match_path, half, goal_center, category, player_bboxes, labels,
                       min_dist_to_goal=5, min_dist_to_goalkeeper=2.5):
    half_match_calibration = load_calibration(match_path, half)

    for idx, bboxes in player_bboxes[half].items():

        homography = load_homography(half_match_calibration[idx][0]["homography"])
        positions = np.asarray([calculate_radar_position(bb, homography) for bb in bboxes])

        distances_to_goal = np.linalg.norm(positions - goal_center, axis=1)

        gk_idx = np.argmin(distances_to_goal)
        if distances_to_goal[gk_idx] > min_dist_to_goal:
            continue

        others_idx = np.ones(len(bboxes), bool)
        others_idx[gk_idx] = False

        if np.sum(others_idx) == 0:
            labels[half][idx] = [(bboxes[gk_idx], category)]
            continue

        distances_to_goalkeeper = np.linalg.norm(positions[others_idx] - positions[gk_idx], axis=1)
        if np.min(distances_to_goalkeeper) < min_dist_to_goalkeeper:
            continue
        labels[half][idx] = [(bboxes[gk_idx], category)]


def extract_goalkeepers(match_path, player_bboxes, min_dist_to_goal=5, min_dist_to_goalkeeper=2.5):
    labels = {0: {}, 1: {}}
    for half in range(2):
        for i, goalkeeper in enumerate([Players.GOALKEEPER_1, Players.GOALKEEPER_2]):
            goal_center = GOAL_CENTERS[(half + i) % 2, :]
            extract_goalkeeper(match_path, half, goal_center, goalkeeper.value, player_bboxes, labels,
                               min_dist_to_goal, min_dist_to_goalkeeper)
    return labels


def extract_preliminary_labels(match_path, player_bboxes, pointrend_weights, min_dist_to_goals, hist_type,
                               min_dist_to_goal, min_dist_to_goalkeeper, save=True):
    midfielders_blobs = extract_midfielders_blobs(match_path, player_bboxes, pointrend_weights, min_dist_to_goals,
                                                  hist_type)

    hists = [hist for h in range(2) for _, frame in midfielders_blobs[h].items() for _, hist in frame]
    hists = np.squeeze(np.asarray(hists, dtype=np.float32))
    kmeans = KMeans(n_clusters=3, random_state=0).fit(hists)

    midfielders = label_midfielders(midfielders_blobs, kmeans)
    goalkeepers = extract_goalkeepers(match_path, player_bboxes, min_dist_to_goal, min_dist_to_goalkeeper)
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
            for bb, l in labeled_bboxes:
                patch = get_patch(f, bb, copy=False)
                split_idx = is_train[i]
                img_fpath = category_dirs[split_idx][l].joinpath(f'{count[is_train[split_idx]][l] + 1:05}.jpg')
                io.imsave(img_fpath, patch)
                count[split_idx][l] += 1
                i += 1


def create_training_patches(match_path, players_bboxes):
    data_dir, i = Path('data/unlabeled'), 0
    data_dir.mkdir(parents=True, exist_ok=True)
    for half in range(2):
        frames_dir = match_path.joinpath(f'{half + 1}_HQ/frames')
        for idx, bboxes in players_bboxes[half].items():
            frame_path = frames_dir.joinpath(f'{idx + 1:05}.jpg')
            f = io.imread(frame_path)

            for bb in bboxes:
                patch = get_patch(f, bb, copy=False)
                img_fpath = data_dir.joinpath(f'{i + 1:06}.jpg')
                io.imsave(img_fpath, patch)
                i += 1


def main(args):
    labels_path = args.match_path.joinpath('labels.pkl')
    np.random.seed(args.seed)

    players_bboxes = filter_small_players(args.match_path, args.min_calibration_confidence, args.max_player_area)
    if not labels_path.exists():
        labels = extract_preliminary_labels(args.match_path,
                                            players_bboxes,
                                            args.pointrend_path,
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
    parser.add_argument('-p', '--pointrend_path', required=False,
                        help='Path for PointRend weights (default: weights/pointrend_resnet50.pkl)',
                        default='weights/pointrend_resnet50.pkl', type=lambda p: Path(p))
    parser.add_argument('--min_calibration_confidence',
                        help='Minimum calibration confidence for filtering players (default: 0.85)',
                        default=0.85, type=float)
    parser.add_argument('--max_player_area',
                        help='Maximum player area for filtering players (default: 40,000)',
                        default=40000, type=float)
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
