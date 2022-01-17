import json
import logging
import time
from argparse import ArgumentParser
from pathlib import Path
import warnings
import cv2
import numpy as np
from addict import Dict
from skimage import io
from skimage.filters import gaussian
from skimage.measure import label
from skimage.measure import regionprops
from skimage.metrics import mean_squared_error
from skimage.transform import resize
from skimage.util import img_as_bool
from skimage.util import img_as_ubyte
from tqdm import tqdm

from IO import load_bboxes
from IO import load_flow
from IO import load_num_optical_flow_frames
from optical_flow import add_missing_borders
from optical_flow import build_second_moment_matrix
from optical_flow import caculate_dominant_orientation_vector
from optical_flow import caculate_inertia_matrix_components
from optical_flow import flow_sizes_constants
from optical_flow import resize_to_flow_shape_and_remove_borders
from regions import cnts_to_indices
from regions import draw_mask
from regions import get_patch
from regions import scale_mask
from segmentation import to_polygons
from util import images_size
from util import FaissKMeans
from util import load_log_configuration


def calculate_fixed_regions_mask(half_match_path, optical_flow_indices, num_rgb_frames, num_sampling_frames=540,
                                 num_clusters=4, gpu_devices=None):
    def load_abs_flow(idx):
        return np.abs(load_flow(half_match_path, optical_flow_indices[idx])) / num_sampling_frames

    sampling_indices = np.ceil(np.linspace(0, num_rgb_frames - 1, num=num_sampling_frames)).astype(int)
    mean_flow = load_abs_flow(sampling_indices[0])
    for idx in sampling_indices[1:]:
        mean_flow += load_abs_flow(idx)
    mean_flow = gaussian(mean_flow, 3, multichannel=True, mode='reflect')

    kmeans = FaissKMeans(num_clusters, 100, gpu_devices)
    kmeans.fit(mean_flow.reshape((-1, 2)))
    labels = kmeans.predict(mean_flow.reshape((-1, 2)))

    mean_squared_error_ = np.sum(np.power(kmeans.centroids, 2), axis=1) / 2
    static_centroids_idx = [idx for idx, mse in enumerate(mean_squared_error_) if mse < 0.01]

    centers = np.zeros(num_clusters, dtype=np.uint8)
    centers[static_centroids_idx] = 1

    mask = centers[labels.flatten()]
    mask = mask.reshape(mean_flow.shape[:2])

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    dilated_mask = cv2.dilate(mask, disk, iterations=1)
    return 255 * dilated_mask


def segment_fixed_regions(half_match_path, optical_flow_indices, num_rgb_frames, flow_sizes, num_sampling_frames=540,
                          num_clusters=4, gpu_devices=None):
    mask = calculate_fixed_regions_mask(half_match_path, optical_flow_indices, num_rgb_frames, num_sampling_frames,
                                        num_clusters, gpu_devices)
    mask = add_missing_borders(mask, flow_sizes)
    sampling_indices = np.ceil(np.linspace(0, num_rgb_frames - 1, num=num_sampling_frames)).astype(int)

    idx = sampling_indices[0]
    frames_path = half_match_path.joinpath('frames')
    rgb_frame_path = frames_path.joinpath(f'{idx + 1:05}.jpg')
    rgb = io.imread(rgb_frame_path)

    mask = resize(mask, rgb.shape[:2], anti_aliasing=False, preserve_range=True)
    regions = regionprops(label(img_as_bool(mask)))

    patches = [np.zeros((r.image.shape[0], r.image.shape[1], 3), dtype=np.float64) for r in regions]
    for idx in sampling_indices:
        rgb_frame_path = frames_path.joinpath(f'{idx + 1:05}.jpg')
        rgb = io.imread(rgb_frame_path)

        for i, r in enumerate(regions):
            y1, x1, y2, x2 = r.bbox
            patches[i] += get_patch(rgb, (x1, y1, x2, y2), False) / len(sampling_indices)

    for i, r in enumerate(regions):
        r.mask = img_as_ubyte(r.image)
        patches[i] = cv2.bitwise_and(patches[i], patches[i], mask=r.mask)
        patches[i][patches[i] > 255] = 255
        r.patch = img_as_ubyte(patches[i] / 255)

    return regions


def field_segmentation(frame, segmented_people, fixed_regions, fixed_region_min_similarity=0.011, num_clusters=5,
                       max_iter=30, opening_disk_radius=8, min_area_proportion=0.15, gpu_devices=None):
    # Removing fixed regions from frame if found
    for r in fixed_regions:
        y1, x1, y2, x2 = r.bbox
        patch = get_patch(frame, (x1, y1, x2, y2), False)
        masked_patch = cv2.bitwise_and(patch, patch, mask=r.mask)

        similarity = mean_squared_error(masked_patch, r.patch) / 65025
        if similarity > fixed_region_min_similarity:
            continue
        patch[r.image, :] = 0

    black = np.asarray([0., 0., 0.])
    for bb, mask_cnts in segmented_people:
        draw_mask(frame, bb, mask_cnts, black)

    # Color quantization using K-means
    pixels = np.float32(frame.reshape((-1, 3)))

    num_pixels = pixels.shape[0]
    sampled_indices = np.linspace(0, num_pixels - 1, num=num_pixels // 2).astype(int)

    kmeans = FaissKMeans(num_clusters, max_iter, gpu_devices)
    kmeans.fit(pixels[sampled_indices, :])
    centers = kmeans.centroids
    labels = kmeans.predict(pixels)

    # Sorting clusters by number of pixels
    sorted_indices = np.argsort(np.histogram(labels[sampled_indices], bins=np.arange(num_clusters + 1))[0])[::-1]
    centers = np.uint8(centers[sorted_indices])

    # The largest cluster is assumed to be the field
    field_indices = [0]
    field_color = centers[0]
    for idx in range(1, num_clusters):
        similarity = mean_squared_error(field_color, centers[idx]) / 65025
        if similarity < 0.01:
            field_color = (field_color + centers[idx]) / 2
            field_indices.append(idx)

    centers = np.zeros(num_clusters, dtype=np.uint8)
    centers[sorted_indices[field_indices]] = 1

    mask = centers[labels.flatten()]
    mask = mask.reshape(frame.shape[:2])

    # Removing small regions and connecting big ones
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_disk_radius, opening_disk_radius))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk)

    # Filtering regions by area
    min_field_area = np.power(min_area_proportion * np.max(opened_mask.shape[:2]), 2)

    filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
    contours = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        if cv2.contourArea(cnt) < min_field_area:
            continue
        hull = cv2.convexHull(cnt)
        cv2.drawContours(filtered_mask, [hull], 0, 255, -1)

    mask = cv2.bitwise_and(opened_mask, opened_mask, mask=filtered_mask)
    return mask


def velocity_vectors_from_frame(rgb, flow, rgb_shape, flow_sizes, segmented_people, fixed_regions,
                                field_segmentation_args, gpu_devices=None):
    components = caculate_inertia_matrix_components(flow)
    field_mask = field_segmentation(rgb, segmented_people, fixed_regions,
                                    field_segmentation_args.fixed_region_min_similarity,
                                    field_segmentation_args.clustering.num_clusters,
                                    field_segmentation_args.clustering.max_iter,
                                    field_segmentation_args.opening_disk_radius,
                                    field_segmentation_args.min_area_proportion,
                                    gpu_devices)
    field_cnt = to_polygons(field_mask)
    field_mask = img_as_bool(resize_to_flow_shape_and_remove_borders(field_mask, flow_sizes))
    flow_vectors = flow[field_mask, :]

    if len(flow_vectors) > 0:
        M = build_second_moment_matrix(components, mask=field_mask)
        field_vector = caculate_dominant_orientation_vector(M, flow_vectors)
    else:
        field_vector = np.zeros(2)

    scale = np.asarray([f / r for r, f in zip(rgb_shape[1::-1], flow.shape[1::-1])])
    velocity_vectors = []
    for bb, mask_cnts in segmented_people:
        scaled_bb, scaled_mask_cnts = scale_mask(bb, mask_cnts, scale)
        flow_patch = get_patch(flow, scaled_bb, copy=False)

        if len(scaled_mask_cnts) == 0:
            warnings.warn(f'Segmented person with bbox [{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}] has no contour')
            rr, cc = np.indices(flow_patch.shape[0:2])
            rr = rr.flatten()
            cc = cc.flatten()
        else:
            rr, cc = cnts_to_indices(scaled_mask_cnts)

        flow_vectors = flow_patch[rr, cc, :]

        components_patches = [get_patch(c, scaled_bb, copy=False) for c in components]
        M = build_second_moment_matrix(components_patches, indices=(rr, cc))

        player_vector = caculate_dominant_orientation_vector(M, flow_vectors)
        velocity_vectors.append(player_vector - field_vector)

    return field_cnt, field_vector, np.asarray(velocity_vectors)


def get_segmented_people(semantic_seg):
    return [(bb, mask_cnts) for bb, mask_cnts, class_id in zip(semantic_seg['boxes'],
                                                               semantic_seg['masks'],
                                                               semantic_seg['class_ids'])
            if class_id == 0]  # Person class ID is 0


def velocity_vectors_from_half_match(match_path, half, num_rgb_frames, num_optical_flow_frames, fixed_regions_args,
                                     field_segmentation_args, gpu_devices=None):

    segmentation_results_fpath = match_path.joinpath(f'segmentation_results_{half + 1}_HQ.npy')
    semantic_seg = np.load(segmentation_results_fpath, allow_pickle=True)

    optical_flow_indices = np.ceil(np.linspace(num_optical_flow_frames / num_rgb_frames,
                                               num_optical_flow_frames,
                                               num=num_rgb_frames)).astype(int) - 1

    half_match_path = match_path.joinpath(f'{half + 1}_HQ')
    frames_path = half_match_path.joinpath('frames')

    rgb_shape = images_size(frames_path)
    flow_sizes = flow_sizes_constants(rgb_shape)
    fixed_regions = segment_fixed_regions(half_match_path, optical_flow_indices, num_rgb_frames, flow_sizes,
                                          fixed_regions_args.num_sampling_frames, fixed_regions_args.num_clusters,
                                          gpu_devices)

    results = {}
    for idx in tqdm(range(num_rgb_frames), desc='Half-match progress', leave=True, position=0):
        segmented_people = get_segmented_people(semantic_seg[idx])
        if len(segmented_people) == 0:
            continue

        rgb_frame_path = frames_path.joinpath(f'{idx + 1:05}.jpg')
        rgb = io.imread(rgb_frame_path)

        flow = load_flow(half_match_path, optical_flow_indices[idx])
        flow = add_missing_borders(flow, flow_sizes)
        field_mask, field_vector, velocity = velocity_vectors_from_frame(rgb,
                                                                         flow,
                                                                         rgb_shape,
                                                                         flow_sizes,
                                                                         segmented_people,
                                                                         fixed_regions,
                                                                         field_segmentation_args,
                                                                         gpu_devices)
        results[idx] = {'field_mask': field_mask,
                        'field_vector': field_vector,
                        'velocity': velocity}
    return results


def main(match_path: Path, num_optical_flow_frames, fixed_regions_args, field_segmentation_args, gpu_devices=None):
    logging.info(f'Processing match {match_path}')
    results = {}
    for half in range(2):
        num_rgb_frames = len(load_bboxes(match_path, half))
        results[half] = velocity_vectors_from_half_match(match_path,
                                                         half,
                                                         num_rgb_frames,
                                                         num_optical_flow_frames[half],
                                                         fixed_regions_args,
                                                         field_segmentation_args,
                                                         gpu_devices)
    return results


if __name__ == '__main__':

    parser = ArgumentParser(description='Velocity extraction for Players')
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
    parser.add_argument('-c', '--conf', required=False,
                        help='JSON configuration filepath (default: config/velocity.json)',
                        default="config/velocity.json", type=lambda p: Path(p))
    parser.add_argument('-g', '--gpu_devices', nargs='+',
                        help='List of GPU devices to use for clustering (default: [])',
                        default=[], required=False, type=int)
    parser.add_argument('--logs_dir', required=False,
                        help='Path for logging directory (default: velocity_logs)',
                        default="velocity_logs", type=lambda p: Path(p))
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    args = parser.parse_args()

    load_log_configuration(args.log_config, args.logs_dir)

    with open(args.conf) as json_file:
        conf = json.load(json_file)
        conf = Dict(conf)

    if args.matches:
        with args.matches.open() as f:
            match_paths = [Path(line) for line in f.read().splitlines()]
    else:
        match_paths = [args.single_match]

    num_optical_flow_frames = load_num_optical_flow_frames(args.dataset_path)

    for match_path in tqdm(match_paths, desc='Overall Progress', leave=True, position=0):

        velocity_results_fpath = match_path.joinpath(f'velocity_results.npy')
        if velocity_results_fpath.exists():
            continue

        start = time.time()
        velocity = main(match_path,
                        num_optical_flow_frames[match_path],
                        conf.fixed_regions,
                        conf.field_segmentation,
                        args.gpu_devices)

        np.save(velocity_results_fpath, velocity)
        logging.info(f'Match processing time is {time.time() - start} seconds')
