import logging
import time
from pathlib import Path
import warnings
import cv2
import numpy as np
from typing import Dict as Dict, List
from skimage import io
from skimage.filters import gaussian
from skimage.measure import label
from skimage.measure import regionprops
from skimage.metrics import mean_squared_error
from skimage.transform import resize
from skimage.util import img_as_bool
from skimage.util import img_as_ubyte
from tqdm import tqdm

from IO import load_num_optical_flow_frames
from optical_flow import add_missing_borders
from optical_flow import build_second_moment_matrix
from optical_flow import caculate_dominant_orientation_vector
from optical_flow import caculate_inertia_matrix_components
from optical_flow import flow_sizes_constants
from optical_flow import resize_to_flow_shape_and_remove_borders
from regions import cnts_to_indices
from regions import scale_mask
from segmentation import to_polygons
from util import Shape, images_size
from util import FaissKMeans
from util import load_log_configuration

import hydra
from omegaconf import DictConfig
from kitman.regions import get_patch
from kitman.visualizations import draw_mask

from kitman.segmentation import SegmentedPlayer, POINTREND_PERSON_ID

from typing import List
from skimage.measure._regionprops import RegionProperties

from IO import Match



def calculate_fixed_regions_mask(
    match: Match,
    half: int,
    optical_flow_indices: List[int] | np.ndarray,
    gpu_devices: List[int],
    cfg: DictConfig,
) -> np.ndarray:
    def load_abs_flow(idx: int) -> np.ndarray:
        optical_flow = match.optical_flow(half, optical_flow_indices[idx])
        return np.abs(optical_flow) / cfg.num_sampling_frames

    sampling_indices = np.ceil(
        np.linspace(0, match.num_rgb_frames[half] - 1, num=cfg.num_sampling_frames)
    ).astype(int)
    mean_flow = load_abs_flow(sampling_indices[0])
    for idx in sampling_indices[1:]:
        mean_flow += load_abs_flow(idx)
    mean_flow = gaussian(mean_flow, 3, multichannel=True, mode="reflect")

    kmeans = FaissKMeans(cfg.num_clusters, 100, gpu_devices)
    kmeans.fit(mean_flow.reshape((-1, 2)))
    labels = kmeans.predict(mean_flow.reshape((-1, 2)))

    mean_squared_error_ = np.sum(np.power(kmeans.centroids, 2), axis=1) / 2
    static_centroids_idx = [
        idx for idx, mse in enumerate(mean_squared_error_) if mse < 0.01
    ]

    centers = np.zeros(cfg.num_clusters, dtype=np.uint8)
    centers[static_centroids_idx] = 1

    mask = centers[labels.flatten()]
    mask = mask.reshape(mean_flow.shape[:2])

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    dilated_mask = cv2.dilate(mask, disk, iterations=1)
    return 255 * dilated_mask


def segment_fixed_regions(
    match: Match,
    half: int,
    optical_flow_indices: List[int] | np.ndarray,
    flow_sizes: np.ndarray,
    cfg: DictConfig,
) -> List[RegionProperties]:

    mask = calculate_fixed_regions_mask(
        match, half, optical_flow_indices, cfg.gpu_devices, cfg.velocity.fixed_regions
    )

    mask = add_missing_borders(mask, flow_sizes)
    sampling_indices = np.ceil(
        np.linspace(
            0,
            match.num_rgb_frames[half] - 1,
            num=cfg.velocity.fixed_regions.num_sampling_frames,
        )
    ).astype(int)

    idx = sampling_indices[0]
    rgb = io.imread(match.frames[half, idx])

    mask = resize(mask, rgb.shape[:2], anti_aliasing=False, preserve_range=True)
    regions = regionprops(label(img_as_bool(mask)))

    patches = [
        np.zeros((r.image.shape[0], r.image.shape[1], 3), dtype=np.float64)
        for r in regions
    ]
    for idx in sampling_indices:
        rgb = io.imread(match.frames[half, idx])

        for i, r in enumerate(regions):
            y1, x1, y2, x2 = r.bbox
            bbox = (x1, y1, x2, y2)
            patches[i] += get_patch(rgb, bbox, False) / len(sampling_indices)

    for i, r in enumerate(regions):
        r.mask = img_as_ubyte(r.image)
        patches[i] = cv2.bitwise_and(patches[i], patches[i], mask=r.mask)
        patches[i][patches[i] > 255] = 255
        r.patch = img_as_ubyte(patches[i] / 255)
    return regions


def field_segmentation(
    frame: np.ndarray,
    segmented_people: List[SegmentedPlayer],
    fixed_regions: List[RegionProperties],
    gpu_devices: List[int],
    cfg: DictConfig,
) -> np.ndarray:
    # Removing fixed regions from frame if found
    for r in fixed_regions:
        y1, x1, y2, x2 = r.bbox
        patch = get_patch(frame, (x1, y1, x2, y2), False)
        masked_patch = cv2.bitwise_and(patch, patch, mask=r.mask)

        similarity = mean_squared_error(masked_patch, r.patch) / 65025
        if similarity > cfg.fixed_region_min_similarity:
            continue
        patch[r.image, :] = 0

    black = np.asarray([0.0, 0.0, 0.0])
    for person in segmented_people:
        draw_mask(frame, person.bb, person.mask_cnts, black)

    # Color quantization using K-means
    pixels = np.float32(frame.reshape((-1, 3)))

    num_pixels = pixels.shape[0]
    sampled_indices = np.linspace(0, num_pixels - 1, num=num_pixels // 2).astype(int)

    kmeans = FaissKMeans(
        cfg.clustering.num_clusters, cfg.clustering.max_iter, gpu_devices
    )
    kmeans.fit(pixels[sampled_indices, :])
    centers = kmeans.centroids
    labels = kmeans.predict(pixels)

    # Sorting clusters by number of pixels
    sorted_indices = np.argsort(
        np.histogram(
            labels[sampled_indices], bins=np.arange(cfg.clustering.num_clusters + 1)
        )[0]
    )[::-1]
    centers = np.uint8(centers[sorted_indices])

    # The largest cluster is assumed to be the field
    field_indices = [0]
    field_color = centers[0]
    for idx in range(1, cfg.clustering.num_clusters):
        similarity = mean_squared_error(field_color, centers[idx]) / 65025
        if similarity < 0.01:
            field_color = (field_color + centers[idx]) / 2
            field_indices.append(idx)

    centers = np.zeros(cfg.clustering.num_clusters, dtype=np.uint8)
    centers[sorted_indices[field_indices]] = 1

    mask = centers[labels.flatten()]
    mask = mask.reshape(frame.shape[:2])

    # Removing small regions and connecting big ones
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.opening_disk_radius, cfg.opening_disk_radius)
    )
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk)

    # Filtering regions by area
    min_field_area = np.power(
        cfg.min_area_proportion * np.max(opened_mask.shape[:2]), 2
    )

    filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
    contours = cv2.findContours(
        opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    for cnt in contours:
        if cv2.contourArea(cnt) < min_field_area:
            continue
        hull = cv2.convexHull(cnt)
        cv2.drawContours(filtered_mask, [hull], 0, 255, -1)

    mask = cv2.bitwise_and(opened_mask, opened_mask, mask=filtered_mask)
    return mask


def velocity_vectors_from_frame(
    rgb: np.ndarray,
    flow: np.ndarray,
    rgb_shape: Shape,
    flow_sizes: Dict,
    segmented_people: List[SegmentedPlayer],
    fixed_regions: List[RegionProperties],
    gpu_devices: List[int],
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = caculate_inertia_matrix_components(flow)
    field_mask = field_segmentation(
        rgb, segmented_people, fixed_regions, gpu_devices, cfg
    )
    field_cnt = to_polygons(field_mask)
    field_mask = img_as_bool(
        resize_to_flow_shape_and_remove_borders(field_mask, flow_sizes)
    )
    flow_vectors = flow[field_mask, :]

    if len(flow_vectors) > 0:
        M = build_second_moment_matrix(components, mask=field_mask)
        field_vector = caculate_dominant_orientation_vector(M, flow_vectors)
    else:
        field_vector = np.zeros(2)

    scale = np.asarray([f / r for r, f in zip(rgb_shape[1::-1], flow.shape[1::-1])])
    velocity_vectors = []
    for person in segmented_people:
        scaled_bb, scaled_mask_cnts = scale_mask(person.bb, person.mask_cnts, scale)
        flow_patch = get_patch(flow, scaled_bb, copy=False)

        if len(scaled_mask_cnts) == 0:
            warnings.warn(
                f"Segmented person with bbox [{person.bb[0]}, {person.bb[1]}, {person.bb[2]}, {person.bb[3]}] has no contour"
            )
            rr, cc = np.indices(flow_patch.shape[0:2])
            rr = rr.flatten()
            cc = cc.flatten()
        else:
            rr, cc = cnts_to_indices(scaled_mask_cnts)
            # FIXME: Polygon function sometimes exceeds the patch shape
            indices = (rr < flow_patch.shape[0]) & (cc < flow_patch.shape[1])
            if len(indices) < len(rr):
                warnings.warn(f"Generated polygon exceeds patch shape")

            rr, cc = rr[indices], cc[indices]

        flow_vectors = flow_patch[rr, cc, :]

        components_patches = [get_patch(c, scaled_bb, copy=False) for c in components]
        M = build_second_moment_matrix(components_patches, indices=(rr, cc))

        player_vector = caculate_dominant_orientation_vector(M, flow_vectors)
        velocity_vectors.append(player_vector - field_vector)

    return field_cnt, field_vector, np.asarray(velocity_vectors)


def velocity_vectors_from_half_match(match: Match, half: int, cfg: DictConfig) -> Dict[int, Dict[str, np.ndarray]]:

    # Matching corresponding RGB frames to optical flow frames.
    if match.paths.fixed_indices_results[half].exists():
        match_indices = np.load(match.paths.fixed_indices_results[half])[
            "match_indices"
        ]
        num_all_frames = len(
            np.load(match.paths.all_segmentations[half], allow_pickle=True)
        )
        optical_flow_indices = [
            int(i * (match.num_optical_flow_frames[half] - 1) / (num_all_frames - 1))
            for i in match_indices
        ]
    else:
        optical_flow_indices = (
            np.ceil(
                np.linspace(
                    match.num_optical_flow_frames[half] / match.num_rgb_frames[half],
                    match.num_optical_flow_frames[half],
                    num=match.num_rgb_frames[half],
                )
            ).astype(int)
            - 1
        )

    rgb_shape = images_size(match.paths.frames[half, 0])
    flow_sizes = flow_sizes_constants(rgb_shape)

    fixed_regions = segment_fixed_regions(
        match, half, optical_flow_indices, flow_sizes, cfg
    )

    results = {}
    segmented_people = SegmentedPlayer.load(
        match.paths.segmentations[half], class_id_filters=POINTREND_PERSON_ID
    )
    for idx in tqdm(
        range(match.num_rgb_frames[half]),
        desc=f"Half-match {half + 1} progress",
        leave=True,
        position=0,
    ):

        if len(segmented_people[idx]) == 0:
            continue

        rgb = io.imread(match.paths.frames[half, idx])
        flow = match.load_optical_flow(half, optical_flow_indices[idx])
        flow = add_missing_borders(flow, flow_sizes)
        field_mask, field_vector, velocity = velocity_vectors_from_frame(
            rgb,
            flow,
            rgb_shape,
            flow_sizes,
            segmented_people[idx],
            fixed_regions,
            cfg.gpu_devices,
            cfg.velocity.field_segmentation,
        )
        results[idx] = {
            "field_mask": field_mask,
            "field_vector": field_vector,
            "velocity": velocity,
        }
    return results


@hydra.main(version_base=None, config_path="conf", config_name="velocity_")
def main(cfg: DictConfig):
    # Validation for the mutually exclusive group
    if (cfg.single_match is None) == (cfg.matches is None):
        raise ValueError(
            "You must provide either 'single_match' OR 'matches', but not both/neither."
        )

    load_log_configuration(cfg.logs.config, cfg.logs.dir)

    if cfg.matches:
        with cfg.matches.open() as f:
            match_paths = [Path(line) for line in f.read().splitlines()]
    else:
        match_paths = [cfg.single_match]

    num_optical_flow_frames = load_num_optical_flow_frames(cfg.dataset_path)

    for match_path in tqdm(
        match_paths, desc="Overall Progress", leave=True, position=0
    ):
        match = Match(match_path)
        if match.paths.velocity_results.exists():
            continue
        start = time.time()

        match.num_optical_flow_frames = num_optical_flow_frames[match_path]
        match.load_metadata()

        logging.info(f"Processing match {match_paths.match}")
        velocity = {}
        for half in range(2):
            velocity[half] = velocity_vectors_from_half_match(match, half, cfg)

        np.save(match.paths.velocity_results, velocity)
        logging.info(f"Match processing time is {time.time() - start} seconds")


if __name__ == "__main__":
    main()
