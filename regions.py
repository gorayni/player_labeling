from functools import partial

import cv2
import colour
import numpy as np
from addict import Dict
from skimage.draw import polygon
from kitman.regions import iou

rgb2hsv = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HSV)
rgb2lab = partial(cv2.cvtColor, code=cv2.COLOR_RGB2LAB)


def bhattacharyya_distance(u, v):
    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)


def rgb2xyz(rgb):
    height, width = rgb.shape[:2]
    rgb = rgb.reshape((height * width, 3))

    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.34570, 0.35850])

    matrix_RGB_to_XYZ = np.array(
        [[0.41240000, 0.35760000, 0.18050000],
         [0.21260000, 0.71520000, 0.07220000],
         [0.01930000, 0.11920000, 0.95050000]]
    )
    chromatic_adaptation_transform = 'Bradford'
    xyz = colour.RGB_to_XYZ(rgb, illuminant_RGB, illuminant_XYZ, matrix_RGB_to_XYZ, chromatic_adaptation_transform)
    return xyz.reshape((height, width, 3))


def rgb2oklab(rgb):
    rgb = rgb.astype(np.float32) / 255
    xyz = rgb2xyz(rgb)
    height, width = xyz.shape[:2]
    oklab = colour.XYZ_to_Oklab(xyz)
    oklab = oklab.reshape((height, width, 3))
    oklab[:, :, 0] = oklab[:, :, 0] * 100
    oklab[:, :, 1] = oklab[:, :, 1] * 127 + 127
    oklab[:, :, 2] = oklab[:, :, 1] * 127 + 127
    oklab = np.clip(oklab, 0, 255).astype(np.uint8)
    return oklab


def match_by_iou(bboxes):
    num_left, num_right = map(len, bboxes)
    iou_scores = np.asarray([[iou(bboxes[0][j], bboxes[1][i]) for j in range(num_left)] for i in range(num_right)])

    if num_left <= num_right:
        correspondence = np.argmax(iou_scores, axis=0).tolist()
        left_indices, right_indices = [], []
        for i, c in enumerate(correspondence):
            left_indices.append(i)
            right_indices.append(c)
    else:
        correspondence = np.argmax(iou_scores, axis=1).tolist()
        left_indices, right_indices = [], []
        for i, c in enumerate(correspondence):
            left_indices.append(c)
            right_indices.append(i)
    return left_indices, right_indices


def calculate_hist(img, mask=None, hist_type='rgb'):
    hist_options = Dict({'rgb': {'channels': [0, 1, 2],
                                 'grid': [8, 8, 8],
                                 'values_ranges': [0, 256, 0, 256, 0, 256]},
                         'hsv': {'channels': [0, 1, 2],
                                 'grid': [8, 8, 8],
                                 'values_ranges': [0, 180, 0, 256, 0, 256]},
                         'hs': {'channels': [0, 1],
                                'grid': [8, 8],
                                'values_ranges': [0, 180, 0, 256]},
                         'lab': {'channels': [0, 1, 2],
                                 'grid': [8, 8, 8],
                                 'values_ranges': [0, 256, 0, 256, 0, 256]},
                         'ab': {'channels': [1, 2],
                                'grid': [8, 8],
                                'values_ranges': [0, 256, 0, 256]},
                         'oklab': {'channels': [1, 2],
                                   'grid': [8, 8],
                                   'values_ranges': [0, 256, 0, 256]},
                         })
    if options := hist_options.get(hist_type, None):
        hist = cv2.calcHist([img], options.channels, mask, options.grid, options.values_ranges)
        hist = np.expand_dims(hist.flatten(), axis=0)
        hist /= hist.sum()
        return hist.astype(np.float32)
    return None


def calculate_patch_hist(patch, mask=None, hist_type='rgb'):
    if hist_type == 'hsv' or hist_type == 'hs':
        patch = rgb2hsv(patch)
    elif hist_type == 'lab' or hist_type == 'ab':
        patch = rgb2lab(patch)
    elif hist_type == 'oklab':
        patch = rgb2oklab(patch)
    return calculate_hist(patch, mask, hist_type)


def scale_mask(bb, cnts, scale: np.ndarray):
    scaled_mask_bb = (np.tile(scale, 2) * bb).astype(np.int32)
    scaled_mask_shape = scaled_mask_bb[2:] - scaled_mask_bb[:2]

    mask_shape = bb[2:] - bb[:2]
    mask_scale = scaled_mask_shape / mask_shape

    scaled_mask_cnts = [(mask_scale * cnt).astype(np.int32) for cnt in cnts]
    return scaled_mask_bb, scaled_mask_cnts


def cnts_to_indices(contours):
    indices = np.concatenate([polygon(c[:, 1], c[:, 0]) for c in contours], axis=1).T
    return indices[:, 0], indices[:, 1]
