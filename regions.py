import warnings
from functools import partial

import cv2
import numpy as np
from addict import Dict
from skimage.draw import polygon

rgb2hsv = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HSV)
rgb2lab = partial(cv2.cvtColor, code=cv2.COLOR_RGB2LAB)


def area(bbox):
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = area(boxA)
    boxBArea = area(boxB)

    return interArea / float(boxAArea + boxBArea - interArea)


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
                         'ab': {'channels': [0, 1],
                                'grid': [8, 8],
                                'values_ranges': [0, 256, 0, 256]},
                         })
    if options := hist_options.get(hist_type, None):
        hist = cv2.calcHist([img], options.channels, mask, options.grid, options.values_ranges)
        hist = np.expand_dims(hist.flatten(), axis=0)
        hist /= hist.sum()
        return hist.astype(np.float32)
    return None


def get_patch(frame, bbox, copy=True):
    x1, y1, x2, y2 = bbox
    if len(frame.shape) == 3:
        patch = frame[y1:y2, x1:x2, :]
    else:
        patch = frame[y1:y2, x1:x2]
    return np.copy(patch) if copy else patch


def to_mask(bb, contours):
    width, height = bb[2:] - bb[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for cnt in contours:
        rr, cc = polygon(cnt[:, 1], cnt[:, 0])

        # FIXME: Polygon function sometimes exceeds the patch shape
        indices = (rr < height) & (cc < width)
        if len(indices) < len(rr):
            warnings.warn(f'Generated polygon exceeds patch shape')
        rr, cc = rr[indices], cc[indices]

        mask[rr, cc] = 255
    return mask


def draw_mask(frame, bb, contours, color=None):
    x1, y1, x2, y2 = bb

    if color is None:
        color = 255

    patch = frame[y1:y2, x1:x2, ...]
    for cnt in contours:
        rr, cc = polygon(cnt[:, 1], cnt[:, 0])

        # FIXME: Polygon function sometimes exceeds the patch shape
        original_rr_size = len(rr)
        rr = np.minimum(rr, patch.shape[0] - 1)
        cc = np.minimum(cc, patch.shape[1] - 1)

        if len(rr) < original_rr_size:
            warnings.warn(f'Generated polygon exceeds patch shape')

        patch[rr, cc, ...] = color


def get_masked_patch(frame, bbox, mask_contour):
    patch = get_patch(frame, bbox, copy=False)
    mask = to_mask(bbox, mask_contour)
    return cv2.bitwise_and(patch, patch, mask=mask)


def calculate_patch_hist(patch, mask=None, hist_type='rgb'):
    if hist_type == 'hsv' or hist_type == 'hs':
        patch = rgb2hsv(patch)
    elif hist_type == 'lab' or hist_type == 'ab':
        patch = rgb2lab(patch)
    return calculate_hist(patch, mask, hist_type)


def scale_mask(bb, cnts, scale):
    scaled_mask_bb = (np.tile(scale, 2) * bb).astype(np.int32)
    scaled_mask_shape = scaled_mask_bb[2:] - scaled_mask_bb[:2]

    mask_shape = bb[2:] - bb[:2]
    mask_scale = scaled_mask_shape / mask_shape

    scaled_mask_cnts = [(mask_scale * cnt).astype(np.int32) for cnt in cnts]
    return scaled_mask_bb, scaled_mask_cnts


def cnts_to_indices(contours):
    indices = np.concatenate([polygon(c[:, 1], c[:, 0]) for c in contours], axis=1).T
    return indices[:, 0], indices[:, 1]
