import cv2
import numpy as np
from functools import partial

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


def calculate_hist(img, mask=None, hist_type='rgb'):
    if hist_type == 'rgb':
        # The histogram consists of 512 bins (quantize the RGB space into cubes by grid 8*8*8)
        hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    if hist_type == 'hsv':
        # The histogram consists of 512 bins (quantize the HSV space into cubes by grid 8*8*8)
        hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    if hist_type == 'hs':
        # The histogram consists of 512 bins (quantize the HS space into cubes by grid 8*8)
        hist = cv2.calcHist([img], [0, 1], mask, [8, 8], [0, 180, 0, 256])
    if hist_type == 'lab':
        # The histogram consists of 512 bins (quantize the LAB space into cubes by grid 8*8*8)
        hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    if hist_type == 'ab':
        # The histogram consists of 512 bins (quantize the AB space into cubes by grid 8*8)
        hist = cv2.calcHist([img], [0, 1], mask, [8, 8], [0, 256, 0, 256])
    hist = np.expand_dims(hist.flatten(), axis=0)
    hist /= hist.sum()
    return hist.astype(np.float32)


def get_patch(frame, bbox, copy=True):
    x1, y1, x2, y2 = bbox
    if copy:
        return np.copy(frame[y1:y2, x1:x2, :])
    else:
        return frame[y1:y2, x1:x2, :]


def calculate_patch_hist(patch, mask=None, hist_type='rgb'):
    if hist_type == 'hsv' or hist_type == 'hs':
        patch = rgb2hsv(patch)
    elif hist_type == 'lab' or hist_type == 'ab':
        patch = rgb2lab(patch)
    return calculate_hist(patch, mask, hist_type)
