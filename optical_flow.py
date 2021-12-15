import flowiz
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy import linalg as LA
import cv2
from skimage.transform import resize

# These shape sizes come from the optical flow calculated using FlowNet 2.0
RGB_SHAPE = np.asarray((877, 1560))
ORIGINAL_FLOW_BORDERS = np.asarray((22.5, 12))
ORIGINAL_FLOW_SHAPE = (RGB_SHAPE - 2 * ORIGINAL_FLOW_BORDERS).astype(int)

FLOW_SHAPE = np.asarray((256, 472))
FLOW_BORDERS = ORIGINAL_FLOW_BORDERS * FLOW_SHAPE / ORIGINAL_FLOW_SHAPE
COMPLETE_FLOW_SHAPE = FLOW_SHAPE + np.ceil(FLOW_BORDERS).astype(int) + np.floor(FLOW_BORDERS).astype(int)


def restore_flow_original_size(flow):
    flow = resize(flow, ORIGINAL_FLOW_SHAPE, anti_aliasing=False, preserve_range=True)
    return cv2.copyMakeBorder(flow,
                              np.ceil(ORIGINAL_FLOW_BORDERS[0]).astype(int),
                              np.floor(ORIGINAL_FLOW_BORDERS[0]).astype(int),
                              np.ceil(ORIGINAL_FLOW_BORDERS[1]).astype(int),
                              np.floor(ORIGINAL_FLOW_BORDERS[1]).astype(int),
                              cv2.BORDER_CONSTANT)


def add_missing_borders(flow):
    return cv2.copyMakeBorder(flow,
                              np.ceil(FLOW_BORDERS[0]).astype(int),
                              np.floor(FLOW_BORDERS[0]).astype(int),
                              np.ceil(FLOW_BORDERS[1]).astype(int),
                              np.floor(FLOW_BORDERS[1]).astype(int),
                              cv2.BORDER_CONSTANT)


def resize_to_flow_shape_and_remove_borders(img):
    img = resize(img, COMPLETE_FLOW_SHAPE, anti_aliasing=False, preserve_range=True)
    img[:np.ceil(FLOW_BORDERS[0]).astype(int), :] = 0
    img[-np.floor(FLOW_BORDERS[0]).astype(int):, :] = 0
    img[:, :np.ceil(FLOW_BORDERS[1]).astype(int)] = 0
    img[:, -np.floor(FLOW_BORDERS[1]).astype(int):] = 0
    return img


def caculate_inertia_matrix_components(flow):
    u = np.power(flow[:, :, 0], 2)
    v = np.power(flow[:, :, 1], 2)
    uv = flow[:, :, 0] * flow[:, :, 1]
    return u, v, uv


def build_second_moment_matrix(components, mask=None, indices=None):
    # M = np.sum([np.tensordot(gradient_vectors[i], gradient_vectors[i].T, axes=0) for i in range(N)], axis=0)
    if mask is not None:
        u, v, uv = [c[mask].sum(axis=0) for c in components]
    else:
        rr, cc = indices
        u, v, uv = [c[rr, cc].sum(axis=0) for c in components]

    M = uv * np.ones((2, 2))
    M[0, 0] = u
    M[1, 1] = u
    return M


# Second Moment Matrix Method
def caculate_dominant_orientation_vector(M, gradient_vectors):

    magnitude = np.trace(M) / gradient_vectors.shape[0]

    l, e = LA.eig(M)
    i = np.argsort(l)[1]
    vector = e[:, i]

    # Calculating orientation histograms of 4 quadrants
    hists = np.apply_along_axis(lambda g: np.histogram(g, bins=[-1, 0, 1])[0], 1, gradient_vectors.T)
    direction = 2. * np.argmax(hists, axis=1) - 1

    # Setting the correct direction
    vector = direction * np.abs(vector)

    return magnitude * vector


def flow_colormap():
    flow = np.asarray([[np.cos(r), np.sin(r)] for r in np.linspace(0, 2 * np.pi, num=1000)])
    flow = flow[:, np.newaxis, :]
    rgb = flowiz.convert_from_flow(flow, 'rgb').squeeze() / 255
    return LinearSegmentedColormap.from_list('flow', rgb)
