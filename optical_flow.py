import cv2
import numpy as np
from addict import Dict
from numpy import linalg as LA
from skimage.transform import resize


# These shape sizes come from the optical flow calculated using FlowNet 2.0
def flow_sizes_constants(rgb_shape):
    flow_sizes = Dict()
    if rgb_shape[0] == 480:
        flow_sizes.RGB_SHAPE = np.asarray((390, 693))
        flow_sizes.ORIGINAL_FLOW_BORDERS = np.asarray((3, 26.5))
        flow_sizes.FLOW_SHAPE = np.asarray((256, 426))
    elif rgb_shape[0] == 576:
        flow_sizes.RGB_SHAPE = np.asarray((468, 699))
        flow_sizes.ORIGINAL_FLOW_BORDERS = np.asarray((10, 29.5))
        flow_sizes.FLOW_SHAPE = np.asarray((256, 365))
    elif rgb_shape[0] == 768:
        flow_sizes.RGB_SHAPE = np.asarray((624, 1109))
        flow_sizes.ORIGINAL_FLOW_BORDERS = np.asarray((24, 10.5))
        flow_sizes.FLOW_SHAPE = np.asarray((256, 483))
    elif rgb_shape[0] == 720:
        flow_sizes = Dict()
        flow_sizes.RGB_SHAPE = np.asarray((585, 1040))
        flow_sizes.ORIGINAL_FLOW_BORDERS = np.asarray((8, 4.5))
        flow_sizes.FLOW_SHAPE = np.asarray((256, 455))
    elif rgb_shape[0] == 1080:
        flow_sizes.RGB_SHAPE = np.asarray((877, 1560))
        flow_sizes.ORIGINAL_FLOW_BORDERS = np.asarray((22.5, 12))
        flow_sizes.FLOW_SHAPE = np.asarray((256, 472))

    flow_sizes.ORIGINAL_FLOW_SHAPE = (flow_sizes.RGB_SHAPE - 2 * flow_sizes.ORIGINAL_FLOW_BORDERS).astype(int)
    flow_sizes.FLOW_BORDERS = flow_sizes.ORIGINAL_FLOW_BORDERS * flow_sizes.FLOW_SHAPE / flow_sizes.ORIGINAL_FLOW_SHAPE
    flow_sizes.COMPLETE_FLOW_SHAPE = flow_sizes.FLOW_SHAPE + np.ceil(flow_sizes.FLOW_BORDERS).astype(int) + np.floor(
        flow_sizes.FLOW_BORDERS).astype(int)
    return flow_sizes


def restore_flow_original_size(flow, flow_sizes):
    flow = resize(flow, flow_sizes.ORIGINAL_FLOW_SHAPE, anti_aliasing=False, preserve_range=True)
    return cv2.copyMakeBorder(flow,
                              np.ceil(flow_sizes.ORIGINAL_FLOW_BORDERS[0]).astype(int),
                              np.floor(flow_sizes.ORIGINAL_FLOW_BORDERS[0]).astype(int),
                              np.ceil(flow_sizes.ORIGINAL_FLOW_BORDERS[1]).astype(int),
                              np.floor(flow_sizes.ORIGINAL_FLOW_BORDERS[1]).astype(int),
                              cv2.BORDER_CONSTANT)


def add_missing_borders(flow, flow_sizes):
    return cv2.copyMakeBorder(flow,
                              np.ceil(flow_sizes.FLOW_BORDERS[0]).astype(int),
                              np.floor(flow_sizes.FLOW_BORDERS[0]).astype(int),
                              np.ceil(flow_sizes.FLOW_BORDERS[1]).astype(int),
                              np.floor(flow_sizes.FLOW_BORDERS[1]).astype(int),
                              cv2.BORDER_CONSTANT)


def resize_to_flow_shape_and_remove_borders(img, flow_sizes):
    img = resize(img, flow_sizes.COMPLETE_FLOW_SHAPE, anti_aliasing=False, preserve_range=True)
    img[:np.ceil(flow_sizes.FLOW_BORDERS[0]).astype(int), :] = 0
    img[-np.floor(flow_sizes.FLOW_BORDERS[0]).astype(int):, :] = 0
    img[:, :np.ceil(flow_sizes.FLOW_BORDERS[1]).astype(int)] = 0
    img[:, -np.floor(flow_sizes.FLOW_BORDERS[1]).astype(int):] = 0
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
    M[1, 1] = v
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
