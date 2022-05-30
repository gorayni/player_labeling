from collections import OrderedDict
from itertools import combinations

import cv2
import numpy as np
from nltk.cluster import KMeansClusterer
from nltk.cluster import euclidean_distance
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from regions import bhattacharyya_distance


def fit_svm(class_samples, gamma, kernel):
    svc = SVC(gamma=gamma, kernel=kernel, probability=True)
    X, y = [], []
    for category, samples in enumerate(class_samples.values()):
        X.extend(samples.hists)
        y.extend(category * np.ones((len(samples))))
    X = np.asarray(X)
    y = np.asarray(y)
    svc.fit(X, y)
    return svc


def calc_distance_matrix(kmeans):
    n = kmeans.num_clusters()
    distance_matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        distance_matrix[i, j] = distance_matrix[j, i] = bhattacharyya_distance(kmeans._means[i], kmeans._means[j])
    return distance_matrix


def triangle_area(sides):
    sides = np.asarray(sides)
    p = np.sum(sides) / 2
    return np.sqrt(p * np.product(p - sides))


def most_distant_triplets(distance_matrix, nodes=None):
    if not nodes:
        n = distance_matrix.shape[0]
        nodes = np.arange(n)

    triplet, max_dist = None, -1
    for (i, j, k) in combinations(nodes, 3):
        dist = triangle_area([distance_matrix[i, j], distance_matrix[i, k], distance_matrix[j, k]])
        if dist > max_dist:
            triplet = [i, j, k]
            max_dist = dist
    return triplet


def calc_distribution_distances(gmms, hists, criteria):
    def bhattacharyya_dist(u, v):
        axis = 1 if len(v.shape) > 1 else 0
        return np.sqrt(
            np.abs(1 - np.sum(np.sqrt(u * v), axis=axis) / np.sqrt(u.mean() * v.mean(axis=axis) * np.power(u.size, 2))))

    criteria = np.median if criteria == 'median' else np.min
    distances = [criteria([bhattacharyya_dist(g.means_[j, :], hists) for j in range(g.n_components)], axis=0) for g in
                 gmms.values()]
    return np.asarray(distances).T


class Samples:
    def __init__(self):
        self.hists = []
        self.locations = []

    @property
    def hists_(self):
        return np.asarray(self.hists).squeeze()

    def extend(self, other):
        self.hists.extend([h for h in other.hists])
        self.locations.extend(other.locations)

    def add(self, hist, location):
        self.hists.append(hist)
        self.locations.append(location)

    def __getitem__(self, key):
        return self.hists[key], self.locations[key]

    def __len__(self):
        return len(self.hists)


def init_clusters(midfielders_blobs, hists, labels, num_clusters):
    clusters = [Samples() for _ in range(num_clusters)]
    i = 0
    for half, players_blobs in midfielders_blobs.items():
        for frame_idx, blobs in players_blobs.items():
            for blob in blobs:
                clusters[labels[i]].add(hists[i, :], (half, frame_idx, blob))
                i += 1
    return clusters


def calculate_gmms(samples, num_components):
    nodes = [k for k in samples.keys()]
    hists = {i: np.asarray(s.hists).squeeze() for i, s in samples.items()}
    gmms = OrderedDict([(i, GaussianMixture(n_components=num_components, random_state=0).fit(hists[i])) for i in nodes])
    max_scores = np.asarray([gmm.score_samples(hists[i]).max() for i, gmm in gmms.items()])
    return gmms, max_scores


def init_gmm(clusters, nodes, num_components, min_likelihood):

    gmms = OrderedDict([(n, GaussianMixture(n_components=1, random_state=0).fit(clusters[n].hists_)) for n in nodes])

    samples = OrderedDict([(n, Samples()) for n in nodes])

    num_clusters = len(clusters)
    new_clusters = [Samples() for _ in range(num_clusters)]

    for i in range(num_clusters):
        if i not in nodes:
            new_clusters[i].extend(clusters[i])
            continue

        scores = gmms[i].score_samples(clusters[i].hists_)
        for j, score in enumerate(scores / scores.max()):
            if score >= min_likelihood:
                samples[i].add(*clusters[i][j])
            else:
                new_clusters[i].add(*clusters[i][j])

    gmms, max_scores = calculate_gmms(samples, num_components)
    return gmms, max_scores, clusters, samples


def expand_gaussians(gmms, max_scores, clusters, samples, min_thres, other_thres, num_components, criteria):
    num_clusters = len(clusters)
    new_clusters = [Samples() for _ in range(num_clusters)]
    nodes = list(gmms.keys())

    for i in range(num_clusters):
        if len(clusters[i]) < 1:
            continue
        hists = clusters[i].hists_
        scores = np.asarray([g.score_samples(hists) for g in gmms.values()]).T
        distances = calc_distribution_distances(gmms, hists, criteria)

        for j, score in enumerate(scores / max_scores):
            if score[0] < 0.85 and score[1] < 0.85 and score[2] < 0.85:
                new_clusters[i].add(*clusters[i][j])
                continue

            min_index = np.argmin(distances[j, :])
            others = [0, 1, 2]
            others.remove(min_index)

            min_dist = distances[j, min_index]
            other_dists = [distances[j, k] - other_thres for k in others]
            if min_dist < min_thres and min_dist < other_dists[0] and min_dist < other_dists[1]:
                samples[nodes[min_index]].add(*clusters[i][j])
            else:
                new_clusters[i].add(*clusters[i][j])

    gmms, max_scores = calculate_gmms(samples, num_components)
    return gmms, max_scores, new_clusters


def predict_candidates(midfielders_blobs, hists, class_samples, gamma, kernel, svc_probability_thres):
    svm = fit_svm(class_samples, gamma, kernel)
    probabilities = svm.predict_proba(hists)

    labels = np.argmax(probabilities, axis=1)
    labels_count = np.unique(labels, return_counts=True)[1]
    referee_index = np.argmin(labels_count)

    if referee_index > 0:
        class_indices = [referee_index, 0, 1 if referee_index == 2 else 2]
        relabel = {c: i for i, c in enumerate(class_indices)}
    else:
        relabel = {i: i for i in [0, 1, 2]}

    candidates = {i: Samples() for i in range(3)}
    i = 0
    for half, players_blobs in midfielders_blobs.items():
        for frame_idx, blobs in players_blobs.items():
            for blob in blobs:
                label = labels[i]
                if probabilities[i, label] >= svc_probability_thres:
                    sample = (hists[i, :], (half, frame_idx, blob))
                    candidates[relabel[label]].add(*sample)
                i += 1
    return candidates


def filter_candidates(candidates, min_likelihood=0.99):
    filtered_candidates = {i: Samples() for i in range(3)}
    for i, samples in candidates.items():
        hist = samples.hists_
        gmm = GaussianMixture(n_components=1, random_state=0).fit(hist)
        scores = gmm.score_samples(hist)
        scores /= scores.max()

        for score, sample in zip(scores, samples):
            if score > min_likelihood:
                filtered_candidates[i].add(*sample)
    return filtered_candidates


def label_midfielders(midfielders_blobs, num_clusters=10, distance='euclidean', distances_thresholds=None,
                      non_outlier_thres=100, num_components=10, criteria='median', min_likelihood=0.99,
                      svc_probability_thres=0.99, gamma=0.1, kernel='rbf'):

    hists = [blob.hist for half in range(2) for _, frame in midfielders_blobs[half].items() for blob in frame]
    hists = np.squeeze(np.asarray(hists, dtype=np.float32))

    distance = euclidean_distance if distance == 'euclidean' else bhattacharyya_distance
    clusterer = KMeansClusterer(num_clusters, distance)
    labels = clusterer.cluster(hists, True)

    distance_matrix = calc_distance_matrix(clusterer)
    labels_count = np.zeros(num_clusters, dtype=np.int32)
    for i in labels:
        labels_count[i] += 1

    clusters = init_clusters(midfielders_blobs, hists, labels, num_clusters)

    non_outliers = [i for i in range(num_clusters) if labels_count[i] > non_outlier_thres]
    nodes = most_distant_triplets(distance_matrix, non_outliers)

    gmms, max_scores, clusters, class_samples = init_gmm(clusters, nodes, num_components, min_likelihood)

    if distances_thresholds is None:
        distances_thresholds = [(0.15, 0.25), (0.2, 0.3), (0.15, 0.25), (0.2, 0.3)]

    num_samples = sum([len(s) for s in class_samples.values()])
    for (min_thres, other_distance_thres) in distances_thresholds:
        last_num_samples = -1
        while num_samples != last_num_samples:
            last_num_samples = num_samples
            gmms, max_scores, clusters = expand_gaussians(gmms, max_scores, clusters, class_samples, min_thres,
                                                          other_distance_thres, num_components, criteria)
            num_samples = sum([len(s) for s in class_samples.values()])

    candidates = predict_candidates(midfielders_blobs, hists, class_samples, gamma, kernel, svc_probability_thres)
    candidates = filter_candidates(candidates, min_likelihood)

    labels = dict()
    for category, samples in candidates.items():
        for half, frame_idx, blob in samples.locations:
            if half not in labels:
                labels[half] = {}
            if frame_idx not in labels[half]:
                labels[half][frame_idx] = []

            blob.category = category
            labels[half][frame_idx].append(blob)
    return labels
