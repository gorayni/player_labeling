import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from matplotlib.colors import ListedColormap


def show_confusion_matrix(true_labels, predicted_labels, labels, figsize=None,
                          normalize=True, annot=False, cmap=None,
                          square=False, linecolor='white', ticks_size=None,
                          linewidths=0, show_yticks=True, show_xticks=False,
                          cbar=True, cbar_kws=None):
    if not figsize:
        figsize = (5, 5)

    if type(true_labels[0]) == type(labels[0]):
        cm = metrics.confusion_matrix(true_labels, predicted_labels, labels)
    else:
        cm = metrics.confusion_matrix(true_labels, predicted_labels, np.arange(np.size(labels)))

    # normalize confusion matrix
    if normalize:
        num_instances_per_class = cm.sum(axis=1)
        zero_indices = num_instances_per_class == 0
        if any(zero_indices):
            num_instances_per_class[zero_indices] = 1
            warnings.warn('One or more classes does not have instances')
        cm = cm / num_instances_per_class[:, np.newaxis]
        vmax = 1.
    else:
        vmax = np.max(cm)

    if not cmap:
        cmap = list(sns.color_palette("RdBu_r", 200).as_hex())
        cmap = ListedColormap(cmap)

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    if show_yticks:
        yticklabels = labels
    else:
        yticklabels = []

    if show_xticks:
        xticklabels = labels
    else:
        xticklabels = []

    ax = sns.heatmap(cm, annot=annot, cmap=cmap, linewidths=linewidths, xticklabels=xticklabels,
                     yticklabels=yticklabels, square=square,
                     linecolor=linecolor, vmax=vmax, cbar=cbar,
                     cbar_kws=cbar_kws)

    if show_yticks:
        for ticklabel in ax.get_yaxis().get_ticklabels():
            ticklabel.set_rotation('horizontal')
            if ticks_size:
                ticklabel.set_fontsize(ticks_size)

    if show_xticks:
        ax.xaxis.tick_top()
        for ticklabel in ax.get_xaxis().get_ticklabels():
            ticklabel.set_rotation('vertical')
            if ticks_size:
                ticklabel.set_fontsize(ticks_size)

    return fig, ax


def plot_players_cm(groundtruth, predictions, labels=None, show_title=True, show_ylabel=True):
    if not labels:
        labels = ['Referee', 'Team A', 'Team B', 'Goalkeeper 1', 'Goalkeeper 2']
    fig, ax = show_confusion_matrix(groundtruth, predictions, labels,
                                    figsize=(8, 8),
                                    annot=False,
                                    cmap=sns.color_palette("Reds", as_cmap=True),
                                    linewidths=0.1,
                                    linecolor='white',
                                    square=True,
                                    show_xticks=True,
                                    show_yticks=True,
                                    ticks_size=14,
                                    cbar_kws={"shrink": 0.75})

    if show_title:
        acc = metrics.accuracy_score(groundtruth, predictions)
        ax.set_title('Accuracy {:.2f}%'.format(acc * 100), y=1.3, fontsize=15, fontweight='bold')

    if show_ylabel:
        ax.set_ylabel('Groundtruth', fontsize=15, fontweight='bold')

    return fig, ax
