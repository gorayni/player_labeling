import warnings
from functools import cache

import flowiz
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from matplotlib.colors import LinearSegmentedColormap
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


def draw_velocity_vectors(ax, xy, velocity, field_vector, img_width, magnitude_scale=0.0025):
    ax.quiver(xy[:, 0], xy[:, 1],
              velocity[:, 0], velocity[:, 1],
              color=(1, 1, 0),
              units='xy',
              scale=magnitude_scale,
              width=magnitude_scale * img_width,
              headwidth=3)
    ax.quiver(img_width // 2, 50,
              field_vector[0], field_vector[1],
              color=(0, 1, 0),
              units='xy',
              scale=magnitude_scale,
              width=magnitude_scale * img_width,
              headwidth=3)
    ax.quiver(50, 50,
              1, 0,
              color=(1, 1, 0),
              units='xy',
              scale=magnitude_scale,
              width=0.0025 * img_width,
              headwidth=3)


@cache
def flow_colormap():
    flow = np.asarray([[np.cos(r), np.sin(r)] for r in np.linspace(0, 2 * np.pi, num=1000)])
    flow = flow[:, np.newaxis, :]
    rgb = flowiz.convert_from_flow(flow, 'rgb').squeeze() / 255
    return LinearSegmentedColormap.from_list('flow', rgb)


def plot_flow_and_masked_flow(flow_rgb, masked_flow_rgb, xy, velocity, field_vector, title=None,
                              magnitude_scale=0.0025):
    fig = plt.figure(figsize=(16, 15))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[9, 1])

    if title:
        fig.suptitle(title)

    ax = fig.add_subplot(spec[0, 0])
    ax.imshow(flow_rgb)
    draw_velocity_vectors(ax, xy, velocity, field_vector, flow_rgb.shape[1], magnitude_scale)

    # Adding Flow colormap
    ax = fig.add_subplot(spec[0, 1], polar=True)
    mpl.colorbar.ColorbarBase(ax,
                              cmap=flow_colormap(),
                              norm=mpl.colors.Normalize(0.0, 2 * np.pi),
                              orientation='horizontal')
    ax.set_axis_off()

    ax = fig.add_subplot(spec[1, 0])
    ax.imshow(masked_flow_rgb)
    draw_velocity_vectors(ax, xy, velocity, field_vector, masked_flow_rgb.shape[1], magnitude_scale)

    # Adding Flow colormap
    ax = fig.add_subplot(spec[1, 1], polar=True)
    mpl.colorbar.ColorbarBase(ax,
                              cmap=flow_colormap(),
                              norm=mpl.colors.Normalize(0.0, 2 * np.pi),
                              orientation='horizontal')
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def plot_masked_flow_and_rgb(masked_flow_rgb, rgb, xy, velocity, field_vector, title=None, magnitude_scale=0.0025):
    fig = plt.figure(figsize=(16, 15))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[9, 1])

    if title:
        fig.suptitle(title)

    ax = fig.add_subplot(spec[0, 0])
    ax.imshow(masked_flow_rgb)
    draw_velocity_vectors(ax, xy, velocity, field_vector, masked_flow_rgb.shape[1], magnitude_scale)

    # Adding Flow colormap
    ax = fig.add_subplot(spec[0, 1], polar=True)
    mpl.colorbar.ColorbarBase(ax,
                              cmap=flow_colormap(),
                              norm=mpl.colors.Normalize(0.0, 2 * np.pi),
                              orientation='horizontal')
    ax.set_axis_off()

    ax = fig.add_subplot(spec[1, 0])
    ax.imshow(rgb)
    draw_velocity_vectors(ax, xy, velocity, field_vector, rgb.shape[1], magnitude_scale)

    fig.tight_layout()
    return fig, ax
