import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_xy(Series, xlabel, ylabel, title, save_fig=False):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left=0.25)
    Series = [pd.Series(s) for s in Series]
    ax.plot(Series[0], Series[1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if save_fig:
        plt.savefig(os.path.join("Graphs", title))
    else:
        plt.show()


def plot_duo(Series, labels, xlabel, ylabel, title, save_fig=False, path=None):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left=0.25)
    Series = [pd.Series(s) for s in Series]
    ax.plot(Series[0].index, Series[0], "b", label=labels[0])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    locs, xticks_labels = plt.xticks()
    if len(Series[0]) < 10000:
        xticks_labels = [int(int(loc) / 10) for loc in locs]
    else:
        xticks_labels = [round(int(loc) / 36000, 2) for loc in locs]
        ax.set_xlabel(xlabel[:-3] + "(h)")

    plt.xticks(locs, xticks_labels)
    ax2 = ax.twinx()
    ax2.plot(Series[1].index, Series[1], "r", label=labels[1])
    ax2.set_ylabel(ylabel)

    lines = ax.get_lines() + ax2.get_lines()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="center left",
        bbox_to_anchor=(0.25, -0.3),
    )

    ax.set_title(title)
    if save_fig:
        if path:
            plt.savefig(os.path.join(path, "Graphs", title))
        else:
            plt.savefig(os.path.join("Graphs", title))
    else:
        plt.show()


def plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=False, path=None):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left=0.25)
    Series = [pd.Series(s) for s in Series]

    for i, s in enumerate(Series):
        ax.plot(s.index, s, label=labels[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    locs, xticks_labels = plt.xticks()
    if len(Series[0]) < 10000:
        xticks_labels = [int(int(loc) / 10) for loc in locs]
    else:
        xticks_labels = [round(int(loc) / 36000, 2) for loc in locs]
        ax.set_xlabel(xlabel[:-3] + "(h)")
    plt.xticks(locs, xticks_labels)
    lines = ax.get_lines()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if len(Series) > 1:
        ax.legend(
            lines,
            [line.get_label() for line in lines],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    ax.set_title(title)
    if save_fig:
        if path:
            plt.savefig(os.path.join(path, "Graphs", title))
        else:
            plt.savefig(os.path.join("Graphs", title))
    else:
        plt.show()
