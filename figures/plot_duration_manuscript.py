"""Duration figure from archived CSVs; no experiment is run.

All plotted quantities and computational bounds come from the source columns
listed in each function. Midpoints are display coordinates, not new estimates.
Figure sizes, colors, axis margins, annotation positions and rounding are
author-chosen presentation settings. PDF width is exactly 6.2 inches.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "data"
BLUE, GREEN, RED, GRAY = "#0F4D92", "#328578", "#B64342", "#767676"
LIGHT, INK = "#D9DFE3", "#272727"
WIDTH = 6.2  # Author-chosen full-text figure width, inches.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": .7,
    "lines.linewidth": 1.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def read(relative):
    return pd.read_csv(RESULTS / relative)


def interval(frame, lower, upper):
    lo, hi = frame[lower].to_numpy(float), frame[upper].to_numpy(float)
    if np.any(lo > hi):
        raise ValueError(f"Reversed computational bounds: {lower}, {upper}")
    return lo, hi, (lo + hi) / 2


def style(ax, axis="y"):
    ax.grid(axis=axis, color="#E7EAED", linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7)


def bounds_line(ax, x, lo, hi, color, label, linestyle="-", marker=None):
    mid = (lo + hi) / 2
    ax.fill_between(x, lo, hi, color=color, alpha=.16, linewidth=0)
    ax.plot(x, mid, color=color, label=label, linestyle=linestyle,
            marker=marker, markersize=4, markerfacecolor="white")


def save(fig, stem):
    # All PDF text must remain at least 9 pt at the declared 6.2-inch width.
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text() and artist.get_fontsize() < 9:
            raise ValueError(f"Small figure text: {artist.get_text()}")
    fig.savefig(HERE / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def figure_duration():
    data = read("D30.csv")
    customers = [("正态参照", "Normal"), ("LSE A", "LSE A"),
                 ("LSE B", "LSE B"), ("LSE C", "LSE C")]
    settings = [("五次容量下界_兆瓦", "五次容量上界_兆瓦", "5 records", BLUE, "-"),
                ("二十次容量下界_兆瓦", "二十次容量上界_兆瓦", "20 records", GREEN, "-"),
                ("完整信息下界_兆瓦", "完整信息上界_兆瓦", "Complete information", RED, "--")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.135, top=.86, hspace=.32, wspace=.17)
    for ax, (source, name), letter in zip(axes.flat, customers, "abcd"):
        rows = data[data["客户"] == source].sort_values("真实持续能力_小时")
        x = rows["真实持续能力_小时"].to_numpy()
        for lower, upper, label, color, line in settings:
            lo, hi, _ = interval(rows, lower, upper)
            bounds_line(ax, x, lo, hi, color, label, line)
        ax.set_title(f"{letter}  {name}", loc="left", weight="bold", pad=8)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(bottom=0)
        ax.set_xticks([4, 8, 12, 14.5])  # Display ticks within the 22 archived duration settings.
        ax.set_xticklabels(["4", "8", "12", "14.5"])
        style(ax)
    handles = [Line2D([], [], color=c, linestyle=ls, label=label)
               for _, _, label, c, ls in settings]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .98),
               ncol=3, handlelength=2, columnspacing=1.2)
    fig.supylabel("Net accredited capacity (MW)", x=.015, fontsize=9)
    fig.supxlabel("True maximum response duration (h)", y=.035, fontsize=9)
    save(fig, "fig04_duration")


if __name__ == "__main__":
    figure_duration()
