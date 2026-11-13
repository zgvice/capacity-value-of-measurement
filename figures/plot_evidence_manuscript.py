"""Evidence-rule figure from archived CSVs; no experiment is run.

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
RESULTS = HERE.parent / "results"
BLUE, GREEN, RED, GRAY = "#0F4D92", "#328578", "#B64342", "#767676"
LIGHT, INK = "#D9DFE3", "#272727"
WIDTH = 6.2  # Author-chosen full-text figure width, inches.
DPI = 350  # Reading-preview export setting; PDF remains vector.
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
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.pdf and {stem}.png")


def figure_evidence():
    folder = "exp2/plot/"
    checks = read(folder + "verification_capacity.csv")
    history = read(folder + "evidence_capacity.csv").sort_values("历史次数")
    fig = plt.figure(figsize=(WIDTH, 6.4))
    methods = fig.add_axes([.355, .735, .60, .20])
    credit = fig.add_axes([.125, .415, .83, .215])
    duration = fig.add_axes([.125, .105, .83, .215])
    fig.text(.06, .975, "a  Verification of the same 20 records", fontsize=10, weight="bold")
    names = ["Pointwise", "Pointwise + within-event", "Across events", "All joint"]
    lo, hi, mid = interval(checks, "认证下界_兆瓦", "认证上界_兆瓦")
    y = np.arange(len(checks))
    method_colors = [BLUE, BLUE, GREEN, GREEN]
    methods.barh(y, mid, height=.53, color=method_colors, alpha=.83)
    methods.errorbar(mid, y, xerr=np.vstack([mid-lo, hi-mid]), fmt="none",
                     color=INK, linewidth=.8, capsize=3)
    for yy, value in zip(y, mid):
        methods.text(value + 3, yy, f"{value:.1f}", va="center")
    methods.set_yticks(y, names)
    methods.invert_yaxis()
    methods.set(xlim=(0, 145), xlabel="Net capacity (MW)")  # Display margin above 123.9 MW.
    methods.tick_params(axis="y", length=0)
    methods.spines["left"].set_visible(False)
    style(methods, "x")

    n = history["历史次数"].to_numpy()
    lo, hi, mid = interval(history, "认证下界_兆瓦", "认证上界_兆瓦")
    credit.fill_between(n, lo, hi, step="post", color=BLUE, alpha=.18, linewidth=0)
    credit.step(n, mid, where="post", color=BLUE)
    credit.plot(n, mid, "o", color=BLUE, markersize=3, markerfacecolor="white")
    credit.set(xlim=(.5, 20.5), ylim=(0, 150), ylabel="Net capacity (MW)")
    credit.set_title("b  Capacity after each additional record", loc="left", weight="bold", pad=9)
    credit.set_xticks([1, 5, 10, 15, 20])
    credit.set_yticks([0, 50, 100, 150])
    # Records 16 and 19 are the accepted manuscript comparisons; all values come from CSV.
    for record, position in [(16, (11, 112)), (19, (18.2, 143))]:
        row = history[history["历史次数"] == record].iloc[0]
        value = (row["认证下界_兆瓦"] + row["认证上界_兆瓦"]) / 2
        gain = row["新增约值_兆瓦"]
        credit.scatter(record, value, s=31, color=RED, zorder=6)
        credit.annotate(f"Record {record}: +{gain:.2f} MW", (record, value), position,
                        ha="center", va="center", color=RED,
                        arrowprops={"arrowstyle": "-", "color": RED, "lw": .8})
        for ax in (credit, duration):
            ax.axvline(record, color=RED, alpha=.45, linestyle="--", linewidth=.8)
    style(credit)

    h = history["确认时长_小时"].to_numpy()
    duration.step(n, h, where="post", color=GREEN)
    duration.plot(n, h, "o", color=GREEN, markersize=3, markerfacecolor="white")
    for record in (16, 19):
        value = float(history.loc[history["历史次数"] == record, "确认时长_小时"].iloc[0])
        duration.scatter(record, value, color=RED, s=31, zorder=6)
    duration.set(xlim=(.5, 20.5), ylim=(0, 9), xlabel="Number of submitted event records",
                 ylabel="Minimum feasible\nduration (h)")
    duration.set_xticks([1, 5, 10, 15, 20])
    duration.set_yticks([0, 3, 6, 9])
    duration.set_title("c  Capability supported by the records", loc="left", weight="bold", pad=9)
    duration.text(20.1, h[-1] - 2, f"{h[-1]:.2f} h", ha="right", color=GREEN)
    style(duration)
    save(fig, "fig03_evidence")


if __name__ == "__main__":
    figure_evidence()
