"""Reproduce the risk and historical coverage comparison from D27.

The layout follows Figure 7 of the results report. Plotted positions are
midpoints of the archived numerical bounds, expressed as a share of nomination.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "data"
CUSTOMERS = [
    ("正态参照", "Normal"),
    ("LSE A", "LSE A"),
    ("LSE B", "LSE B"),
    ("LSE C", "LSE C"),
]
# D27's two one-factor comparisons use the same central setting.
PANELS = [
    ("a", "Annual CVaR level", "historical coverage fixed at 95%",
     ("tail_0.9", "central", "tail_0.99")),
    ("b", "Historical coverage target", "annual CVaR level fixed at 95%",
     ("coverage_0.9", "central", "coverage_0.99")),
]
LEVELS = ("90%", "95%", "99%")
GRAY, GREEN, GOLD = "#626C76", "#287A70", "#A17934"


def load_results(directory):
    with (directory / "D27.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {(r["客户"], r["情景"]): r for r in csv.DictReader(handle)}
    settings = json.loads((directory / "Sensitivity_Parameters.json").read_text(encoding="utf-8-sig"))
    return rows, settings["parameters"]["nomination_mw"]


def bounds_as_percent(row, nomination):
    return [
        tuple(100 * float(row[f"{kind}{end}_兆瓦"]) / nomination for end in ("下界", "上界"))
        for kind in ("五次容量", "二十次容量", "完整信息")
    ]


def make_figure(rows, nomination):
    # Figure dimensions, spacing and point sizes are presentation settings for
    # a plot printed at the manuscript text width.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.linewidth": 0.65,
        "text.color": "#272727",
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 3.92), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.172, right=0.982, bottom=0.19, top=0.84, wspace=0.13, hspace=0.72)

    for row_idx, (letter, title, fixed, settings) in enumerate(PANELS):
        title_y = (0.97, 0.525)[row_idx]
        fig.text(0.02, title_y, letter, fontsize=10, weight="bold", va="top")
        fig.text(0.063, title_y, title, fontsize=9.2, weight="bold", va="top")
        fig.text(0.98, title_y, fixed, fontsize=8, ha="right", va="top", color=GRAY)
        for col, (setting, level) in enumerate(zip(settings, LEVELS)):
            ax = axes[row_idx, col]
            for y, (customer, _) in enumerate(reversed(CUSTOMERS)):
                bounds = bounds_as_percent(rows[customer, setting], nomination)
                mid = [(lo + hi) / 2 for lo, hi in bounds]
                ax.axhline(y, color="#E3E7EA", lw=0.55, zorder=0)
                ax.plot(mid[:2], [y, y], color=GREEN, lw=1.3, zorder=1)
                ax.plot(mid[1:], [y, y], color=GRAY, lw=0.8, ls=(0, (2, 2)), zorder=1)
                for j, (lo, hi) in enumerate(bounds):
                    ax.hlines(y, lo, hi, color=(GRAY, GREEN, GOLD)[j], lw=0.6, zorder=2)
                ax.plot(mid[0], y, "o", mfc="white", mec=GRAY, ms=4, zorder=3)
                ax.plot(mid[1], y, "o", mfc=GREEN, mec="white", mew=0.45, ms=4.7, zorder=4)
                ax.plot(mid[2], y, "|", color=GOLD, ms=8, mew=1.2, zorder=5)
                ax.annotate(
                    f"{mid[1]:.1f}", (mid[1], y), xytext=(0, 4.2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=GREEN,
                )
            ax.set(xlim=(0, 32), ylim=(-0.40, 3.58))
            ax.set_xticks([0, 10, 20, 30])
            ax.set_yticks(range(len(CUSTOMERS)), [label for _, label in reversed(CUSTOMERS)])
            ax.tick_params(axis="x", length=2.6, pad=3, labelbottom=True)
            ax.tick_params(axis="y", length=0, pad=4, labelleft=col == 0)
            ax.set_title(level, pad=4, fontsize=8.8, weight="bold")

    fig.text(0.577, 0.096, "Net accredited allowance (% of nomination)", ha="center", fontsize=8.5)
    handles = [
        Line2D([], [], marker="o", mfc="white", mec=GRAY, lw=0, ms=4, label="5 records"),
        Line2D([], [], marker="o", mfc=GREEN, mec="white", lw=0, ms=4.7, label="20 records"),
        Line2D([], [], marker="|", color=GOLD, lw=0, ms=8, mew=1.2, label="Full information"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.577, 0.005))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=HERE / "figB02_risk_coverage.pdf")
    args = parser.parse_args()
    fig = make_figure(*load_results(args.results_dir))
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
