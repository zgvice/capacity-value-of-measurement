"""Draw the appendix comparison from the archived portfolio results (D13)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


FIGURES = Path(__file__).resolve().parent
SOURCE = FIGURES.parent / "data" / "D13.csv"
OUTPUT = FIGURES / "figB01_empirical_aggregation.pdf"

# Colors and line types follow the archived English portfolio figure.
SINGLES = {
    "LSE A": ("LSE A", "#446E9A", "o"),
    "LSE B": ("LSE B", "#4C9691", "s"),
    "LSE C": ("LSE C", "#B18D57", "^"),
}
PORTFOLIO = ("Portfolio", "#8A688D", "D")
PANELS = (
    ("A+B", ("LSE A", "LSE B")),
    ("A+C", ("LSE A", "LSE C")),
    ("B+C", ("LSE B", "LSE C")),
    ("A+B+C", tuple(SINGLES)),
)


def draw_series(ax, rows, color, marker, *, portfolio):
    x = rows["每户历史次数"].to_numpy()
    low = rows["认证下界_兆瓦"].to_numpy()
    high = rows["认证上界_兆瓦"].to_numpy()
    # Line weights, marker sizes, and shading are figure design choices.
    ax.fill_between(x, low, high, step="post", color=color, alpha=0.20, linewidth=0)
    ax.plot(
        x, (low + high) / 2, drawstyle="steps-post", color=color,
        linestyle="-" if portfolio else "--", linewidth=2.1 if portfolio else 1.0,
        alpha=0.85 if portfolio else 1.0,
        marker=marker, markersize=3.2 if portfolio else 2.6,
        markevery=(0, 2) if portfolio else (1, 2),
        markerfacecolor="white", markeredgewidth=0.7,
    )


def main():
    results = pd.read_csv(SOURCE)
    # The archived comparison reports one through twenty records per member.
    results = results.loc[results["每户历史次数"].between(1, 20)]
    groups = {name: rows.sort_values("每户历史次数") for name, rows in results.groupby("组合")}

    # Match the English manuscript's sans-serif figures at 6.2 inches.
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 9,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "text.color": "#293744", "axes.labelcolor": "#293744",
        "xtick.color": "#52606B", "ytick.color": "#52606B",
        "legend.frameon": False, "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.2, 4.6))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.89, wspace=0.15, hspace=0.30)
    for letter, (title, members), ax in zip("abcd", PANELS, axes.flat):
        _, color, marker = PORTFOLIO
        draw_series(ax, groups[title], color, marker, portfolio=True)
        for member in members:
            _, color, marker = SINGLES[member]
            draw_series(ax, groups[member], color, marker, portfolio=False)
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=7)
        # Tick spacing and 300 MW ceiling give all panels a common display scale.
        ax.set(xlim=(0.6, 20.4), ylim=(0, 300), xticks=[1, 5, 10, 15, 20],
               yticks=[0, 100, 200, 300])
        ax.tick_params(length=2.5)
        ax.grid(axis="y", color="#E6EBEF", linewidth=0.6)
        ax.set_axisbelow(True)

    handles = [
        Line2D([], [], label=label, color=color, marker=marker, markerfacecolor="white",
               markersize=3.8, linestyle="--", linewidth=1.2)
        for label, color, marker in SINGLES.values()
    ]
    label, color, marker = PORTFOLIO
    handles.append(Line2D([], [], label=label, color=color, marker=marker,
                          markerfacecolor="white", markersize=3.8, linewidth=1.8))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 0.995),
               handlelength=2.1, columnspacing=1.25)
    fig.supylabel("Net accredited allowance (MW)", x=0.02, fontsize=9)
    fig.supxlabel("Complete call records per customer", x=0.55, y=0.02, fontsize=9)
    fig.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
