"""Manuscript version of the existing winter/summer event-validation figure.

Read archived CSVs only. The upper panels retain the original two selected call
windows and simulated 15-minute delivery; lower panels retain the 15/20-record
capacity comparison and independently evaluated annual risk. Plot dimensions,
colors, label positions and rounding are author-chosen presentation settings.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "data"
KEYS = ["正态1", "LSE A", "LSE B", "LSE C"]
CAP_KEYS = ["正态参照", "LSE A", "LSE B", "LSE C"]
LABELS = ["Normal", "LSE A", "LSE B", "LSE C"]
# Same representative dates as plot_experiment_four_main.py and its original figure.
DAYS = ["2022-12-24", "2025-06-24"]
STAGES = ["十五次历史", "二十次历史"]
WIDTH, HEIGHT = 6.2, 7.6  # Author-chosen manuscript dimensions in inches.
INK, GRAY, TEAL = "#272727", "#767676", "#328578"
LIGHT = "#B9C5CC"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
})


def time_label(hours):
    minutes = round(float(hours) * 60)  # Convert source clock-hours to minutes.
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def draw():
    process = pd.read_csv(SOURCE / "Event_Plot_Paths.csv")
    events = pd.read_csv(SOURCE / "Event_Plot_Summary.csv")
    capacity = pd.read_csv(SOURCE / "D21.csv").set_index("客户")
    risks = pd.read_csv(SOURCE / "D22.csv")
    budget = float(risks["风险预算_兆瓦时"].iloc[0])
    duration = float(events["真实持续能力_小时"].iloc[0])
    nomination = float(process["指令功率_兆瓦"].iloc[0])

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    fig.text(.035, .972, "Simulated customers in historical call windows", fontsize=11, weight="bold")
    fig.text(.035, .942,
             f"Four customer types; true duration {duration:g} h; nomination {nomination:,.0f} MW")
    fig.text(.035, .900, "a  Simulated delivery during winter and summer calls", fontsize=10, weight="bold")

    # Original color scale: 100% means meeting instruction; 50--150% covers these data.
    cmap = LinearSegmentedColormap.from_list(
        "delivery", ["#3D6C95", "#D7E5EF", "#FBFBFA", "#EADFCB", "#A97E48"])
    norm = TwoSlopeNorm(vmin=50, vcenter=100, vmax=150)
    # Author-chosen axes and annotation locations; common customer ordering in both panels.
    layouts = [(.200, .290, .545), (.630, .280, .955)]
    # Clock ticks preserve source-local times; these are display choices, not sampled times.
    ticks = [[6, 10, 14, 20.5], [15, 18, 20, 22]]
    for index, (day, (left, width, gap_x)) in enumerate(zip(DAYS, layouts)):
        sample = process[process["日期"] == day]
        times = np.sort(sample["时段开始_小时"].unique())
        table = sample.pivot(index="客户", columns="时段开始_小时",
                             values="实际调节占要求_百分比").reindex(index=KEYS, columns=times)
        event = events[(events["日期"] == day) & (events["客户"] == KEYS[0])].iloc[0]
        start, end = float(event["调用起点_小时"]), float(event["调用终点_小时"])
        # Actual interval edges retain the last interval and its source duration.
        edges = np.r_[times, end]
        if table.isna().any().any():
            raise ValueError(f"Missing event interval in {day}")
        if table.to_numpy().min() < norm.vmin or table.to_numpy().max() > norm.vmax:
            raise ValueError(f"Delivery outside the original color scale in {day}")
        ax = fig.add_axes([left, .695, width, .118])
        im = ax.pcolormesh(edges, np.arange(len(KEYS) + 1), table.to_numpy(),
                           cmap=cmap, norm=norm, shading="flat", rasterized=False)
        ax.set_ylim(len(KEYS), 0)
        ax.set_yticks(np.arange(len(KEYS)) + .5, LABELS if index == 0 else [])
        ax.set_xticks(ticks[index], [time_label(t) for t in ticks[index]])
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", length=2, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(1, len(KEYS)):
            ax.axhline(row, color="white", lw=.8)
        fig.text(left, .859, day, weight="bold")
        fig.text(left, .837, f"{time_label(start)}–{time_label(end)} ({end-start:g} h)")
        fig.text(gap_x, .823, "Gap\n(%)", ha="center", va="bottom", color=GRAY)
        for row, key in enumerate(KEYS):
            gap = float(events[(events["日期"] == day) & (events["客户"] == key)]
                        ["逐时不足占要求_百分比"].iloc[0])
            ax.text((gap_x-left)/width, row+.5, f"{gap:.1f}",
                    transform=ax.get_yaxis_transform(), ha="center", va="center")

    cax = fig.add_axes([.200, .633, .710, .015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([50, 75, 100, 125, 150])  # Original percentage scale, displayed at equal steps.
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, pad=3)
    cb.ax.axvline(100, color=INK, lw=.8)
    fig.text(.200, .584, "Simulated delivery / instruction (%)")
    fig.text(.035, .555, "Gap: positive interval shortfalls / instructed energy.", color=GRAY)

    handles = [
        Line2D([0], [0], color=LIGHT, marker="o", mfc="white", mec=GRAY,
               lw=1.2, markersize=5, label="15 records (five dates held out)"),
        Line2D([0], [0], color=LIGHT, marker="o", mfc=TEAL, mec="white",
               lw=1.2, markersize=5, label="20 records (five dates restored)"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.535, .508),
               ncol=1, handlelength=1.6, labelspacing=.45)
    fig.text(.035, .455, "b  Net accredited capacity", fontsize=10, weight="bold")
    fig.text(.630, .455, "c  Annual tail risk", fontsize=10, weight="bold")
    axb = fig.add_axes([.200, .214, .290, .205])
    axr = fig.add_axes([.660, .214, .280, .205])
    fig.text(.557, .425, "Gain\n(MW)", ha="center", color=GRAY)
    for row, (key, cap_key) in enumerate(zip(KEYS, CAP_KEYS)):
        y = len(KEYS) - 1 - row
        item = capacity.loc[cap_key]
        # Original figure plots certified lower bounds, matching the independently tested awards.
        caps = [float(item["十五次认证下界_兆瓦"]), float(item["二十次认证下界_兆瓦"])]
        normalized_risk = []
        for stage, award in zip(STAGES, caps):
            test = risks[(risks["客户"] == key) & (risks["资料"] == stage)].iloc[0]
            if not np.isclose(float(test["核定净容量_兆瓦"]), award):
                raise ValueError("Risk evaluation does not match the displayed award")
            # Source risk is evaluated at 系统总增载边界_兆瓦 = C0 + Ar, not at Ar.
            normalized_risk.append(100 * float(test["独立年度风险_兆瓦时"]) / budget)
        for ax, values in [(axb, caps), (axr, normalized_risk)]:
            ax.axhline(y, color="#ECEFF1", lw=.6, zorder=0)
            ax.plot(values, [y, y], color=LIGHT, lw=1.6, zorder=1)
            ax.scatter(values[0], y, s=31, facecolors="white", edgecolors=GRAY,
                       linewidths=1.1, zorder=3, clip_on=False)
            ax.scatter(values[1], y, s=25, color=TEAL, edgecolors="white",
                       linewidths=.6, zorder=4, clip_on=False)
        axb.text(caps[1]+8, y, f"{caps[1]:.1f}", va="center", color=TEAL)
        axb.text((.557-.200)/.290, y, f"+{caps[1]-caps[0]:.1f}",
                 transform=axb.get_yaxis_transform(), ha="center", va="center", color=TEAL)
        # Place the high-risk label left of its marker so it does not cross the budget line.
        high_label = normalized_risk[1] > 75  # Author-chosen label-placement cutoff, percent.
        label_x = normalized_risk[1] - 5 if high_label else normalized_risk[1] + 2.5
        axr.text(label_x, y, f"{normalized_risk[1]:.1f}%", va="center",
                 ha="right" if high_label else "left", color=TEAL,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": .2})
    # Author-chosen display ranges leave room for the source-value annotations.
    axb.set(xlim=(0, 335), ylim=(-.55, 3.55))
    axb.set_yticks([3, 2, 1, 0], LABELS)
    axb.set_xticks([0, 100, 200, 300])
    axb.set_xlabel("Net accredited capacity\n$A_r$ (MW)", labelpad=7)
    axr.set(xlim=(0, 120), ylim=(-.55, 3.55))
    axr.set_yticks([])
    axr.set_xticks([0, 50, 100])
    axr.axvline(100, color=INK, lw=.9, linestyle=(0, (3, 3)), zorder=1)
    axr.text(100, 3.64, "Budget", ha="center", va="bottom")
    axr.set_xlabel("CVaR / budget (%)\nAt total boundary\n$q=C_0+A_r$", labelpad=7)
    for ax in (axb, axr):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", length=3, pad=4)
    fig.text(.035, .078, "Independent future power; unchanged annual system scenarios.", color=GRAY)
    fig.text(.035, .049, f"Annual risk budget: {budget:,.0f} MWh; CVaR level: 95%.", color=GRAY)
    # The 95% CVaR level is the original experiment's tail_level=0.95; no risk is recomputed.
    fig.savefig(HERE / "fig05_event_validation.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
