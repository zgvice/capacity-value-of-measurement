"""Restore the four-panel Experiment 1 figure in English from saved results.

The panel order, source selection, capability colors, annual risk curves and
month-by-hour accounting follow code/core/plot_experiment_one.py.
This script reads archived CSV/JSON results and never runs an experiment.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
SOURCE = REVISION / "results" / "exp1"
DISPLAY = SOURCE / "plot"
AUDIT = REVISION / "checks" / "exp1_check.json"

# Original figure's semantic colors: continuation for 0, 1, 2 and 4 hours.
COLORS = ["#7B858E", "#79A9A3", "#4E779D", "#8A688D"]
INK, GRAY, GOLD = "#293744", "#75808A", "#9A7247"
WIDTH, HEIGHT, DPI = 6.2, 7.0, 350  # Requested manuscript size; preview resolution.

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
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def draw():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    parameters = audit["parameters"]
    cap = pd.read_csv(DISPLAY / "capacity_comparison.csv").sort_values("最多续调_小时")
    curves = pd.read_csv(DISPLAY / "annual_risk_curve.csv")
    history = pd.read_csv(SOURCE / "shared_records.csv")
    future = pd.read_csv(DISPLAY / "call_trace.csv")
    hourly = pd.read_csv(DISPLAY / "shortfall_timing.csv")

    extensions = cap["最多续调_小时"].to_numpy(float)
    budget = float(cap["缺供预算_兆瓦时"].iloc[0])
    reference = float(cap["无资源增载_兆瓦"].iloc[0])
    gain = float(cap["净增加_兆瓦"].iloc[-1] - cap["净增加_兆瓦"].iloc[0])
    q = float(hourly["增载_兆瓦"].iloc[0])
    start_hour = parameters["diagnostic_start_hour"]
    end_hour = parameters["diagnostic_end_hour"]

    # Original accounting: common total load q and the zero-continuation tail
    # weights, applied to the shortfall saving from zero to four extra hours.
    total_saving = float(hourly["续调少缺供_兆瓦时"].sum())
    evening = (hourly["小时"] >= start_hour) & (hourly["小时"] < end_hour)
    share = float(hourly.loc[evening, "续调少缺供_兆瓦时"].sum() / total_saving)
    tolerance = parameters["accounting_tolerance_mwh"]
    if not np.isclose(total_saving, audit["common_tail_saving_MWh"], atol=tolerance, rtol=0):
        raise ValueError("Hourly accounting does not match the saved common-tail total")
    if not np.isclose(share, audit["diagnostic_interval_share"], atol=1e-12, rtol=0):
        # 1e-12 is a numerical comparison tolerance for the same saved sums.
        raise ValueError("Evening share does not match the saved common-tail result")

    # Same month order as the original figure: June to the following May.
    month_order = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
    month_labels = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
    hourly["month"] = hourly["月日"].str[:2].astype(int)
    matrix = hourly.pivot_table(index="month", columns="小时", values="续调少缺供_兆瓦时", aggfunc="sum")
    matrix = matrix.reindex(index=month_order, columns=range(24)).fillna(0)
    if not np.isclose(matrix.to_numpy().sum(), total_saving, atol=tolerance, rtol=0):
        raise ValueError("Month-hour heat map does not preserve the saved total")

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    # Author-chosen panel placement preserves the original two-by-two structure.
    axh = fig.add_axes([.125, .695, .355, .215])
    axt = fig.add_axes([.625, .695, .345, .215])
    axr = fig.add_axes([.125, .255, .355, .270])
    heat = fig.add_axes([.625, .255, .345, .270])

    # (a) Select the historical call with the most readings, as in the source.
    stamp = pd.to_datetime(history["读数时段开始"])
    dates = stamp.dt.strftime("%Y-%m-%d")
    day = dates.value_counts().idxmax()
    record = history.loc[dates == day]
    times = stamp.loc[dates == day]
    clock = times.dt.hour.to_numpy() + times.dt.minute.to_numpy() / 60
    interval = float(np.median(np.diff(clock)))  # Sampling interval in archived timestamps.
    stop = float(clock[-1] + interval)
    for field, color, label in [("基线_兆瓦", "#596873", "Baseline"), ("电表_兆瓦", "#B68A52", "Meter")]:
        values = record[field].to_numpy()
        axh.step(np.r_[clock, stop], np.r_[values, values[-1]], where="post", color=color, lw=1.15, label=label)
    axh.axvline(stop, color=GRAY, lw=.7, ls=(0, (3, 3)))
    axh.set(xlim=(clock[0], stop), xlabel="Clock time on 24 Dec 2022", ylabel="Consumption (MW)")
    ticks = np.linspace(clock[0], stop, 4)  # Original four evenly spaced clock ticks.
    axh.set_xticks(ticks, [f"{int(t):02d}:{int(round((t % 1) * 60)):02d}" for t in ticks])
    axh.set_title("a  Shared simulated records", loc="left", pad=17, weight="bold")
    axh.legend(loc="lower left", ncol=2, handlelength=1.3, columnspacing=.8, borderaxespad=.35)
    axh.grid(axis="y", color="#E7EBEF", lw=.5)
    axh.tick_params(length=2.5)
    fig.text(.125, .610, "20 simulated events, 544 readings.\nTrue delivery: 500 MW throughout.", fontsize=9, color=GRAY, va="top", linespacing=1.3)

    # (b) Existing representative future call; no new instruction is constructed.
    original = float(future["原邀约结束_小时"].iloc[0] - future["原邀约开始_小时"].iloc[0])
    allowed = float(future["延续邀约结束_小时"].iloc[0] - future["原邀约开始_小时"].iloc[0])
    for i, extension in enumerate(extensions):
        row = len(extensions) - 1 - i
        axt.barh(row, original, height=.44, color="#DAE1E6", edgecolor="none")
        if extension:
            axt.barh(row, min(extension, allowed - original), left=original, height=.44, color=COLORS[i], edgecolor="none")
    axt.axvline(original, color=GRAY, lw=.75, ls=(0, (3, 3)))
    axt.set_yticks([3, 2, 1, 0], [f"+{h:g} h" for h in extensions])
    axt.set(xlim=(0, allowed), ylim=(-.65, 3.65), xlabel="Hours since start of future event")
    axt.set_xticks([0, original / 2, original, allowed])
    axt.set_title("b  Continuation capability", loc="left", pad=17, weight="bold")
    axt.text(original, 3.77, "Planned end", ha="center", va="bottom", fontsize=9, color=GRAY)
    axt.spines["left"].set_visible(False)
    axt.tick_params(axis="y", length=0)
    axt.tick_params(axis="x", length=2.5)
    fig.text(.625, .610, "Gray: scheduled event.\nColor: continued response at 500 MW.", fontsize=9, color=GRAY, va="top", linespacing=1.3)

    # (c) Existing full-year risk evaluations, including the saved capacity roots.
    for h, color in zip(extensions, COLORS):
        part = curves.loc[curves["最多续调_小时"] == h].sort_values("资源净容量_兆瓦")
        axr.plot(part["资源净容量_兆瓦"], part["全年新增缺供风险_兆瓦时"], color=color, lw=1.6, label=f"+{h:g} h")
    axr.axhline(budget, color=INK, lw=.85, ls=(0, (4, 3)))
    axr.text(.99, budget + 330, f"Budget {budget:,.0f}", transform=axr.get_yaxis_transform(), ha="right", va="bottom", fontsize=9)
    # Original alternating annotation heights and offsets, solely for readability.
    annotation_y = [11500, 5800, 11500, 5800]
    annotation_dx = [-12, 0, 0, 0]
    for i, row in enumerate(cap.to_dict("records")):
        x = row["净增加_兆瓦"]
        axr.scatter(x, budget, s=24, color=COLORS[i], edgecolor="white", lw=.7, zorder=5)
        axr.annotate(f"{x:.2f}", xy=(x, budget), xytext=(x + annotation_dx[i], annotation_y[i]), ha="center", va="center", fontsize=9.5, weight="bold", color=COLORS[i], arrowprops={"arrowstyle": "-", "color": COLORS[i], "lw": .75})
    axr.set(xlim=(0, curves["资源净容量_兆瓦"].max()), ylim=(0, 2 * budget), xlabel="Added load above $C_0$ (MW)", ylabel="Annual CVaR of added\nunserved energy (MWh)")
    axr.set_xticks([0, 50, 100, 150, 200])  # Same axis tick locations as the source figure.
    axr.set_yticks([0, 4000, 8000, 12000, 16000])  # Presentation spacing in original MWh units.
    axr.legend(loc="upper left", ncol=2, handlelength=1.35, columnspacing=.8, borderaxespad=.3)
    axr.grid(axis="y", color="#E7EBEF", lw=.5)
    axr.tick_params(length=2.5)
    axr.set_title("c  Annual risk and capacity", loc="left", pad=15, weight="bold")
    fig.text(.125, .153, "Capacity gain: 0 h to 4 h", fontsize=9, color=GRAY)
    fig.text(.125, .116, f"{gain:.2f} MW", fontsize=13, color=COLORS[-1], weight="bold")

    # (d) Original month-by-clock-hour decomposition, with common tail weights.
    cmap = LinearSegmentedColormap.from_list("annual_saving", ["#FAFBFC", "#DCCFE2", "#98779D", "#593B65"])
    image = heat.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=0, interpolation="nearest")
    heat.set_yticks(range(12), month_labels)
    heat.set_xticks([0, 6, 12, 18, 23])
    heat.set_xlabel("Hour of day")
    heat.tick_params(length=0)
    heat.set_title("d  Timing of avoided shortfall", loc="left", pad=15, weight="bold")
    heat.set_yticks(np.arange(-.5, 12, 1), minor=True)
    heat.tick_params(which="minor", length=0)
    heat.grid(which="minor", axis="y", color="white", lw=.3)
    for spine in heat.spines.values():
        spine.set_visible(False)
    # The saved diagnostic window is [17:00,21:00): bins 17,18,19 and 20.
    heat.add_patch(Rectangle((start_hour - .5, -.5), end_hour - start_hour, 12, fill=False, edgecolor="#A77D4D", lw=1.0))
    cbax = fig.add_axes([.625, .155, .345, .013])
    bar = fig.colorbar(image, cax=cbax, orientation="horizontal")
    bar.set_ticks([0, 500, 1000, 1500])  # Readable ticks on the original MWh scale.
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)
    bar.set_label("Avoided shortfall (MWh)", labelpad=3)
    fig.text(.625, .075, f"17:00–21:00: {share:.1%}", fontsize=10, color=GOLD, weight="bold")

    fig.text(.07, .037, f"40,300 annual scenarios; net capacity subtracts the {reference:.2f} MW reference.", fontsize=9, color=GRAY)
    fig.text(.07, .013, f"Heat map: total load addition {q:.2f} MW; common 0 h tail weights.", fontsize=9, color=GRAY)

    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text() and text.get_fontsize() < 9:
            raise ValueError(f"Figure text below 9 pt: {text.get_text()}")
    for extension in ("pdf", "png"):
        fig.savefig(HERE / f"fig02_records.{extension}", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved four-panel fig02_records at {WIDTH} × {HEIGHT} in; minimum font 9 pt")
    print(f"Capacity gap = {gain:.8f} MW; common-tail saving = {total_saving:.8f} MWh")
    print(f"17:00–21:00 share = {share:.10%}; total load addition = {q:.8f} MW")


if __name__ == "__main__":
    draw()
