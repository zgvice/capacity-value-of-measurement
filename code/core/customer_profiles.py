"""由原表实际功率/目标功率建立经验分布及15分钟时序关系。"""
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata
from numba import njit,prange
from common import OUT
SOURCE=OUT/'hubei/meter_series.csv'
NODES=32  # 当前等概率分位积分区间数；年度生成使用实验参数显式传入的值。
NOMINATION_MW=1000.  # 既定申报量，用于原始映射说明；实际年度功率按各实验申报量生成。

def _empirical_nodes(sorted_values, count=NODES):
    """Exact cell means of an equal-observation empirical quantile function."""
    values = np.asarray(sorted_values, float)
    n = len(values)
    cumulative = np.r_[0., np.cumsum(values)]
    positions = np.linspace(0., float(n), count + 1)
    whole = np.floor(positions).astype(int)
    fraction = positions - whole
    integrals = cumulative[whole].copy()
    interior = whole < n
    integrals[interior] += fraction[interior] * values[whole[interior]]
    return count * np.diff(integrals / n)


def _empirical_profile(group):
    raw = pd.read_csv(SOURCE, dtype={'event': str})
    g = raw[raw.user == group].sort_values(['event', 'minute']).reset_index(drop=True)
    if g.empty or (g.target_kW <= 0).any() or (g.actual_kW <= 0).any():
        raise ValueError(f'Positive actual and reference power required: {group}')
    shape = (g.actual_kW / g.target_kW).to_numpy(float)
    if not np.allclose(shape, 1+g.target_error_fraction.to_numpy(), rtol=1e-12, atol=1e-12):
        raise AssertionError('Actual/Target identity')
    starts = g.event.ne(g.event.shift()) | g.minute.diff().ne(15)
    blocks = starts.cumsum().astype(int)
    adjacent = ~starts.to_numpy()[1:]
    scores = norm.ppf((rankdata(shape, method='average')-.5)/len(shape))
    rho = float(np.corrcoef(scores[:-1][adjacent], scores[1:][adjacent])[0, 1])
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError(f'Nonstationary or undefined fitted rank correlation: {group}')
    mean = float(shape.mean())
    values, counts = np.unique(shape, return_counts=True)
    nodes = _empirical_nodes(np.sort(shape))
    centered = nodes-mean
    response = (g.actual_kW-g.baseline_kW)/g.nomination_kW
    source_rates = dict(pointwise_mean=float(response.mean()),
        nomination_weighted_mean=float((g.actual_kW-g.baseline_kW).sum()/g.nomination_kW.sum()),
        definition='source increase-load metered response / nomination; distinct from Actual/Target')
    profile = dict(group=group, kind='empirical', records=len(g),
        events=int(g.event.nunique()), event_blocks=int(blocks.nunique()),
        mean_shape=mean, SD_shape=float(shape.std(ddof=0)),
        min_shape=float(shape.min()), max_shape=float(shape.max()),
        adjacent_pairs=int(adjacent.sum()), gaussian_rank_AR1=rho,
        marginal=dict(support=values.tolist(), probabilities=(counts/len(g)).tolist()),
        true_a_MW=NOMINATION_MW*mean, true_b_MW=NOMINATION_MW,
        true_power_mean_MW=NOMINATION_MW*mean,
        true_power_SD_MW=NOMINATION_MW*float(shape.std(ddof=0)),
        true_power_min_MW=NOMINATION_MW*float(shape.min()),
        true_power_max_MW=NOMINATION_MW*float(shape.max()),
        quantile_cell_mean_shapes=nodes.tolist(), quantile_cell_mean_z=centered.tolist(),
        nodes_mean_z=float(centered.mean()), nodes_SD_z=float(centered.std(ddof=0)),
        source_response_rates=source_rates)
    mapped = pd.DataFrame(dict(group=group, event=g.event, minute=g.minute, time=g.time,
        event_block=blocks, source=g.source, sheet=g.sheet, column=g.column,
        source_baseline_kW=g.baseline_kW, source_nomination_kW=g.nomination_kW,
        source_target_kW=g.target_kW, source_actual_kW=g.actual_kW,
        operating_shape=shape, observable_condition_z=shape-mean,
        virtual_available_power_MW=NOMINATION_MW*shape,
        source_metered_response_ratio=response))
    return profile, mapped


@njit(parallel=True, cache=True)
def _ar_scores(innovations, rho):
    n, length = innovations.shape
    out = np.empty_like(innovations)
    factor = np.sqrt(1-rho*rho)
    for i in prange(n):
        out[i, 0] = innovations[i, 0]
        for t in range(1, length):
            out[i, t] = rho*out[i, t-1]+factor*innovations[i, t]
    return out


@njit(parallel=True, cache=True)
def _reorder_nodes(scores, nodes):
    n, length = scores.shape
    out = np.empty_like(scores)
    repeats = n//len(nodes)
    for t in prange(length):
        ordering = np.argsort(scores[:, t])
        for rank in range(n):
            out[ordering[rank], t] = nodes[rank//repeats]
    return out

