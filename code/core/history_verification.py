"""历史误差的联合覆盖边界及调用内外缺供分账。"""
import math,time
import numpy as np
from scipy.linalg import cholesky,toeplitz
from threadpoolctl import threadpool_limits
from numba import njit,prange
from common import CORES
from customer_history_inputs import _load_draw

def say(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def joint_bounds(c, public, meter):
    F, W = public['error_operator'], public['W']
    # 原15分钟误差生成：每天共同扰动与日内相关扰动；96为每日15分钟点数。
    day_size = 96
    daily = c.load_daily_variance_share + (1-c.load_daily_variance_share)*toeplitz(c.load_rho15**np.arange(day_size))
    covariance = np.zeros((public['n'], public['n']))
    for day in np.unique(F.indices//day_size):
        part = F[:, day*day_size:(day+1)*day_size].tocsr()
        ids = np.unique(part.nonzero()[0])
        block = part[ids].toarray()
        covariance[np.ix_(ids, ids)] += block @ daily @ block.T
    factor = cholesky(covariance, lower=True)
    operator = np.asarray(W @ factor)
    statistic_sd = np.sqrt(np.einsum('ij,ij->i', operator, operator))
    measured_sd = statistic_sd*public['error_scale_MW']
    discrepancy = float(np.max(np.abs(measured_sd-public['error_stat_SD_MW'])))
    if not np.allclose(measured_sd, public['error_stat_SD_MW'], rtol=c.covariance_relative_tolerance, atol=c.solver_tolerance):
        raise RuntimeError('联合核验与原核验使用的误差协方差不一致')
    operator /= statistic_sd[:, None]
    families = public['error_families']
    masks = {'global': families == 'global', 'other': families != 'global'}
    maxima = {name: np.empty(c.calibration_draws) for name in masks}
    rng = np.random.default_rng(c.calibration_seed)
    say(f'联合边界：{operator.shape[0]}项统计，使用全部{CORES}个核心校准')
    with threadpool_limits(limits=CORES, user_api='blas'):
        for start in range(0, c.calibration_draws, c.calibration_batch):
            size = min(c.calibration_batch, c.calibration_draws-start)
            score = np.abs(rng.standard_normal((size, public['n'])) @ operator.T)
            for name, mask in masks.items():
                maxima[name][start:start+size] = score[:, mask].max(axis=1)
            if start == 0 or (start+c.calibration_batch) % (16*c.calibration_batch) == 0:
                # 16批仅为进度打印间隔，不进入任何统计计算。
                say(f'联合边界：已校准{start+size}/{c.calibration_draws}条误差路径')
    failure = (1-c.verification_coverage)*c.error_budget_share
    critical = np.empty(len(families))
    calibration = {}
    for name, share in (('global', c.global_error_budget_share), ('other', 1-c.global_error_budget_share)):
        delta = failure*share
        # 独立模拟路径与待核验误差路径可交换：第ceil((N+1)(1-delta))顺序统计量
        # 保证边际失败概率不超过delta；不另扣未说明的校准失败概率，不降低原95%覆盖要求。
        rank = math.ceil((c.calibration_draws+1)*(1-delta))
        value = float(np.partition(maxima[name], rank-1)[rank-1])
        critical[masks[name]] = value
        old = float(public['error_critical'][masks[name]][0])
        calibration[name] = dict(statistics=int(masks[name].sum()), failure_budget=delta,
            order_statistic=rank, guaranteed_marginal_failure=(c.calibration_draws+1-rank)/(c.calibration_draws+1),
            old_multiplier=old, joint_multiplier=value)
        say(f'联合边界：{name}系数由{old:.4f}变为{value:.4f}')
    new_bounds = measured_sd*critical
    # 真误差只在本地核对覆盖；不作为ISO输入。沿用原负荷种子、协方差和基线算子。
    load = _load_draw(F.shape[1]//day_size, public['error_scale_MW'], reference=c.load_mw,
        rho=c.load_rho15, daily_share=c.load_daily_variance_share, seed=c.load_seed)
    centered_error = np.asarray(F @ load.ravel())
    realized = np.abs(np.asarray(W @ centered_error))
    true_covered = bool(np.all(realized <= new_bounds+c.solver_tolerance))
    if not true_covered:
        raise RuntimeError('本轮实际误差实现不在联合核验范围内，不能把容量提高解释为有效资料价值')
    if not np.all(new_bounds <= public['bounds']+c.solver_tolerance):
        raise RuntimeError('本轮联合边界并非原边界的收紧，需分别报告而不能假定容量不下降')
    return new_bounds, dict(calibration=calibration, covariance_SD_max_difference_MW=discrepancy,
        actual_error_path_retained=true_covered,
        old_largest_single_point_bound_MW=float(public['bounds'][:public['n']].max()),
        joint_largest_single_point_bound_MW=float(new_bounds[:public['n']].max()),
        confidence=c.verification_coverage, power_failure_budget=(1-c.verification_coverage)*(1-c.error_budget_share),
        calibration_draws=c.calibration_draws, calibration_seed=c.calibration_seed,
        calibrated_set_is_nested_in_original=True)


@njit(parallel=True, cache=True)
def outside_call_increments(q, reference, duration, gap, call_ptr, calls, variants):
    # 三列分别记录原先不缺电时段、原先缺电时段、固定调用所用增载量下尚不缺电时段。
    # 全部使用原缺口和原调用；本函数只分账，不改变调用或资源能力。
    totals = np.zeros((len(gap), 3))
    for j in prange(len(gap)):
        for h in range(gap.shape[1]):
            value = gap[j, h]
            increment = max(value+q, 0.)-max(value, 0.)
            if value <= 0.:
                totals[j, 0] += increment
            else:
                totals[j, 1] += increment
            if value+reference <= 0.:
                totals[j, 2] += increment
    result = np.empty((len(gap)*variants, 3))
    for k in prange(len(result)):
        j = k//variants
        result[k] = totals[j]
        for event in range(call_ptr[k], call_ptr[k+1]):
            start = calls[event, 0]
            finish = min(calls[event, 2], start+duration)
            for h in range(int(start), int(np.ceil(finish))):
                overlap = max(0., min(finish, h+1.)-max(start, float(h)))
                value = gap[j, h]
                increment = overlap*(max(value+q, 0.)-max(value, 0.))
                if value <= 0.:
                    result[k, 0] -= increment
                else:
                    result[k, 1] -= increment
                if value+reference <= 0.:
                    result[k, 2] -= increment
    return result

