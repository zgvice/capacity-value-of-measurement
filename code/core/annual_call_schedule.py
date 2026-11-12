"""固定客户能力下，构造并比较同一组ISO可行调用安排。"""
import time
import numpy as np
from numba import njit,set_num_threads
from common import CORES,cvar
from fixed_call_profile_physics import FixedCallProfiles,annual_losses,background_prefix
from call_schedule_physics import onset_hours,call_formats,call_ends
from normal_customer import normal_power_paths

def say(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


@njit(cache=True)
def merge_touching_calls(block, time_tolerance=1e-10):
    # 默认沿用原小时边界精度；实验二显式传入集中参数。
    # 相接调用合并后按最初开始时刻计算持续时间，避免换指令就把可用时长清零。
    if len(block) == 0:
        return block
    out = np.empty_like(block)
    used = 1
    out[0] = block[0]
    for i in range(1, len(block)):
        if block[i, 0] <= out[used-1, 2]+time_tolerance:
            out[used-1, 1] = max(out[used-1, 1], block[i, 1])
            out[used-1, 2] = max(out[used-1, 2], block[i, 2])
        else:
            out[used] = block[i]
            used += 1
    return out[:used]


class ScheduleChoice:
    def __init__(self, c):
        self.c = c
        set_num_threads(CORES)
        self.engine = FixedCallProfiles(threads=CORES, risk_budget=c.risk_budget_mwh,
            tail_level=c.tail_level, surplus_mw=c.system_surplus_mw,
            maximum_extension_hours=c.maximum_extension_hours,
            reference_tolerance=c.reference_tolerance_mw,
            transport_iterations=c.transport_iterations, transport_solver_tolerance=c.transport_solver_tolerance,
            transport_gap_mwh=c.transport_gap_mwh)
        self.engine.set_power(normal_power_paths(c))
        e = self.engine
        self.gap = -e.surplus*e.g[e.profiles]
        for j in range(len(self.gap)):
            sl = slice(e.ptr[j], e.ptr[j+1])
            self.gap[j, e.hours[sl]] = e.values[sl]
        self.lengths = np.array([row[2] for row in call_formats()])
        self.plans = []
        self.product_weights = e.pi[None, :]/len(e.power)

    def add_plan(self, threshold):
        # 沿用原缺口触发和20种调用格式；额外延长上限取自参数入口，None表示已取消。
        starts, counts = onset_hours(self.gap, threshold)
        ptr, blocks, merges = [0], [], 0
        for j in range(len(self.gap)):
            for variant in range(self.engine.variants):
                original = call_ends(self.gap[j], starts[j], counts[j], self.lengths,
                    variant, threshold, self.c.maximum_extension_hours)
                block = merge_touching_calls(original, self.c.time_tolerance_hours)
                merges += len(original)-len(block)
                blocks.append(block)
                ptr.append(ptr[-1]+len(block))
        plan = dict(threshold_MW=float(threshold), ptr=np.asarray(ptr, np.int64),
                    calls=np.concatenate(blocks), merged_touching_calls=merges)
        for old in self.plans:
            if np.array_equal(old['ptr'], plan['ptr']) and np.array_equal(old['calls'], plan['calls']):
                say('新触发水平产生同一套调用安排，复用已有方案')
                return False
        self.plans.append(plan)
        say(f'加入调用方案{len(self.plans)}：缺口参照增载{threshold:.3f}兆瓦，'
            f'{len(plan["calls"])}段调用，相接合并{merges}处')
        return True

    def losses(self, q, duration):
        e = self.engine
        if e.last_q != q:
            e.last_prefix = background_prefix(q, e.g, e.power, e.surplus)
            e.last_q = q
        total = e.baseline(q)[1]
        best = None
        for plan in self.plans:
            value = annual_losses(q, duration, e.g, e.power, e.last_prefix,
                e.profiles, e.ptr, e.hours, e.values, e.s0, total,
                plan['ptr'], plan['calls'], e.variants, e.surplus)
            if best is None:
                best = value
            else:
                np.minimum(best, value, out=best)
        return best

    def risk(self, q, duration, complete_information):
        loss = self.losses(q, duration)
        if complete_information:
            value = cvar(loss, self.product_weights, self.c.tail_level)
            return dict(lower_MWh=value, upper_MWh=value)
        return self.engine.transport.risk(loss)

    def capacity(self, duration, complete_information, lower=None, upper=None):
        e, c = self.engine, self.c
        if duration <= 0.:
            return e.capacity(duration, tolerance=c.capacity_tolerance_mw)
        lo = e.reference_bracket['lower_MW'] if lower is None else float(lower)
        hi = e.reference_bracket['upper_MW']+float(e.power.max())
        if upper is not None:
            hi = min(hi, float(upper))  # 复用已算出的同一可行方案集合的容量上界。
        if self.risk(lo, duration, complete_information)['upper_MWh'] > c.risk_budget_mwh:
            raise RuntimeError('调用方案容量搜索的下端尚不满足风险预算')
        calls = 0
        label = '完整信息' if complete_information else f'配对未知、可调{duration:.3f}小时'
        while hi-lo > c.capacity_tolerance_mw:
            q = (lo+hi)/2
            value = self.risk(q, duration, complete_information)
            calls += 1
            if value['upper_MWh'] <= c.risk_budget_mwh:
                lo = q
            elif value['lower_MWh'] > c.risk_budget_mwh:
                hi = q
            else:
                raise RuntimeError('风险数值区间跨过预算，需要提高求解精度')
            if calls % 4 == 0:  # 每4步打印一次数值求根进度。
                say(f'{label}：净容量区间{lo-e.reference:.3f}—{hi-e.reference:.3f}兆瓦')
        low = self.risk(lo, duration, complete_information)
        high = self.risk(hi, duration, complete_information)
        if not low['upper_MWh'] <= c.risk_budget_mwh < high['lower_MWh']:
            raise RuntimeError('重排调用后的容量风险根未夹住预算')
        return dict(net_lower_MW=lo-e.reference, net_upper_MW=hi-e.reference,
            total_lower_MW=lo, total_upper_MW=hi, duration_hours=duration,
            risk_lower_boundary_MWh=low['upper_MWh'], risk_upper_boundary_MWh=high['lower_MWh'],
            root_evaluations=calls, feasible_plan_count=len(self.plans))

