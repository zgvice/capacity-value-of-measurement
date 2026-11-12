"""Fitted quarter-hour power with the already frozen future call schedules."""
from functools import lru_cache

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.optimize import linprog
import warnings

from common import CORES, DATA, cvar, monotone_root
from common_settings import TAIL_LEVEL,RISK_BUDGET_MWH,SYSTEM_SURPLUS_MW
from system_reference import _sort_zero_background, _zero_loss
from call_schedule_physics import onset_hours,call_ends,call_formats,DEFAULT_INVITATION_EXTENSION_HOURS
from free_coupling_capacity import TailTransport,tail_argmax

set_num_threads(CORES)  # CORES来自os.cpu_count；按用户要求使用机器可用核心。
HOURS = 8760  # 沿用年度场景整理中的完整一年时序。
REFERENCE = 378.6853328347206  # exp1/capacity.csv：无新增资源的容量边界，用于固定调用。
BUDGET = RISK_BUDGET_MWH  # 沿用既定主实验的新增缺供风险预算（兆瓦时），不是实测值或市场规定。
# 下文1000*g的1000沿用原方案非缺电时段余量的中央尺度；原方案第54行明确为数值补全。
# 下文4和0.25是每小时4个15分钟时段的单位换算，非新的资源参数。


class CompactTail(TailTransport):
    """Exact tail transport via a small dual master and feasible tail witnesses."""
    def __init__(self,pi,R,alpha=TAIL_LEVEL,threads=CORES,*,iterations=500,
                 solver_tolerance=1e-10,gap_mwh=.002):
        # alpha=.95沿用主实验风险标准：最严重5%年度场景的平均新增缺供。
        self.pi = np.asarray(pi,float)
        self.R,self.J,self.alpha = int(R),len(pi),float(alpha)
        self.c = 1-self.alpha
        self.resource_weights = np.full(self.R,1./self.R)
        self.resource_cap = self.resource_weights/self.c
        self.system_cap = self.pi/self.c
        self.threads = threads
        self.iterations = iterations  # 原风险分解迭代上限；实验二从集中参数传入。
        self.solver_tolerance = solver_tolerance  # 原风险线性规划数值容差。
        self.gap_mwh = gap_mwh  # 原风险证书的数值间隙，单位兆瓦时。
        self.last_v = np.zeros(self.R)

    def system_tail(self,values):
        order = np.argsort(-values,kind='stable')
        weights = self.pi[order]
        mass = np.minimum(weights,np.maximum(self.c-np.r_[0.,np.cumsum(weights)[:-1]],0.))/self.c
        out = np.empty(self.J)
        out[order] = mass
        return out

    def risk(self,loss):
        loss = np.asarray(loss,float)
        assert loss.shape == (self.R,self.J) and np.all(loss>=0.)
        scale = max(1.,float(loss.max())/100.)  # 沿用原TailTransport的数值缩放，不改变风险度量。
        L = loss/scale
        w0 = self.system_tail(L.mean(axis=0))
        z0 = self.resource_weights[:,None]*w0[None,:]
        intercepts = [float(np.sum(L*z0))]
        gradients = [np.zeros(self.R)]
        witnesses = [None]
        best_tail = z0
        lower = self._primal_lower(loss,z0)
        upper = np.inf
        v = np.maximum(self.last_v/scale,0.)
        best_v = v.copy()
        columns = np.arange(self.J)
        for iteration in range(self.iterations):
            chosen,envelope=tail_argmax(L,v)
            w = self.system_tail(envelope)
            used = np.flatnonzero(w>0.)
            eta = float(envelope[used].min())
            u = np.maximum(envelope-eta,0.)
            dual = np.r_[-u,-v,-eta]
            bound = self._dual_upper(-L,dual,scale)
            if bound < upper:
                upper,best_v = bound,v.copy()
            gradients.append(self.resource_cap-np.bincount(chosen,weights=w,minlength=self.R))
            intercepts.append(float(w@L[chosen,columns]))
            witnesses.append((chosen[used].copy(),used,w[used].copy()))
            A = np.c_[np.asarray(gradients),-np.ones(len(gradients))]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options')
                # 1e-10是本轮风险主问题的数值可行性/对偶容差。
                master = linprog(np.r_[np.zeros(self.R),1.],A_ub=A,b_ub=-np.asarray(intercepts),
                    bounds=[(0.,None)]*self.R+[(None,None)],method='highs',
                    options={'threads':self.threads,'primal_feasibility_tolerance':self.solver_tolerance,
                             'dual_feasibility_tolerance':self.solver_tolerance})
            if not master.success:raise RuntimeError(master.message)
            mixture = np.maximum(-master.ineqlin.marginals,0.)
            mixture /= mixture.sum()
            z = mixture[0]*z0
            for coefficient,item in zip(mixture[1:],witnesses[1:]):
                if coefficient>0.:
                    r,j,weight = item
                    z[r,j] += coefficient*weight
            z = self._repair_tail(z)
            new_lower = self._primal_lower(loss,z)
            if new_lower>lower:
                lower,best_tail = new_lower,z
            # 0.002兆瓦时为本轮求解间隙，后项为浮点误差留量；均属于数值精度选择。
            numerical_width = self.gap_mwh+8*np.finfo(float).eps*loss.size*max(1.,float(loss.max()))
            if upper-lower <= numerical_width:
                self.last_v = best_v*scale
                residual = self._margin_residual(best_tail)
                assert residual<2e-9 and lower<=upper+1e-6
                return dict(lower_MWh=lower,upper_MWh=max(upper,lower),tail=best_tail,
                            margin_residual=residual,dual_iterations=iteration+1)
            v = np.maximum(master.x[:self.R],0.)
        raise RuntimeError(('Tail transport did not close its certified risk bounds',lower,upper))


@njit(cache=True, parallel=True)
def background_prefix(q, g, power, surplus=SYSTEM_SURPLUS_MW):
    R, T = power.shape
    F = len(g)
    prefix = np.empty((R, F, HOURS + 1))
    for rf in prange(R * F):
        r, f = rf // F, rf % F
        prefix[r, f, 0] = 0.
        total = 0.
        correction = 0.
        for h in range(HOURS):
            raw = max(q - surplus * g[f, h], 0.)
            value = 0.
            for k in range(4):
                value += .25 * min(raw, power[r, 4*h+k])
            y = value - correction
            updated = total + y
            correction = (updated-total)-y
            total = updated
            prefix[r, f, h+1] = total
    return prefix


@njit(cache=True)
def prefix_value(t, q, background, power, cumulative, surplus=SYSTEM_SURPLUS_MW):
    h = int(t)
    if h >= HOURS:
        return cumulative[HOURS]
    out = cumulative[h]
    raw = max(q - surplus * background[h], 0.)
    for k in range(4):
        length = max(0., min(t, h+(k+1)/4.)-(h+k/4.))
        out += length * min(raw, power[4*h+k])
    return out


@njit(cache=True, parallel=True)
def annual_losses(q, duration, g, power, prefix, profiles, ptr, hours, values,
                  s0, total, call_ptr, calls, variants, surplus=SYSTEM_SURPLUS_MW):
    R, J = len(power), len(profiles)
    out = np.empty((R, J*variants))
    for rk in prange(R * J * variants):
        r = rk // (J*variants)
        k = rk % (J*variants)
        j, f = k // variants, profiles[k // variants]
        saved = 0.
        first, last = call_ptr[k], call_ptr[k+1]
        for c in range(first, last):
            start, end, stop = calls[c]
            finish = min(stop, start+duration)
            saved += prefix_value(finish,q,g[f],power[r],prefix[r,f],surplus)
            saved -= prefix_value(start,q,g[f],power[r],prefix[r,f],surplus)
        # Replace background-gap savings at every official shortage hour.
        current = first
        for ix in range(ptr[j], ptr[j+1]):
            h = hours[ix]
            while current < last and min(calls[current,2],calls[current,0]+duration) <= h:
                current += 1
            c = current
            while c < last and calls[c,0] < h+1:
                start = max(float(h), calls[c,0])
                finish = min(float(h+1),calls[c,2],calls[c,0]+duration)
                if finish > start:
                    actual = max(values[ix]+q,0.)
                    background = max(q-surplus*g[f,h],0.)
                    for quarter in range(4):
                        length = max(0.,min(finish,h+(quarter+1)/4.)-max(start,h+quarter/4.))
                        p = power[r,4*h+quarter]
                        saved += length*(min(actual,p)-min(background,p))
                c += 1
        out[r,k] = max(total[j]-saved-s0[j],0.)
    return out


class FixedCallProfiles:
    def __init__(self,threads=CORES,*,risk_budget=BUDGET,tail_level=TAIL_LEVEL,surplus_mw=SYSTEM_SURPLUS_MW,
                 maximum_extension_hours=DEFAULT_INVITATION_EXTENSION_HOURS,reference_tolerance=.0001,
                 transport_iterations=500,transport_solver_tolerance=1e-10,transport_gap_mwh=.002):
        # 默认值均来自原年度算例；敏感性分析显式传入当前风险标准、余量和调用延续时长。
        self.threads = int(threads)
        self.transport_options = dict(iterations=transport_iterations,
            solver_tolerance=transport_solver_tolerance,gap_mwh=transport_gap_mwh)
        self.risk_budget,self.tail_level,self.surplus = float(risk_budget),float(tail_level),float(surplus_mw)
        set_num_threads(self.threads)
        with np.load(DATA/'annual/exact_path_groups.npz') as src:
            self.profiles,self.weights,self.ptr,self.hours,self.values,self.atoms = (
                src[k] for k in ('profile_ids','weights','ptr','hours','values','atom_to_group'))
        with np.load(DATA/'annual/background_arrays.npz') as src:
            self.g = src['g_committed_median_allnumeric']
        assert len(self.atoms) == 40300 and np.isclose(self.weights.sum(),1.)
        self.s0 = np.array([self.values[self.ptr[j]:self.ptr[j+1]].sum() for j in range(len(self.weights))])
        self.ordered,self.prefix0 = _sort_zero_background(self.g,self.surplus)
        # 无资源风险根随当前风险参数和余量重算，再用该根固定调用；不沿用写死的378.6853。
        root,br = monotone_root(lambda q:cvar(self.baseline(q)[0],self.weights,alpha=self.tail_level),
            self.risk_budget,tol=reference_tolerance)
        self.reference,self.reference_bracket = root,br
        gap = -self.surplus*self.g[self.profiles]
        for j in range(len(gap)):
            sl = slice(self.ptr[j],self.ptr[j+1])
            gap[j,self.hours[sl]] = self.values[sl]
        onsets,counts = onset_hours(gap,self.reference)
        lengths = np.array([row[2] for row in call_formats()])
        self.variants = len(lengths)
        ptr, calls = [0], []
        for j in range(len(gap)):
            for v in range(self.variants):
                block = call_ends(gap[j],onsets[j],counts[j],lengths,v,self.reference,maximum_extension_hours)
                calls.append(block)
                ptr.append(ptr[-1]+len(block))
        self.call_ptr = np.asarray(ptr,np.int64)
        self.calls = np.concatenate(calls)
        self.pi = np.repeat(self.weights/self.variants,self.variants)
        self.transport = None
        self.last_q = None
        self.last_prefix = None

    @lru_cache(maxsize=128)
    def baseline(self,q):
        return _zero_loss(q,self.ordered,self.prefix0,self.g,self.profiles,self.ptr,
                          self.hours,self.values,self.s0,self.surplus)

    def set_power(self,power):
        power = np.ascontiguousarray(power,float)
        assert power.ndim == 2 and power.shape[1] == 35040
        assert np.isfinite(power).all() and np.all(power >= 0.)
        self.power = power
        self.transport = CompactTail(self.pi,len(power),alpha=self.tail_level,threads=self.threads,
                                     **self.transport_options)
        self.last_q, self.last_prefix = None,None

    def loss(self,q,duration):
        if self.last_q != q:
            self.last_prefix = background_prefix(q,self.g,self.power,self.surplus)
            self.last_q = q
        total = self.baseline(q)[1]
        return annual_losses(q,duration,self.g,self.power,self.last_prefix,self.profiles,
                             self.ptr,self.hours,self.values,self.s0,total,
                             self.call_ptr,self.calls,self.variants,self.surplus)

    def risk(self,q,duration):
        return self.transport.risk(self.loss(q,duration))

    def capacity(self,duration,tolerance=.1):
        # 0.1兆瓦沿用既有年度认证的求根区间精度，不是统计置信区间。
        if duration<=0.:
            # 原模型允许零响应；持续时长为零时，两次相同无资源风险根相减，净容量恰为零。
            lo,hi = self.reference_bracket['lower_MW'],self.reference_bracket['upper_MW']
            return dict(duration_hours=0.,net_lower_MW=0.,net_upper_MW=0.,
                total_lower_MW=lo,total_upper_MW=hi,
                risk_lower_boundary_MWh=cvar(self.baseline(lo)[0],self.weights,alpha=self.tail_level),
                risk_upper_boundary_MWh=cvar(self.baseline(hi)[0],self.weights,alpha=self.tail_level),root_evaluations=0)
        lo,hi = self.reference_bracket['lower_MW'],self.reference_bracket['upper_MW']+float(self.power.max())
        calls = 0
        while hi-lo > tolerance:
            mid = (lo+hi)/2.
            value = self.risk(mid,duration)
            calls += 1
            if value['upper_MWh'] <= self.risk_budget:
                lo = mid
            elif value['lower_MWh'] > self.risk_budget:
                hi = mid
            else:
                raise RuntimeError('Risk arithmetic interval straddles budget; refine numerical precision')
        lower = self.risk(lo,duration)
        upper = self.risk(hi,duration)
        assert lower['upper_MWh'] <= self.risk_budget < upper['lower_MWh']
        return dict(duration_hours=float(duration),net_lower_MW=lo-self.reference,
                    net_upper_MW=hi-self.reference,total_lower_MW=lo,total_upper_MW=hi,
                    risk_lower_boundary_MWh=lower['upper_MWh'],
                    risk_upper_boundary_MWh=upper['lower_MWh'],root_evaluations=calls)


def self_check():
    from experiment_one_physics import losses as constant_losses
    engine = FixedCallProfiles()
    gap = -engine.surplus*engine.g[engine.profiles]
    for j in range(len(gap)):
        sl = slice(engine.ptr[j],engine.ptr[j+1])
        gap[j,engine.hours[sl]] = engine.values[sl]
    starts,counts = onset_hours(gap,REFERENCE)
    lengths = np.array([row[2] for row in call_formats()])
    q = 520.
    old = constant_losses(q,REFERENCE,gap,starts,counts,lengths,
                           engine.baseline(q)[1],engine.s0,np.array([0.,1.,2.,4.]),500.,4.)
    engine.set_power(np.full((1,35040),500.))
    errors = []
    # 18.5小时=既有最长14.5小时调用+已冻结的未来最多4小时延续；只用于同一真实轨迹的等价核对。
    duration = float(lengths.max()+4.)
    assert np.all(engine.calls[:,2]<=engine.calls[:,0]+duration+1e-10)
    result = engine.loss(q,duration)
    errors.append(float(np.max(np.abs(result[0]-old[-1]))))
    assert max(errors)<1e-6,errors
    resource = 700.+250.*np.sin(np.arange(35040)*.031)
    engine.set_power(resource[None,:])
    duration = 8.3  # 本次逐分钟重放的自选检查点（小时），不进入正式参数或结果。
    computed = engine.loss(q,duration)[0]
    indices = np.array([0,17,81,490,1109,7333,17881,20153,28743,35759])
    brute_errors = []
    for k in indices:
        j = k//engine.variants
        availability = np.zeros(HOURS*60,dtype=np.bool_)
        for start,end,stop in engine.calls[engine.call_ptr[k]:engine.call_ptr[k+1]]:
            finish = min(stop,start+duration)
            availability[int(round(60*start)):int(round(60*finish))] = True
        raw = np.repeat(gap[j]+q,60)
        brute = max(float(np.maximum(raw-np.repeat(resource,15)*availability,0.).sum()/60)
                    -engine.s0[j],0.)
        brute_errors.append(abs(brute-computed[k]))
    assert max(brute_errors)<1e-6,brute_errors
    risk = engine.risk(q,duration)
    direct = cvar(computed,engine.pi)
    assert risk['lower_MWh']-1e-6 <= direct <= risk['upper_MWh']+1e-6
    rng = np.random.default_rng(2026110401)
    test_pi = np.array([.10,.23,.08,.31,.28])
    small_errors = []
    for _ in range(4):
        small_loss = rng.uniform(0.,200.,(3,5))
        dense = TailTransport(test_pi,3,alpha=.55,threads=CORES).risk(small_loss)
        compact = CompactTail(test_pi,3,alpha=.55,threads=CORES).risk(small_loss)
        assert compact['lower_MWh']<=dense['upper_MWh']+1e-7
        assert dense['lower_MWh']<=compact['upper_MWh']+1e-7
        small_errors.append(compact['upper_MWh']-compact['lower_MWh'])
    return dict(constant_power_all_scenarios_max_error_MWh=max(errors),
                varying_power_direct_replay_max_error_MWh=max(brute_errors),
                direct_CVaR_MWh=direct,risk_bounds=[risk['lower_MWh'],risk['upper_MWh']],
                compact_transport_max_certificate_width_MWh=max(small_errors),
                annual_atoms=40300,root_reference_MW=REFERENCE)


if __name__=='__main__':
    import json
    from common import AUDIT,write_json
    checked = self_check()
    write_json(AUDIT/'exp2_annual_engine_check.json',checked)
    print(json.dumps(checked,ensure_ascii=False,indent=2))
