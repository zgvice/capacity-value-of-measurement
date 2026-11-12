"""四类客户的核心参数补算。统一932项统计，冻结同一调用方案池。

结果只保存参数、历史核验、风险根与差额；完整年度数组和功率轨迹复用或在内存生成。
"""
from __future__ import annotations
import argparse, gc, json, math, time
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from scipy.stats import truncnorm

from common import OUT, AUDIT, CORES, cvar, monotone_root, write_json
from portfolio_inputs import PortfolioParameters, profiles
from portfolio_history import calibration_maxima, multipliers, history_power_bounds
from normal_customer import build_history
from historical_duration import HistoricalDuration
from run_experiment_four import future_power, untreated_load
from annual_call_schedule import ScheduleChoice
from portfolio_sparse_physics import SparsePortfolioAnnual
from fixed_call_profile_physics import CompactTail
from sensitivity_parameters import SensitivityParameters

DEST = OUT/'exp4/sens'
POLICY = SensitivityParameters()
NAMES = ('正态1', 'LSE A', 'LSE B', 'LSE C')  # 正态参照及三个匿名实测来源。
LABEL = {'正态1': '正态参照', **{s: s for s in NAMES[1:]}}
FILE_CUSTOMER = {'正态1': 'normal', 'LSE A': 'lse_a', 'LSE B': 'lse_b', 'LSE C': 'lse_c'}
FILE_TASK = {'风险': 'risk', '核验': 'verification'}
PLAN_FILE = DEST/'fixed_schedule.npz'
WORKER_THREADS=None  # 每个计算进程的线程分配固定，复用进程时不重设HiGHS全局调度器。


def say(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parameters():
    values = read(AUDIT/'portfolio_capacity.json')['parameters']
    return replace(PortfolioParameters(**values), capacity_tolerance_mw=POLICY.capacity_tolerance_mw)


def threads_for_worker(threads):
    global WORKER_THREADS
    if WORKER_THREADS is None:WORKER_THREADS=int(threads)
    threads=WORKER_THREADS
    # 线程数按进程固定；多个独立批次共同使用全部22核。
    import annual_call_schedule, historical_duration, fixed_call_profile_physics
    annual_call_schedule.CORES = threads
    historical_duration.CORES = threads
    fixed_call_profile_physics.CORES = threads
    set_num_threads(threads)


def public_inputs(c):
    public, _, _, maximum, _, _, _ = build_history(c)
    if public['W'].shape[0] != 932:  # 当前实验二、三定稿统计全集；与实验四955项留出设计分开。
        raise RuntimeError('主比较统计全集与932项定稿不一致')
    with np.load(OUT/'exp3/portfolio_iso_inputs.npz') as z:
        data = {k: z[k].copy() for k in ('prototype_names', 'call_baselines', 'call_meters', 'original_meters')}
    model = profiles(c)
    maxima = calibration_maxima(c, public)
    return public, maximum, data, model, maxima


def critical_values(c, public, maxima):
    critical, detail = multipliers(c, public, 1, maxima)
    return critical, detail


def error_membership(public, error, bounds, count, bias_bound, tolerance):
    """检验同一条真实误差能否满足原有随机误差界与允许均值偏差。"""
    keep=public['required']<=count
    coefficients=np.asarray(public['W'].sum(axis=1)).ravel()[keep]
    values=(public['W']@error)[keep];bound=np.asarray(bounds)[keep]
    lower=max(-bias_bound,float(np.max((values-bound)/coefficients)))
    upper=min(bias_bound,float(np.min((values+bound)/coefficients)))
    return dict(truth_error_retained=bool(lower<=upper+tolerance),
        admissible_bias_lower_MW=lower,admissible_bias_upper_MW=upper,
        zero_bias_error_retained=bool(np.all(abs(values)<=bound+tolerance)),
        maximum_error_ratio=float(np.max(abs(values)/bound)))


def historical_case(c, public, maximum, data, model, maxima, name, *, actual_rate=None, bound_rate=None):
    critical, calibration = critical_values(c, public, maxima)
    scale = c.error_sd_rate if bound_rate is None else bound_rate
    p = dict(public, bounds=public['error_stat_SD_MW']*critical*(scale/c.error_sd_rate))
    i = list(data['prototype_names']).index(name)
    baseline = data['call_baselines'][i].copy()
    meter = data['call_meters'][i].copy()
    untreated = untreated_load(c, public, i)
    overlap = np.clip((np.minimum(public['left']+.25, np.minimum(public['stops'], public['starts']+c.true_duration_hours))
        -np.maximum(public['left'], public['starts']))/.25, 0., 1.)  # 15分钟读数与邀约交叠。
    original_delivery = untreated-meter
    clipped, reduced = 0, 0.
    if actual_rate is not None:
        ratio = actual_rate/c.error_sd_rate
        baseline = c.load_mw+(baseline-c.load_mw)*ratio
        changed_load = c.load_mw+(untreated-c.load_mw)*ratio
        prior = c.load_mw+(data['original_meters'][i]-c.load_mw)*ratio
        if min(changed_load.min(), prior.min()) < 0:
            raise ValueError(f'{name}的实际波动档出现负正常用电，不能作为物理情景')
        potential = np.divide(original_delivery, overlap, out=np.zeros_like(overlap), where=overlap>0.)
        delivery = np.minimum(potential, changed_load)*overlap
        clipped = int(np.sum(potential>changed_load))
        reduced = float(.25*np.sum(original_delivery-delivery))
        meter = changed_load-delivery
        untreated = changed_load
    p['baseline'] = baseline
    power = float(history_power_bounds(c, model, [name], [1.], public['n'])[0])
    error = baseline-untreated-c.error_mean_rate*c.load_mw
    rows = []
    for count in POLICY.history_counts:
        selected = public['call_ids']>=public['J']-count
        keep = public['required']<=count
        # 一项统计只在其涉及的全部记录均可见时保留，避免把20次统计截成5次。
        if public['W'][keep][:, ~selected].getnnz():
            raise RuntimeError('五次资料混入不可见记录的统计量')
        h = HistoricalDuration(p, meter, count, power, bias_bound=c.load_mw*c.bias_bound_rate,
            solver_tolerance=c.solver_tolerance, duration_check_tolerance=c.duration_check_tolerance_hours)
        duration = h.minimum(maximum, minimum=c.verified_min_duration_hours)
        error_check=error_membership(public,error,p['bounds'],count,c.load_mw*c.bias_bound_rate,c.solver_tolerance)
        true_response = original_delivery if actual_rate is None else untreated-meter
        pcap = power*overlap
        true_power_retained = bool(np.all(true_response[selected]<=pcap[selected]+c.solver_tolerance))
        rows.append(dict(count=count, duration_hours=duration, points=int(selected.sum()), statistics=int(keep.sum()),
            **error_check,truth_power_retained=true_power_retained,
            truth_retained=bool(error_check['truth_error_retained'] and true_power_retained),power_bound_MW=power,
            physically_limited_points=clipped, reduced_delivery_MWh=reduced))
    return rows, calibration


def audit_saved_histories():
    """按实际使用的偏差并集核对保存记录，不重算历史最优时长及全年容量。"""
    c=parameters();public,_,data,_,maxima=public_inputs(c)
    scenario_map={r['key']:r for r in scenarios(c)}
    scenario_map['underestimated_bound']=dict(changes={},bound_rate=POLICY.error_sd_rates[0])
    for name in NAMES:
        path=DEST/f'calc_{FILE_CUSTOMER[name]}.json';saved=read(path);i=list(data['prototype_names']).index(name)
        original_error=data['call_baselines'][i]-untreated_load(c,public,i)
        for row in saved['history_rows']:
            scenario=scenario_map[row['scenario']];cc=replace(c,**scenario['changes'])
            critical,_=critical_values(cc,public,maxima)
            scale=scenario.get('bound_rate',cc.error_sd_rate)/cc.error_sd_rate
            bound=public['error_stat_SD_MW']*critical*scale
            error=original_error*(scenario.get('actual_rate',cc.error_sd_rate)/cc.error_sd_rate)-cc.error_mean_rate*cc.load_mw
            row.update(error_membership(public,error,bound,row['count'],cc.load_mw*cc.bias_bound_rate,cc.solver_tolerance))
            row['truth_retained']=bool(row['truth_error_retained'] and row['truth_power_retained'])
        lookup={(r['scenario'],r['count']):r for r in saved['history_rows']}
        for row in saved['rows']:
            if row['history_count']:row['truth_retained']=lookup[row['scenario'],row['history_count']]['truth_retained']
        write_json(path,saved)
    export()


def save_plans(choice, path=PLAN_FILE):
    arrays = {}
    for j, p in enumerate(choice.plans):
        arrays[f'ptr_{j}'] = p['ptr']; arrays[f'calls_{j}'] = p['calls']
    arrays['thresholds'] = np.asarray([p['threshold_MW'] for p in choice.plans])
    np.savez_compressed(path, **arrays)


def load_choice(c):
    choice = ScheduleChoice(c)
    # 不调用make_schedule；各参数档使用已冻结的同一组实际调用起止数组。
    with np.load(PLAN_FILE) as z:
        choice.plans = [dict(threshold_MW=float(t), ptr=z[f'ptr_{j}'].copy(), calls=z[f'calls_{j}'].copy())
                        for j, t in enumerate(z['thresholds'])]
    return choice


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    c = parameters(); public, maximum, data, model, maxima = public_inputs(c)
    history = {}
    for name in NAMES:
        history[name] = historical_case(c, public, maximum, data, model, maxima, name)[0]
    source = read(AUDIT/'exp2_check.json')['invitation_comparison']
    choice = ScheduleChoice(c)
    for p in source['plans']: choice.add_plan(p['trigger_reference_MW'])
    old = pd.read_csv(OUT/'exp3/portfolio_sequence.csv')
    normal_old = old[(old['组合']=='正态1户') & old['每户历史次数'].isin(POLICY.history_counts)].sort_values('每户历史次数')
    probes = []
    engine = SparsePortfolioAnnual(c, choice, future_power(c, model, '正态1')[None, :, :],
                                  source['capacity_cases'][-1]['total_upper_MW'])
    for h, (_, row) in zip(history['正态1'], normal_old.iterrows()):
        q = choice.engine.reference+float(row['认证下界_兆瓦'])
        value = engine.risk(q, engine.duration_segments([h['duration_hours']]), False)
        probes.append(dict(count=h['count'], q_MW=q, duration_hours=h['duration_hours'],
            original_risk_lower_MWh=value['lower_MWh'], original_risk_upper_MWh=value['upper_MWh']))
    original_count = len(choice.plans)
    for p in probes: choice.add_plan(p['q_MW'])
    for p in probes:
        engine.clear_geometry()
        value = engine.risk(p['q_MW'], engine.duration_segments([p['duration_hours']]), False)
        p.update(extended_risk_lower_MWh=value['lower_MWh'], extended_risk_upper_MWh=value['upper_MWh'],
            risk_reduction_lower_MWh=p['original_risk_lower_MWh']-value['upper_MWh'])
    # 扩充池中包括原方案；所有后续参数档及四类客户共用同一池。
    save_plans(choice)
    write_json(DEST/'sensitivity_params.json', dict(parameters=c.values(), sensitivity=POLICY.__dict__,
        names=list(NAMES), statistics=public['W'].shape[0], history_counts=list(POLICY.history_counts),
        history_dates={str(n): [str(d['task']['date']) for d in public['designs'][-n:]] for n in POLICY.history_counts},
        source_plan_count=original_count, frozen_plan_count=len(choice.plans),
        frozen_plan_thresholds_MW=[p['threshold_MW'] for p in choice.plans], plan_probes=probes,
        statistical_rule='主比较932项统计；逐项保留可见记录支持的统计，使用同一校准全集',
        plan_rule='中央5次和20次边界追加安排后冻结；风险和核验敏感性均不重生成调用',
        cores=CORES))
    write_json(DEST/'central_verification.json', history)
    say(f'准备完成：932项统计，冻结{len(choice.plans)}套安排；四类客户保留5次、20次、完整信息。')


def scenarios(c):
    rows = [dict(key='central', family='基准', value=0., label='基准', changes={})]
    for value in POLICY.annual_tail_levels:
        if value != c.tail_level: rows.append(dict(key=f'tail_{value}', family='年度风险水平', value=value,
            label=f'{value:.0%}', changes=dict(tail_level=value)))
    for value in POLICY.risk_budget_factors:
        if value != 1.: rows.append(dict(key=f'budget_{value}', family='新增缺供上限', value=value*c.risk_budget_mwh,
            label=f'{value*c.risk_budget_mwh:g}兆瓦时', changes=dict(risk_budget_mwh=value*c.risk_budget_mwh)))
    for value in POLICY.history_coverages:
        if value != c.verification_coverage: rows.append(dict(key=f'coverage_{value}', family='历史核验覆盖', value=value,
            label=f'{value:.0%}', changes=dict(verification_coverage=value)))
    for value in POLICY.bias_bound_rates:
        if value != c.bias_bound_rate: rows.append(dict(key=f'bias_{value}', family='平均偏差界', value=value,
            label=f'±{value*c.load_mw:g}兆瓦', changes=dict(bias_bound_rate=value)))
    for value in POLICY.error_sd_rates:
        if value != c.error_sd_rate: rows.append(dict(key=f'actual_{value}', family='实际历史误差波动', value=value,
            label=f'{value*c.load_mw:g}兆瓦', changes={}, actual_rate=value, bound_rate=value))
        if value>c.error_sd_rate: rows.append(dict(key=f'bound_{value}', family='固定记录核验范围', value=value,
            label=f'{value*c.load_mw:g}兆瓦', changes={}, bound_rate=value))
    return rows


def set_risk(engine, c, reference):
    engine.c = c; engine.choice.c = c
    engine.e.tail_level = c.tail_level; engine.e.risk_budget = c.risk_budget_mwh
    engine.e.reference, engine.e.reference_bracket = reference
    engine.e.transport = CompactTail(engine.e.pi, engine.R, alpha=c.tail_level, threads=engine.e.threads,
        iterations=c.transport_iterations, solver_tolerance=c.transport_solver_tolerance, gap_mwh=c.transport_gap_mwh)


def reference_for(engine, c):
    key = (c.tail_level, c.risk_budget_mwh)
    if not hasattr(engine, '_references'): engine._references = {}
    if key not in engine._references:
        engine._references[key] = monotone_root(lambda q: cvar(engine.e.baseline(q)[0], engine.e.weights, c.tail_level),
            c.risk_budget_mwh, tol=c.reference_tolerance_mw)
    return engine._references[key]


def checked_capacity(engine, H, complete, seed=None):
    segments = engine.duration_segments([H])
    if not segments:
        return engine.capacity(segments, complete)
    if seed:
        lo, hi = float(seed['total_lower_MW']), float(seed['total_upper_MW'])
        a, b = engine.risk(lo, segments, complete), engine.risk(hi, segments, complete)
        if hi-lo<=engine.c.capacity_tolerance_mw and a['upper_MWh']<=engine.c.risk_budget_mwh<b['lower_MWh']:
            return dict(total_lower_MW=lo, total_upper_MW=hi,
                net_lower_MW=lo-engine.e.reference, net_upper_MW=hi-engine.e.reference,
                risk_lower_MWh=a['upper_MWh'], risk_upper_MWh=b['lower_MWh'], root_evaluations=2)
        if a['upper_MWh']>engine.c.risk_budget_mwh:
            local=max(engine.e.reference_bracket['lower_MW'],lo-POLICY.capacity_tolerance_mw)
            probe=engine.risk(local,segments,complete)
            lo=local if probe['upper_MWh']<=engine.c.risk_budget_mwh else engine.e.reference_bracket['lower_MW']
        elif b['upper_MWh']<=engine.c.risk_budget_mwh:
            lo=hi
            hi=max(hi,engine.search_upper)  # 风险放宽时先用原完整信息边界；不足时由容量函数自动扩展。
    else: lo, hi = engine.e.reference_bracket['lower_MW'], engine.e.reference_bracket['upper_MW']+engine.profiles[engine.full].max()
    return engine.capacity(segments, complete, lower=lo, upper=hi)


def row_result(name, scenario, count, history, value, c, resolution, reference):
    return dict(customer=name, scenario=scenario['key'], family=scenario['family'], setting=scenario['label'],
        value=scenario['value'], history_count=count, duration_hours=history['duration_hours'],
        tail_level=c.tail_level, risk_budget_MWh=c.risk_budget_mwh, history_coverage=c.verification_coverage,
        bias_bound_MW=c.bias_bound_rate*c.load_mw, future_paths=resolution[0], quadrature_nodes=resolution[1],
        reference_MW=reference[0], truth_retained=history.get('truth_retained', True), **value)


def customer_job(name, resolution, threads, selected_keys=None, filename=None):
    threads_for_worker(threads)
    c = replace(parameters(), future_paths=resolution[0], quadrature_nodes=resolution[1])
    public, maximum, data, model, maxima = public_inputs(c)
    history_cache = {}; rows = []; history_rows=[]
    destination = DEST/(filename or f'calc_{FILE_CUSTOMER[name]}.json')
    saved = read(destination) if destination.exists() else {}
    canonical=DEST/f'calc_{FILE_CUSTOMER[name]}.json'
    if filename is not None and not saved and canonical.exists():saved=read(canonical)
    if saved.get('resolution') == list(resolution):
        rows = saved.get('rows', []);history_rows=saved.get('history_rows',[])
    precision_path=DEST/f'precision_{FILE_CUSTOMER[name]}_{resolution[0]}.json'
    if not rows and filename is None and precision_path.exists():
        precision=read(precision_path)
        if precision.get('resolution')==list(resolution):
            rows=precision['rows'].copy();history_rows=precision.get('history_rows',[]).copy()
    choice = load_choice(c)
    old_groups = read(AUDIT/'portfolio_capacity.json')['groups']
    old_full = old_groups['正态1户' if name=='正态1' else name]['full_information']
    old_curve=pd.read_csv(OUT/'exp3/portfolio_sequence.csv')
    group_name='正态1户' if name=='正态1' else name
    anchors=old_curve[(old_curve['组合']==group_name)&old_curve['每户历史次数'].isin(POLICY.history_counts)]
    seeds={int(r['每户历史次数']):dict(total_lower_MW=choice.engine.reference+float(r['认证下界_兆瓦']),
        total_upper_MW=choice.engine.reference+float(r['认证上界_兆瓦'])) for r in anchors.to_dict('records')}
    engine = SparsePortfolioAnnual(c, choice, future_power(c, model, name)[None,:,:],
        old_full['total_upper_MW'])  # 复用既有完整信息边界作初始索引范围；超出时自动扩大。
    used=[]
    for plan in choice.plans:
        hours=np.minimum(plan['calls'][:,2]-plan['calls'][:,0],c.true_duration_hours)
        total=np.r_[0.,np.cumsum(hours)]
        used.append(float(np.max(total[plan['ptr'][1:]]-total[plan['ptr'][:-1]])))
    state_initial=c.state_power_ratio*c.nomination_mw*8760.  # 沿用原逐期物理嵌入的初态。
    state_consumption=max(used)*float(engine.profiles[engine.full].max())
    if state_consumption>state_initial:raise RuntimeError('本档须显式求解逐期状态，不能使用已消元积分')
    full_cache={(r['tail_level'],r['risk_budget_MWh']):r for r in rows if r['history_count']==0}
    fields=('total_lower_MW','total_upper_MW','net_lower_MW','net_upper_MW','risk_lower_MWh','risk_upper_MWh','root_evaluations')
    capacity_cache={(r['tail_level'],r['risk_budget_MWh'],r['duration_hours'],r['history_count']==0):
        {k:r[k] for k in fields if k in r} for r in rows}
    for scenario in scenarios(c):
        if selected_keys is not None and scenario['key'] not in selected_keys: continue
        cc = replace(c, **scenario['changes'])
        key = (cc.verification_coverage, cc.bias_bound_rate, scenario.get('actual_rate'), scenario.get('bound_rate'))
        if key not in history_cache:
            history_cache[key] = historical_case(cc, public, maximum, data, model, maxima, name,
                actual_rate=scenario.get('actual_rate'), bound_rate=scenario.get('bound_rate'))[0]
        historical = history_cache[key]
        if not any(r['scenario']==scenario['key'] for r in history_rows):
            history_rows.extend(dict(customer=name,scenario=scenario['key'],family=scenario['family'],setting=scenario['label'],
                history_coverage=cc.verification_coverage,bias_bound_MW=cc.bias_bound_rate*cc.load_mw,**h) for h in historical)
        reference = reference_for(engine, cc); set_risk(engine, cc, reference)
        full_key = (cc.tail_level, cc.risk_budget_mwh)
        for item in [*historical, dict(count=0, duration_hours=cc.true_duration_hours, truth_retained=True)]:
            count = item['count']
            if any(r['scenario']==scenario['key'] and r['history_count']==count for r in rows):
                existing = next(r for r in rows if r['scenario']==scenario['key'] and r['history_count']==count)
                if count==0: full_cache[full_key]=existing
                continue
            engine.clear_geometry()
            capacity_key=(cc.tail_level,cc.risk_budget_mwh,item['duration_hours'],count==0)
            if capacity_key in capacity_cache:
                value=capacity_cache[capacity_key]
            elif count==0 and full_key in full_cache:
                value = full_cache[full_key]
                value = {k:value[k] for k in ('total_lower_MW','total_upper_MW','net_lower_MW','net_upper_MW','risk_lower_MWh','risk_upper_MWh','root_evaluations')}
            else:
                seed = old_full if count==0 else seeds.get(count)
                if count>0:
                    candidates = [r for r in rows if r['history_count']==count]
                    if candidates: seed=candidates[-1]
                value = checked_capacity(engine, item['duration_hours'], count==0, seed)
            capacity_cache[capacity_key]=value
            result = row_result(name, scenario, count, item, value, cc, resolution, reference)
            rows.append(result)
            if count==0: full_cache[full_key]=result
            write_json(destination, dict(resolution=list(resolution), rows=rows, history_rows=history_rows,
                physical_state=dict(initial_MWh=state_initial,consumption_upper_MWh=state_consumption)))
            say(f'{LABEL[name]} / {scenario["family"]} {scenario["label"]} / {count or "完整"}：'
                f'{result["net_lower_MW"]:.2f}—{result["net_upper_MW"]:.2f}兆瓦')
    if (selected_keys is None or (filename and '_verification' in filename)) and not any(r['scenario']=='underestimated_bound' for r in history_rows):
        stress=historical_case(c,public,maximum,data,model,maxima,name,bound_rate=POLICY.error_sd_rates[0])[0]
        history_rows.extend(dict(customer=name,scenario='underestimated_bound',family='低估误差范围',setting='143兆瓦',**h) for h in stress)
        write_json(destination,dict(resolution=list(resolution),rows=rows,history_rows=history_rows,
            physical_state=dict(initial_MWh=state_initial,consumption_upper_MWh=state_consumption)))
    return rows


def convergence():
    names = ('正态1', 'LSE B')  # 正态参照及记录最多、持续性较强的拟合客户；选择先于计算结果。
    all_rows = []; level = 0
    while level < len(POLICY.numerical_resolutions):
        resolution = POLICY.numerical_resolutions[level]
        jobs = []
        with ProcessPoolExecutor(max_workers=2) as pool:
            for j,name in enumerate(names):
                # 两个独立客户各用11核；只计算99%下的三个代表点。
                jobs.append(pool.submit(customer_job, name, resolution, CORES//2+(j==0 and CORES%2), ['tail_0.99'],
                    f'precision_{FILE_CUSTOMER[name]}_{resolution[0]}.json'))
            for task in as_completed(jobs): all_rows.extend(task.result())
        if level:
            current = pd.DataFrame(all_rows)
            comparisons = []
            previous_resolution = POLICY.numerical_resolutions[level-1][0]
            for name in names:
                for count in (5,20,0):
                    pair = current[(current.customer==name)&(current.history_count==count)]
                    a=pair[pair.future_paths==previous_resolution].iloc[0]; b=pair[pair.future_paths==resolution[0]].iloc[0]
                    comparisons.append(abs((a.net_lower_MW+a.net_upper_MW-b.net_lower_MW-b.net_upper_MW)/2))
                stages={}
                for r in (previous_resolution,resolution[0]):
                    g=current[(current.customer==name)&(current.future_paths==r)].set_index('history_count')
                    mid=(g.net_lower_MW+g.net_upper_MW)/2
                    stages[r]=(float(mid.loc[20]-mid.loc[5]),float(mid.loc[0]-mid.loc[20]))
                comparisons.extend(abs(x-y) for x,y in zip(stages[previous_resolution],stages[resolution[0]]))
            maximum = max(comparisons)
            say(f'99%数值加密 {previous_resolution}→{resolution[0]}：最大容量变化{maximum:.4f}兆瓦')
            if maximum <= POLICY.refinement_trigger_mw:
                break
        level += 1
    pd.DataFrame(all_rows).to_csv(DEST/'tail99_refinement.csv', index=False, encoding='utf-8-sig')
    write_json(DEST/'precision.json', dict(resolution=list(resolution), numerical_trigger_MW=POLICY.refinement_trigger_mw,
        converged=bool(level<len(POLICY.numerical_resolutions)-1 or maximum<=POLICY.refinement_trigger_mw),
        maximum_change_MW=maximum, names=list(names), tail_level=.99))
    if maximum > POLICY.refinement_trigger_mw:
        raise RuntimeError('256条轨迹仍超过1MW加密标准，需要继续检查受影响端点')
    return resolution


def export():
    rows=[];history=[]
    for name in NAMES:
        p=DEST/f'calc_{FILE_CUSTOMER[name]}.json'
        if p.exists():
            result=read(p);rows.extend(result['rows']);history.extend(result.get('history_rows',[]))
    if not rows:return
    table=pd.DataFrame(rows)
    table.to_csv(DEST/'capacity_risk_bounds.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(history).to_csv(DEST/'historical_verification.csv',index=False,encoding='utf-8-sig')
    changes=[]
    for (name, scenario), g in table.groupby(['customer','scenario'], sort=False):
        g=g[g.history_count.isin([0,5,20])]
        if set(g.history_count)!={0,5,20}:continue
        a,b,o=(g[g.history_count==n].iloc[0] for n in (5,20,0))
        changes.append(dict(客户=LABEL[name],情景=scenario,参数=a.family,取值=a.setting,
            五次容量下界_兆瓦=a.net_lower_MW,五次容量上界_兆瓦=a.net_upper_MW,
            二十次容量下界_兆瓦=b.net_lower_MW,二十次容量上界_兆瓦=b.net_upper_MW,
            完整信息下界_兆瓦=o.net_lower_MW,完整信息上界_兆瓦=o.net_upper_MW,
            资料收益下界_兆瓦=b.net_lower_MW-a.net_upper_MW,资料收益上界_兆瓦=b.net_upper_MW-a.net_lower_MW,
            剩余折减下界_兆瓦=o.net_lower_MW-b.net_upper_MW,剩余折减上界_兆瓦=o.net_upper_MW-b.net_lower_MW,
            五次真实历史保留=bool(a.truth_retained),二十次真实历史保留=bool(b.truth_retained),
            无客户基准_兆瓦=a.reference_MW,年度尾部水平=a.tail_level,缺供上限_兆瓦时=a.risk_budget_MWh,
            未来轨迹数=int(a.future_paths),分位节点数=int(a.quadrature_nodes)))
    pd.DataFrame(changes).to_csv(DEST/'sensitivity_summary.csv', index=False, encoding='utf-8-sig')


def merge_customer(name,completed_label=None):
    labels=('风险','核验') if completed_label is None else (completed_label,)
    paths=[DEST/f'calc_{FILE_CUSTOMER[name]}.json',*[DEST/f'calc_{FILE_CUSTOMER[name]}_{FILE_TASK[label]}.json' for label in labels]]
    rows={};history={};resolution=None;physical=None
    for p in paths:
        if not p.exists():continue
        value=read(p);resolution=value['resolution'];physical=value.get('physical_state',physical)
        rows.update({(r['scenario'],r['history_count']):r for r in value['rows']})
        history.update({(r['scenario'],r['count']):r for r in value.get('history_rows',[])})
    write_json(paths[0],dict(resolution=resolution,rows=list(rows.values()),history_rows=list(history.values()),physical_state=physical))


def pairing_job(name,threads):
    threads_for_worker(threads)
    source=read(DEST/f'calc_{FILE_CUSTOMER[name]}.json');rows=source['rows'];resolution=tuple(source['resolution'])
    stages={key:{r['history_count']:r for r in rows if r['scenario']==key} for key in ('central','tail_0.99')}
    central,high=stages['central'],stages['tail_0.99']
    old_upper=central[0]['net_upper_MW']-central[20]['net_lower_MW']
    new_lower=high[0]['net_lower_MW']-high[20]['net_upper_MW']
    if new_lower-old_upper<=POLICY.refinement_trigger_mw:
        result=dict(customer=name,performed=False,reason='99%剩余信息折减未比中央档扩大1MW以上')
    else:
        c=replace(parameters(),tail_level=.99,future_paths=resolution[0],quadrature_nodes=resolution[1])
        choice=load_choice(c);model=profiles(c)
        engine=SparsePortfolioAnnual(c,choice,future_power(c,model,name)[None,:,:],high[0]['total_upper_MW'])
        value=checked_capacity(engine,c.true_duration_hours,False,high[0])
        result=dict(customer=name,performed=True,value=value,
            duration_deduction_lower_MW=max(0.,value['net_lower_MW']-high[20]['net_upper_MW']),
            duration_deduction_upper_MW=max(0.,value['net_upper_MW']-high[20]['net_lower_MW']),
            pairing_deduction_lower_MW=max(0.,high[0]['net_lower_MW']-value['net_upper_MW']),
            pairing_deduction_upper_MW=max(0.,high[0]['net_upper_MW']-value['net_lower_MW']))
    write_json(DEST/f'paired_{FILE_CUSTOMER[name]}.json',result)
    return result


def pairing():
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(pairing_job,name,CORES//2+(j==0 and CORES%2)) for j,name in enumerate(NAMES)]
        for task in as_completed(pending):results.append(task.result())
    write_json(DEST/'tail99_decomposition.json',dict(results=results))
    rows=[]
    for r in results:
        if not r['performed']:continue
        rows.append(dict(客户=LABEL[r['customer']],持续能力未确认下界_兆瓦=r['duration_deduction_lower_MW'],
            持续能力未确认上界_兆瓦=r['duration_deduction_upper_MW'],
            未来配对未知下界_兆瓦=r['pairing_deduction_lower_MW'],未来配对未知上界_兆瓦=r['pairing_deduction_upper_MW']))
    pd.DataFrame(rows).to_csv(DEST/'tail99_decomposition.csv',index=False,encoding='utf-8-sig')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','convergence','core','pairing','all'], default='all')
    parser.add_argument('--customer',choices=NAMES,help='接续单个客户的补算；默认四类全部计算')
    args=parser.parse_args()
    if args.stage in ('prepare','all'):prepare()
    if args.stage in ('convergence','all'):convergence()
    if args.stage in ('core','all'):
        resolution=tuple(read(DEST/'precision.json')['resolution'])
        base=parameters();risk_keys=['central',*[r['key'] for r in scenarios(base) if r['family'] in ('年度风险水平','新增缺供上限')]]
        evidence_keys=[r['key'] for r in scenarios(base) if r['key'] not in risk_keys]
        active_names=(args.customer,) if args.customer else NAMES
        tasks=[(name,label,keys) for name in active_names for label,keys in [('风险',risk_keys),('核验',evidence_keys)]]
        workers=min(6,CORES,len(tasks))  # 独立批次共享全部核心，每个工作进程的线程数保持固定。
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(customer_job,name,resolution,CORES//workers+(j<CORES%workers),keys,
                f'calc_{FILE_CUSTOMER[name]}_{FILE_TASK[label]}.json'):(name,label) for j,(name,label,keys) in enumerate(tasks)}
            for task in as_completed(pending):
                try:task.result()
                except Exception as error:
                    say(f'{pending[task][0]}补算中断：{error!r}');raise
                merge_customer(*pending[task]);export()
        for name in NAMES:merge_customer(name)
        export()
    if args.stage in ('pairing','all'):pairing()


if __name__=='__main__':main()
