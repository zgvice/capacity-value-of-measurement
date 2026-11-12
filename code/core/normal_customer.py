"""正态客户的历史电表、基线和全年功率生成。"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.stats import norm, truncnorm
from common import ROOT, DATA
from customer_profiles import _ar_scores, _reorder_nodes
from customer_history_inputs import CALENDAR, TASK_FILE, _load_draw

QUARTER = .25  # 原始电表每15分钟计量；换算为小时。
HISTORICAL_SCORES = DATA/'normal_power_quantiles.csv'

def annual_atom_count():
    with np.load(DATA/'annual/exact_path_groups.npz') as data:
        return len(data['atom_to_group'])

def build_history(c):
    if not (0<c.verification_coverage<1 and 0<c.error_budget_share<1 and 0<c.global_error_budget_share<1):
        raise ValueError('核验覆盖水平及预算比例须在0与1之间')
    if not (abs(c.load_rho15)<1 and abs(c.response_rho15)<1 and 0<=c.load_daily_variance_share<=1):
        raise ValueError('关联参数绝对值须小于1，日共同方差比例须在0至1之间')
    tasks = pd.read_csv(TASK_FILE).sort_values(['duration_hours','task_id'],ascending=[False,True])
    tasks = tasks.drop_duplicates('date').sort_values('date')
    with np.load(CALENDAR) as calendar:
        dates = calendar['dates']
        lookup = {str(date):i for i,date in enumerate(dates)}
        designs,calibration_counts = [],[]
        for task in tasks.to_dict('records'):
            day = lookup[str(task['date'])]
            prior = np.arange(max(0,day-c.baseline_days),day)
            prior = prior[(calendar['daytype'][prior]==calendar['daytype'][day]) &
                calendar['complete'][prior] & ~calendar['event'][prior]]
            if not len(prior):raise ValueError(f'{task["date"]}没有当前回看天数下的合格基线日')
            calibration = np.arange(max(0,day-c.error_calibration_days),day)
            calibration = calibration[(calendar['daytype'][calibration]==calendar['daytype'][day]) &
                calendar['complete'][calibration] & ~calendar['event'][calibration]]
            if not len(calibration):raise ValueError(f'{task["date"]}没有误差校准基线日')
            calibration_counts.append(len(calibration))
            q = np.arange(int(np.floor(task['start_hour']/QUARTER+c.time_tolerance_hours)),
                          int(np.ceil(task['end_hour']/QUARTER-c.time_tolerance_hours)))
            # 时间边界容差来自集中参数入口。
            designs.append(dict(task=task,day=day,prior=prior,q=q))
        inflation = np.mean([1+1/n for n in calibration_counts])
        load = _load_draw(len(dates),c.error_sd_rate*c.load_mw/np.sqrt(inflation),
            reference=c.load_mw,rho=c.load_rho15,daily_share=c.load_daily_variance_share,seed=c.load_seed)
        # 参考日与调用日的平均水平差产生指定的基线误差均值；调用日未响应负荷仍以load_mw为中心。
        mean_error = c.error_mean_rate*c.load_mw
        reference_days = np.ones(len(dates),bool)
        reference_days[[item['day'] for item in designs]] = False
        load[reference_days] += mean_error
    J = len(designs)
    baseline = np.concatenate([load[np.ix_(x['prior'],x['q'])].mean(axis=0) for x in designs])
    untreated = np.concatenate([load[x['day'],x['q']] for x in designs])
    call_ids = np.concatenate([np.full(len(x['q']),j,int) for j,x in enumerate(designs)])
    left = np.concatenate([x['q']*QUARTER for x in designs])
    starts = np.concatenate([np.full(len(x['q']),x['task']['start_hour']) for x in designs])
    stops = np.concatenate([np.full(len(x['q']),x['task']['end_hour']) for x in designs])
    n = len(left)
    historical_longest = float(max(stops-starts))
    truth_duration = c.true_duration_hours
    if truth_duration is None or not 0<=c.verified_min_duration_hours<=truth_duration:
        raise ValueError('客户真实持续能力须单独给定，且不低于已核定工程下限')
    # 记录结束后不再产生历史观测；达到最长历史邀约时，各点可行上界已不再随时长变化。
    # 这里只求历史能确认的最短时长，搜索边界与未来邀约及模拟真能力分别设置。
    maximum = max(historical_longest, c.verified_min_duration_hours)
    # 固定分位保存原记录的共同随机数；负荷和功率参数由当前配置代入。
    frozen = pd.read_csv(HISTORICAL_SCORES, float_precision='round_trip')
    innovations = frozen['标准正态分位'].to_numpy()
    if len(innovations) != n:
        raise ValueError('历史功率分位与调用记录不对齐')
    scores = innovations.copy()
    for j in range(J):
        ids = np.flatnonzero(call_ids==j)
        for k in ids[1:]:
            scores[k] = c.response_rho15*scores[k-1]+np.sqrt(1-c.response_rho15**2)*innovations[k]
    if c.response_sd_ratio==0:
        potential = np.full(n,c.nomination_mw*c.response_mean_ratio)
    else:
        a = -c.response_mean_ratio/c.response_sd_ratio
        potential = c.nomination_mw*truncnorm.ppf(norm.cdf(scores),a,np.inf,
            loc=c.response_mean_ratio,scale=c.response_sd_ratio)
    available = np.clip((np.minimum(left+QUARTER,np.minimum(stops,starts+truth_duration))-
                        np.maximum(left,starts))/QUARTER,0,1)
    delivery = potential*available
    meter = untreated-delivery
    if np.any(meter<0):raise ValueError('当前负荷与可调功率组合使调节后用电为负，需调整该参数组合')

    rr,cc,vv = [],[],[]
    i = 0
    for item in designs:
        for q in item['q']:
            rr.append(i);cc.append(item['day']*96+q);vv.append(-1.)
            rr.extend([i]*len(item['prior']));cc.extend(item['prior']*96+q)
            vv.extend([1/len(item['prior'])]*len(item['prior']));i += 1
    F = sparse.csr_matrix((vv,(rr,cc)),shape=(n,load.size))
    weights = [sparse.eye(n,format='csr')]
    required,families,labels = list(J-call_ids),['interval']*n,['15分钟']*n
    hour_bin = np.floor(np.maximum(left-starts,0)+c.hour_bin_tolerance).astype(int)  # 原小时分组容差。
    def add(mask,family,label):
        if mask.any():
            weights.append(sparse.csr_matrix(mask.astype(float)/mask.sum()))
            required.append(int(J-call_ids[mask].min()));families.append(family);labels.append(label)
    for j in range(J):
        add(call_ids==j,'event',f'调用{j}全程')
        for h in np.unique(hour_bin[call_ids==j]):add((call_ids==j)&(hour_bin==h),'event',f'调用{j}小时{h}')
    for count in range(1,J+1):
        known = call_ids>=J-count
        add(known,'global',f'{count}次全程')
        for h in np.unique(hour_bin[known]):add(known&(hour_bin==h),'global',f'{count}次小时{h}')
    W = sparse.vstack(weights,format='csr')
    M = (W@F).tocsr();M.sort_indices()
    corr = toeplitz(c.load_rho15**np.arange(96))  # 96为每日15分钟点数。
    variance = np.zeros(M.shape[0])
    for row in range(len(variance)):
        begin,end = M.indptr[row:row+2]
        ids,z = M.indices[begin:end],M.data[begin:end]
        edges = np.r_[0,np.flatnonzero(np.diff(ids//96))+1,len(ids)]
        for lo,hi in zip(edges[:-1],edges[1:]):
            q,x = ids[lo:hi]%96,z[lo:hi]
            variance[row] += c.load_daily_variance_share*x.sum()**2+(1-c.load_daily_variance_share)*x@corr[np.ix_(q,q)]@x
    error_alpha = (1-c.verification_coverage)*c.error_budget_share
    families = np.asarray(families)
    critical = np.empty(len(families))
    for mask,share in ((families=='global',c.global_error_budget_share),
                       (families!='global',1-c.global_error_budget_share)):
        critical[mask] = norm.ppf(1-error_alpha*share/(2*mask.sum()))  # 2为双侧覆盖分配。
    error_sd = c.error_sd_rate*c.load_mw
    bounds = error_sd/np.sqrt(inflation)*np.sqrt(np.maximum(variance,0))*critical
    power_alpha = (1-c.verification_coverage)*(1-c.error_budget_share)
    power_bound = c.nomination_mw*c.response_mean_ratio if c.response_sd_ratio==0 else c.nomination_mw*truncnorm.ppf(
        1-power_alpha/n,-c.response_mean_ratio/c.response_sd_ratio,np.inf,
        loc=c.response_mean_ratio,scale=c.response_sd_ratio)
    public = dict(n=n,J=J,baseline=baseline,mean_error_MW=mean_error,left=left,starts=starts,stops=stops,call_ids=call_ids,
        W=W,bounds=bounds,required=np.asarray(required),counts=tuple(range(1,J+1)),designs=designs)
    # 联合核验复用原误差算子和标准差，不向ISO提供模拟误差实现或正常用电真值。
    public.update(error_operator=F,error_scale_MW=error_sd/np.sqrt(inflation),
        error_families=families,error_critical=critical,error_stat_SD_MW=bounds/critical)
    first_mean = labels.index('1次全程')
    first = call_ids==J-1
    check = dict(parameters=c.values(),error_mean_MW=mean_error,error_SD_MW=error_sd,
        mean_bias_bound_MW=c.bias_bound_rate*c.load_mw,
        power_prediction_bound_MW=float(power_bound),min_meter_MW=float(meter.min()),
        maximum_relevant_duration_hours=maximum,historical_search_upper_hours=maximum,
        historical_longest_invitation_hours=historical_longest,true_duration_hours=truth_duration,
        original_annual_atoms=annual_atom_count(),fixed_call_count=J,quarter_points=n,
        latest_call_mean_measured_MW=float((baseline-meter)[first].mean()),
        latest_call_mean_error_bound_MW=float(bounds[first_mean]),
        largest_single_point_error_bound_MW=float(bounds[:n].max()),
        true_error_in_verification_set=bool(np.all(abs(W@(baseline-untreated-mean_error))<=bounds+c.solver_tolerance)),
        sources=dict(error_mean_rate=f'参数入口或本次配置的误差中心率：{100*c.error_mean_rate:g}%',
            error_sd_rate=f'集中参数给定的标准差率{100*c.error_sd_rate:g}%；公开资料来源见参数表'))
    records = pd.DataFrame(dict(客户='正态基准',调用编号=np.concatenate([np.full(len(x['q']),x['task']['task_id']) for x in designs]),
        读数时段开始=[(pd.Timestamp(str(dates[x['day']]))+pd.Timedelta(minutes=15*int(q))).isoformat()
                    for x in designs for q in x['q']],基线_兆瓦=baseline,电表_兆瓦=meter))
    return public,meter,float(power_bound),maximum,truth_duration,records,check


def normal_power_paths(c):
    if c.future_paths%c.quadrature_nodes:raise ValueError('未来轨迹数须为积分节点数的整数倍')
    if c.response_sd_ratio==0:return np.full((c.future_paths,35040),c.nomination_mw*c.response_mean_ratio)
    a = -c.response_mean_ratio/c.response_sd_ratio
    edges = truncnorm.ppf(np.linspace(0,1,c.quadrature_nodes+1),a,np.inf,
        loc=c.response_mean_ratio,scale=c.response_sd_ratio)
    z = (edges-c.response_mean_ratio)/c.response_sd_ratio
    nodes = c.response_mean_ratio+c.response_sd_ratio*c.quadrature_nodes*np.diff(-norm.pdf(z))/norm.sf(a)
    # 35040=8760小时×4；不改变15分钟数据网格。
    rng = np.random.default_rng(c.future_seed)
    scores = _ar_scores(rng.standard_normal((c.future_paths,35040)),c.response_rho15)
    return c.nomination_mw*_reorder_nodes(scores,nodes)

