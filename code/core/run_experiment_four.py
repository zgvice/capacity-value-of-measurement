"""实验四：真实调用留出检验、统计覆盖及完整年度敏感性。"""
from __future__ import annotations
import argparse,json,time,os
from dataclasses import fields,replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz,cholesky
from scipy.stats import norm
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from common import ROOT,OUT,AUDIT,CORES,write_json
from experiment_four_parameters import ExperimentFourParameters
from normal_customer import build_history,normal_power_paths
from historical_duration import HistoricalDuration
from customer_history_inputs import CALENDAR,TASK_FILE,_load_draw
from portfolio_inputs import profiles,inverse
from customer_profiles import _ar_scores,_reorder_nodes
from portfolio_annual import frozen_choice
from portfolio_sparse_physics import SparsePortfolioAnnual
from experiment_four_history import verification_history

DEST=OUT/'exp4'
SAVED=DEST/'results.json'


def say(s):print(time.strftime('%H:%M:%S'),s,flush=True)


def inputs():
    old=json.loads((AUDIT/'portfolio_capacity.json').read_text(encoding='utf-8'))
    values=dict(old['parameters']);config=os.environ.get('EXPERIMENT_FOUR_CONFIG')
    if config:
        changed=json.loads(Path(config).read_text(encoding='utf-8-sig'))
        for key,value in changed.items():
            if key in values and value!=values[key]:
                raise ValueError(f'{key}是与前三个实验共用的参数，请先更新前置实验；实验四可覆盖新增事件和敏感性参数。')
        values.update(changed)
    c=ExperimentFourParameters(**values)
    public,_,bound,maximum,_,_,_=build_history(c)
    calibration=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))
    cc=calibration['history_calibration']['calibration']
    critical=np.where(public['error_families']=='global',cc['global']['joint_multiplier'],cc['other']['joint_multiplier'])
    public['bounds']=public['error_stat_SD_MW']*critical
    DEST.mkdir(parents=True,exist_ok=True)
    public,rule=verification_history(c,public,DEST)
    public['verification_rule']=rule
    data=np.load(OUT/'exp3/portfolio_iso_inputs.npz')
    model=profiles(c)
    return c,public,bound,maximum,data,model,calibration,old


def future_power(c,model,name):
    if name=='正态1':return normal_power_paths(c)
    i=['正态1',*c.empirical_users].index(name)  # 实验三portfolio_inputs.py保持相同的逐户随机流编号。
    rng=np.random.default_rng(np.random.SeedSequence([c.future_seed,i]))
    score=_ar_scores(rng.standard_normal((c.future_paths,35040)),model[name]['rho'])  # 原8760小时×四个15分钟。
    return c.nomination_mw*_reorder_nodes(score,model[name]['nodes'])


def untreated_load(c,public,index):
    # 作者按实验三原种子重建调用日正常用电，用于事后核对；不进入ISO资料或历史约束。
    with np.load(CALENDAR) as z:days=len(z['dates'])
    Q=_load_draw(days,1.,reference=c.load_mw,rho=c.load_rho15,daily_share=c.load_daily_variance_share,seed=c.load_seed)-c.load_mw
    if index==0:shock=Q
    else:
        gamma=c.common_error_amplitude**2/(c.common_error_amplitude**2+c.individual_error_amplitude**2)
        V=_load_draw(days,1.,reference=c.load_mw,rho=c.load_rho15,daily_share=c.load_daily_variance_share,
            seed=np.random.SeedSequence([c.individual_load_seed,0]))-c.load_mw
        common=np.sqrt(gamma)*Q+np.sqrt(1-gamma)*V
        private=_load_draw(days,1.,reference=c.load_mw,rho=c.load_rho15,daily_share=c.load_daily_variance_share,
            seed=np.random.SeedSequence([c.individual_load_seed,index]))-c.load_mw
        shock=np.sqrt(gamma)*common+np.sqrt(1-gamma)*private
    load=c.load_mw+public['error_scale_MW']*shock
    return np.concatenate([load[d['day'],d['q']] for d in public['designs']])


def subset(public,selected):
    # 只保留已掌握记录支持的原统计约束；未使用待检验事件的读数或新建统计量。
    rows=np.asarray(public['W'][:,~selected].getnnz(axis=1)).ravel()==0
    p={k:public[k][selected] for k in ('baseline','left','starts','stops','call_ids')}
    p.update(n=int(selected.sum()),J=public['J'],W=public['W'][rows][:,selected],bounds=public['bounds'][rows],
        required=np.zeros(int(rows.sum()),int),mean_error_MW=public['mean_error_MW'])
    return p,rows


def duration(c,public,meter,P,maximum):
    return HistoricalDuration(public,meter,public['J'],P,bias_bound=c.load_mw*c.bias_bound_rate,
        solver_tolerance=c.solver_tolerance,duration_check_tolerance=c.duration_check_tolerance_hours).minimum(
            maximum,minimum=c.verified_min_duration_hours)


def events(c,public,bound,maximum,data,model,calibration,old,result):
    held=set(c.winter_dates+c.summer_dates)
    dates=np.array([str(d['task']['date']) for d in public['designs']])
    selected=~np.isin(dates[public['call_ids']],list(held))
    p,rows=subset(public,selected)
    call_tasks=pd.read_csv(TASK_FILE)
    call_tasks=call_tasks[call_tasks.date.isin(held)].copy()
    call_tasks.to_csv(DEST/'call_tasks.csv',index=False,encoding='utf-8-sig')
    source=calibration['invitation_comparison']
    choice=None
    rule=public['verification_rule']
    old_rows=result.get('heldout',{})
    if result.get('historical_rule')!=rule:result['heldout']={}
    result['historical_rule']=rule
    result.setdefault('heldout',{})
    held_rows=[];trace=[]
    names=['正态1',*c.empirical_users]
    prototype=list(data['prototype_names'])
    for name in names:
        i=prototype.index(name);p['baseline']=data['call_baselines'][i,selected]
        true_load=untreated_load(c,public,i)
        P=bound if name=='正态1' else c.nomination_mw*float(model[name]['values'].max())
        H=duration(c,p,data['call_meters'][i,selected],P,maximum)
        full=dict(public,baseline=data['call_baselines'][i])
        H_all=duration(c,full,data['call_meters'][i],P,maximum)
        group='正态1户' if name=='正态1' else name
        if name not in result['heldout']:
            if choice is None:choice=frozen_choice(c,source)
            powers=future_power(c,model,name)
            engine=SparsePortfolioAnnual(c,choice,powers[None,:,:],old['groups'][group]['full_information']['total_upper_MW'])
            known=engine.capacity(engine.duration_segments([H]),False)
            # 两套资料使用同一联合校准边界。能确认的时长不变时才复用年度容量。
            complete_history=known if abs(H-H_all)<=c.solver_tolerance else engine.capacity(engine.duration_segments([H_all]),False,
                lower=known['total_lower_MW'],upper=old['groups'][group]['full_information']['total_upper_MW'])
            result['heldout'][name]=dict(duration_hours=H,capacity=known,training_dates=dates[~np.isin(dates,list(held))].tolist(),
                training_calls=int(sum(~np.isin(dates,list(held)))),training_quarters=int(selected.sum()),
                training_statistics=int(rows.sum()),full_information=old['groups'][group]['full_information'],
                all_history_duration_hours=H_all,all_history_capacity=complete_history)
            write_json(SAVED,result)
            say(f'{name}：留出五日后确认{H:.3f}小时，全年净容量{known["net_lower_MW"]:.2f}—{known["net_upper_MW"]:.2f}兆瓦')
        info=result['heldout'][name]
        for j,d in enumerate(public['designs']):
            if str(d['task']['date']) not in held:continue
            ids=np.flatnonzero(public['call_ids']==j);task=d['task']
            # 当前14.5小时模拟能力覆盖既有最长14.5小时指令；原历史真实交付按同一固定随机流重建。
            from normal_customer import HISTORICAL_SCORES
            frozen=pd.read_csv(HISTORICAL_SCORES)['标准正态分位'].to_numpy()
            innovations=frozen[ids] if name=='正态1' else np.random.default_rng(
                np.random.SeedSequence([c.history_power_seed,names.index(name),j])).standard_normal(len(ids))
            z=innovations.copy();rho=model[name]['rho']
            for k in range(1,len(z)):z[k]=rho*z[k-1]+np.sqrt(1-rho*rho)*innovations[k]
            potential=c.nomination_mw*inverse(model[name],z,c)
            left=public['left'][ids];start=task['start_hour'];stop=task['end_hour']
            overlap=np.maximum(0.,np.minimum(left+.25,stop)-np.maximum(left,start))  # .25小时来自原15分钟计量。
            available=np.maximum(0.,np.minimum(left+.25,min(stop,start+c.true_duration_hours))-np.maximum(left,start))
            delivered=np.minimum(potential,true_load[ids])*available
            baseline=data['call_baselines'][i,ids];meter=data['call_meters'][i,ids]
            if np.max(abs(true_load[ids]-delivered/.25-meter))>c.solver_tolerance:
                raise RuntimeError('事件交付与原15分钟电表不一致')
            physical_energy=float(delivered.sum());required=c.nomination_mw*float(task['duration_hours'])
            shortage=float(np.maximum(c.nomination_mw*overlap-delivered,0).sum())
            confirm_missing=max(float(task['duration_hours'])-H,0.)
            held_rows.append(dict(客户=name,日期=task['date'],调用起点_小时=start,调用终点_小时=stop,
                调用时长_小时=task['duration_hours'],历史确认时长_小时=H,历史未确认的调用时长_小时=confirm_missing,
                真实持续能力_小时=c.true_duration_hours,要求电量_兆瓦时=required,真实调节电量_兆瓦时=physical_energy,
                逐时不足电量_兆瓦时=shortage,平均调节_兆瓦=physical_energy/task['duration_hours'],
                基线电表显示的平均调节_兆瓦=float(((baseline-meter)*overlap).sum()/task['duration_hours']),
                全年净认证容量下界_兆瓦=info['capacity']['net_lower_MW'],全年净认证容量上界_兆瓦=info['capacity']['net_upper_MW']))
            for k,at in enumerate(ids):
                trace.append(dict(客户=name,日期=task['date'],时段开始_小时=float(left[k]),时段长度_小时=float(overlap[k]),
                    指令功率_兆瓦=c.nomination_mw,真实调节_兆瓦=float(delivered[k]/overlap[k]),
                    基线_兆瓦=float(baseline[k]),电表_兆瓦=float(meter[k]),
                    历史可确认_是否=float(left[k])<start+H))
    pd.DataFrame(held_rows).to_csv(DEST/'event_holdout.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(trace).to_csv(DEST/'event_trace.csv',index=False,encoding='utf-8-sig')
    result['events']=dict(dates=sorted(held),source_tasks=len(call_tasks),representative_dates=len(held),
        source='inputs/frozen/call_tasks.csv；各日最长任务，规则沿用实验二。全部区域/产品任务另表完整保留。',
        all_meter_points_preserved=True,annual_atoms=40300,annual_hours=8760,annual_paths=1788)
    write_json(SAVED,result)


def coverage(c,public,bound,calibration,result):
    F,W=public['error_operator'],public['W'];n=public['n']
    day_size=96  # 原15分钟计量：每日96点。
    daily=c.load_daily_variance_share+(1-c.load_daily_variance_share)*toeplitz(c.load_rho15**np.arange(day_size))
    covariance=np.zeros((n,n))
    for day in np.unique(F.indices//day_size):
        part=F[:,day*day_size:(day+1)*day_size].tocsr();ids=np.unique(part.nonzero()[0]);block=part[ids].toarray()
        covariance[np.ix_(ids,ids)]+=block@daily@block.T
    op=np.asarray(W@cholesky(covariance,lower=True))*public['error_scale_MW']
    rng=np.random.default_rng(c.coverage_seed);retained=error_ok=power_ok=0;ratios=[]
    with threadpool_limits(limits=CORES,user_api='blas'):
        for first in range(0,c.coverage_draws,c.coverage_batch):
            m=min(c.coverage_batch,c.coverage_draws-first)
            e=rng.standard_normal((m,n))@op.T
            good_e=(abs(e)<=public['bounds'][None,:]).all(axis=1)
            # 正态基准相邻功率独立；与实验二相同的非负正态和全点功率上界。
            score=rng.standard_normal((m,n));power=c.nomination_mw*inverse(profiles(c)['正态1'],score,c)
            good_p=(power<=bound).all(axis=1)
            error_ok+=int(good_e.sum());power_ok+=int(good_p.sum());retained+=int((good_e&good_p).sum())
    result['coverage']=dict(repetitions=c.coverage_draws,seed=c.coverage_seed,
        true_complete_error_path_retained=error_ok/c.coverage_draws,
        true_complete_power_path_retained=power_ok/c.coverage_draws,
        true_complete_physical_history_retained=retained/c.coverage_draws,
        retained_count=retained,required_coverage=c.verification_coverage,
        data='独立模拟；沿用原历史日期、完整15分钟误差算子、联合校准边界和功率分布。')
    write_json(SAVED,result);say(f'独立重复{c.coverage_draws}次：完整真实历史保留率{retained/c.coverage_draws:.2%}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['events','coverage','all'],default='all');parser.add_argument('--config');args=parser.parse_args()
    if args.config:os.environ['EXPERIMENT_FOUR_CONFIG']=str(Path(args.config).resolve())
    DEST.mkdir(parents=True,exist_ok=True);set_num_threads(CORES)
    c,public,bound,maximum,data,model,calibration,old=inputs()
    result=json.loads(SAVED.read_text(encoding='utf-8')) if SAVED.exists() else dict(parameters=c.values(),cores=CORES)
    expected=json.loads(json.dumps(c.values()))
    if any(expected[k]!=v for k,v in result['parameters'].items() if k in expected):
        result=dict(parameters=c.values(),cores=CORES)
    result['parameters']=c.values()
    # 参数表从已注明来源的配置文件生成；继承参数的来源保留在原入口。
    source=Path(__file__).with_name('experiment_four_parameters.py').read_text(encoding='utf-8')
    notes={line.strip().split(':',1)[0]:line.split(' # ',1)[1] for line in source.splitlines() if ' # ' in line and ' = ' in line}
    pd.DataFrame([dict(参数=k,取值=v,来源=notes.get(k,'继承实验三；portfolio_inputs.py及normal_case_parameters.py的同名参数注释')) for k,v in c.values().items()]).to_csv(DEST/'parameters.csv',index=False,encoding='utf-8-sig')
    if args.stage in ('events','all'):events(c,public,bound,maximum,data,model,calibration,old,result)
    if args.stage in ('coverage','all'):coverage(c,public,bound,calibration,result)


if __name__=='__main__':main()
