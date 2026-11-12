"""实验二单户与实验三组合共用输入：独立功率、共同及个体基线扰动。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm,truncnorm
from numba import set_num_threads

from common import ROOT,OUT,AUDIT,CORES,write_json
from normal_case_parameters import NormalCaseParameters
from normal_customer import build_history,normal_power_paths,HISTORICAL_SCORES
from customer_history_inputs import CALENDAR,_load_draw
from customer_profiles import _empirical_profile,_empirical_nodes,_ar_scores,_reorder_nodes


@dataclass(frozen=True)
class PortfolioParameters(NormalCaseParameters):
    maximum_extension_hours: float | None = None  # 用户此前认可取消额外延长4小时上限；沿用实验二已归档的延长邀约分支。
    empirical_users: tuple = ('LSE A','LSE B','LSE C')  # 三个匿名实测来源。
    normal_sizes: tuple = (1,2,3,4)  # 本轮用户认可的同类正态客户聚合规模。
    history_power_seed: int = 2026110421  # 既定模拟种子，按客户、调用分流。
    individual_load_seed: int = 2026110521  # 既定逐户负荷模拟种子。
    common_error_amplitude: float = .02  # 既定共同/个体方差拆分情景，见参数统一与实验分工。
    individual_error_amplitude: float = .2  # 既定方差拆分情景；总误差率另由公开资料给定。
    state_power_ratio: float = 1.5  # 既定普通负荷初态1500×8760/1000，属于数值设定。


DEST=OUT/'exp3'
INPUT=DEST/'portfolio_iso_inputs.npz'
AUDIT_FILE=AUDIT/'portfolio_input_check.json'


def parameters(config=None):
    active=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))['invitation_comparison']['parameters']
    if config:active.update(json.loads(Path(config).read_text(encoding='utf-8-sig')))
    return PortfolioParameters(**active)


def definitions(c):
    portfolios=[]
    for size in c.normal_sizes:
        portfolios.append(dict(name=f'正态{size}户',kind='normal',members=[f'正态{i+1}' for i in range(size)]))
    for user in c.empirical_users:portfolios.append(dict(name=user,kind='empirical_single',members=[user]))
    from itertools import combinations
    for pair in combinations(c.empirical_users,2):portfolios.append(dict(name='＋'.join(pair),kind='empirical_pair',members=list(pair)))
    portfolios.append(dict(name='＋'.join(c.empirical_users),kind='empirical_triple',members=list(c.empirical_users)))
    for item in portfolios:item['weights']=[1/len(item['members'])]*len(item['members'])
    return portfolios


def profiles(c):
    result={}
    for name in c.empirical_users:
        p,_=_empirical_profile(name)
        values=np.asarray(p['marginal']['support']);prob=np.asarray(p['marginal']['probabilities'])
        raw=pd.read_csv(OUT/'hubei/meter_series.csv')
        part=raw[raw.user==name];shape=np.sort((part.actual_kW/part.target_kW).to_numpy())  # 用户认可的原表实际值/目标值映射，再乘该户申报量。
        result[name]=dict(kind='empirical',rho=float(p['gaussian_rank_AR1']),values=values,probabilities=prob,
            nodes=_empirical_nodes(shape,c.quadrature_nodes),records=len(shape),events=p['events'])
    if c.response_sd_ratio==0:
        nodes=np.full(c.quadrature_nodes,c.response_mean_ratio)  # 沿用正态基准已有的零方差情景。
    else:
        a=-c.response_mean_ratio/c.response_sd_ratio
        edges=truncnorm.ppf(np.linspace(0,1,c.quadrature_nodes+1),a,np.inf,loc=c.response_mean_ratio,scale=c.response_sd_ratio)
        z=(edges-c.response_mean_ratio)/c.response_sd_ratio
        nodes=c.response_mean_ratio+c.response_sd_ratio*c.quadrature_nodes*np.diff(-norm.pdf(z))/norm.sf(a)
    for i in range(max(c.normal_sizes)):
        result[f'正态{i+1}']=dict(kind='normal',rho=c.response_rho15,nodes=nodes.copy(),records=0,events=0)
    return result


def inverse(profile,score,c):
    u=norm.cdf(score)
    if profile['kind']=='normal':
        if c.response_sd_ratio==0:return np.full_like(score,c.response_mean_ratio)
        return truncnorm.ppf(u,-c.response_mean_ratio/c.response_sd_ratio,np.inf,loc=c.response_mean_ratio,scale=c.response_sd_ratio)
    index=np.searchsorted(np.cumsum(profile['probabilities']),u,side='left')
    return profile['values'][np.minimum(index,len(profile['values'])-1)]


def make_inputs(c):
    set_num_threads(CORES)
    public,normal_meter,_,_,_,_,normal_source=build_history(c)
    with np.load(CALENDAR) as z:dates=z['dates'].copy()
    days=len(dates);calls=public['call_ids'];n=public['n']
    prior=np.unique(np.concatenate([d['prior'] for d in public['designs']]))
    names=['正态1',*c.empirical_users,*[f'正态{i+1}' for i in range(1,max(c.normal_sizes))]]
    model=profiles(c)
    current=np.concatenate([d['day']*96+d['q'] for d in public['designs']])
    gamma=c.common_error_amplitude**2/(c.common_error_amplitude**2+c.individual_error_amplitude**2)
    # Q为实验二已有的标准负荷扰动；正交旋转保留Q，并得到相互独立的共同项与第一户个体项。
    Q=_load_draw(days,1.,reference=c.load_mw,rho=c.load_rho15,daily_share=c.load_daily_variance_share,seed=c.load_seed)-c.load_mw
    V=_load_draw(days,1.,reference=c.load_mw,rho=c.load_rho15,daily_share=c.load_daily_variance_share,
        seed=np.random.SeedSequence([c.individual_load_seed,0]))-c.load_mw
    common=np.sqrt(gamma)*Q+np.sqrt(1-gamma)*V
    frozen=pd.read_csv(HISTORICAL_SCORES)['标准正态分位'].to_numpy()
    availability=np.clip((np.minimum(public['left']+.25,np.minimum(public['stops'],public['starts']+c.true_duration_hours))
        -np.maximum(public['left'],public['starts']))/.25,0,1)  # 15分钟读数与真实持续能力的时间交叠。
    reference=np.ones(days,bool);reference[[d['day'] for d in public['designs']]]=False
    baseline=[];meter=[];original=[];power={};past_power=[];clipped=[];actual_error=[]
    for i,name in enumerate(names):
        p=model[name]
        if i==0:shock=Q
        else:
            private=_load_draw(days,1.,reference=c.load_mw,rho=c.load_rho15,daily_share=c.load_daily_variance_share,
                seed=np.random.SeedSequence([c.individual_load_seed,i]))-c.load_mw
            shock=np.sqrt(gamma)*common+np.sqrt(1-gamma)*private
        load=c.load_mw+public['error_scale_MW']*shock
        load[reference]+=c.error_mean_rate*c.load_mw
        b=np.concatenate([load[d['prior']][:,d['q']].mean(axis=0) for d in public['designs']])
        scores=np.empty(n)
        for event in range(public['J']):
            mask=calls==event
            innovations=frozen[mask] if i==0 else np.random.default_rng(np.random.SeedSequence([c.history_power_seed,i,event])).standard_normal(mask.sum())
            # 每个调用重新起一条已拟合的时序；用户之间使用不同随机流。
            z=innovations.copy()
            for j in range(1,len(z)):z[j]=p['rho']*z[j-1]+np.sqrt(1-p['rho']**2)*innovations[j]
            scores[mask]=z
        potential=c.nomination_mw*inverse(p,scores,c)
        untreated=load.ravel()[current]
        # 调用期实际响应受可用功率和当时用电共同限制，保证电表用电非负。
        delivered=np.minimum(potential,untreated)*availability
        y=untreated-delivered
        baseline.append(b);meter.append(y);original.append(load[prior]);past_power.append(potential)
        actual_error.append(b-untreated-c.error_mean_rate*c.load_mw)
        clipped.append(int(np.sum(potential>untreated)))
        if i==0:future=normal_power_paths(c)
        else:
            innovations=np.random.default_rng(np.random.SeedSequence([c.future_seed,i])).standard_normal((c.future_paths,35040))
            scores=_ar_scores(innovations,p['rho'])
            future=c.nomination_mw*_reorder_nodes(scores,p['nodes'])
        power[name]=future
    baseline=np.asarray(baseline);meter=np.asarray(meter);original=np.asarray(original)
    difference=float(max(np.max(abs(baseline[0]-public['baseline'])),np.max(abs(meter[0]-normal_meter))))
    if difference>c.solver_tolerance:raise RuntimeError('正态单户未复现实验二历史输入')
    portfolios=definitions(c);portfolio_weights=np.zeros((len(portfolios),len(names)))
    for j,item in enumerate(portfolios):
        for member,w in zip(item['members'],item['weights']):portfolio_weights[j,names.index(member)]=w
    np.savez_compressed(INPUT,prototype_names=np.asarray(names),portfolio_names=np.asarray([p['name'] for p in portfolios]),
        portfolio_weights=portfolio_weights,call_baselines=baseline,call_meters=meter,
        original_meters=original,original_days=prior,dates=dates,call_ids=calls,call_left=public['left'],
        call_starts=public['starts'],call_stops=public['stops'])
    rows=[];check=[]
    for item in definitions(c):
        P=np.array([power[name]*w for name,w in zip(item['members'],item['weights'])]);total=P.sum(axis=0)
        theoretical=float(np.sqrt(sum((w*c.nomination_mw*np.std(model[name]['nodes']))**2 for name,w in zip(item['members'],item['weights']))))
        sampled=float(total.std());corr=np.corrcoef(P.reshape(len(P),-1)) if len(P)>1 else np.array([[1.]])
        rows.append(dict(组合=item['name'],户数=len(P),总申报量_兆瓦=c.nomination_mw,总正常负荷_兆瓦=c.load_mw,
            可用功率均值_兆瓦=float(total.mean()),独立叠加积分节点标准差_兆瓦=theoretical,生成组合标准差_兆瓦=sampled,
            实测拟合或正态参照=item['kind']))
        check.append(dict(name=item['name'],members=item['members'],weights=item['weights'],sampled_correlation=corr.tolist(),
            independent_node_SD_MW=theoretical,sampled_SD_MW=sampled,sum_check_MW=float(np.max(abs(total-np.sum(P,axis=0))))))
    pd.DataFrame(rows).to_csv(DEST/'portfolio_power.csv',index=False,encoding='utf-8-sig')
    audit=dict(parameters=c.values(),prototype_names=names,source='hubei/meter_series.csv；实验二正态参照及固定历史分位',
        normal_single_history_max_difference_MW=difference,baseline_common_variance_fraction=gamma,
        prototype_clipped_history_points=dict(zip(names,clipped)),records_per_customer=n,original_input_days=len(prior),
        portfolios=check,cores=CORES,future_paths=c.future_paths,quarter_points=35040,
        generation='单户边际沿用原拟合；各户功率独立随机流；基线负荷含同一共同扰动及各户独立扰动。')
    write_json(AUDIT_FILE,audit)
    return public,model,dict(names=names,baseline=baseline,meter=meter,prior=prior,original=original,availability=availability,
        past_power=np.asarray(past_power),actual_error=np.asarray(actual_error),power=power,gamma=gamma,normal_source=normal_source),rows


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    c=parameters();_,_,_,rows=make_inputs(c)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
