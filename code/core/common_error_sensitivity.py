"""实验三共同误差对照：单户尺度和未来功率固定，比较四户1次及20次历史。"""
import gc
import argparse
from dataclasses import replace
import numpy as np
import pandas as pd
from numba import set_num_threads
from common import OUT,AUDIT,CORES,write_json
import portfolio_inputs as inputs
from portfolio_history import calibration_maxima,multipliers,history_power_bounds,PortfolioHistory
from portfolio_warm_history import WarmPortfolioHistory
from normal_customer import build_history,HISTORICAL_SCORES
from run_experiment_four import untreated_load
from portfolio_sparse_physics import SparsePortfolioAnnual
from refine_portfolio_history_curve import refine_group,WIDTH
from run_core_sensitivity import parameters,load_choice,read,threads_for_worker,DEST as CORE
from sensitivity_parameters import SensitivityParameters

POLICY=SensitivityParameters()
DEST=CORE/'common'


def audit_inputs():
    result=read(DEST/'results.json');key=f'{POLICY.common_variance_fraction:.8f}'
    case=result['cases'][key];c=inputs.PortfolioParameters(**case['parameters'])
    public,_,_,_,_,_,_=build_history(c);models=inputs.profiles(c)
    with np.load(DEST/'portfolio_inputs.npz') as z:
        names=list(z['prototype_names']);baseline=z['call_baselines'].copy();meter=z['call_meters'].copy()
    frozen=pd.read_csv(HISTORICAL_SCORES)['标准正态分位'].to_numpy()
    availability=np.clip((np.minimum(public['left']+.25,np.minimum(public['stops'],public['starts']+c.true_duration_hours))
        -np.maximum(public['left'],public['starts']))/.25,0,1)  # 原15分钟交叠比例。
    members=[f'正态{i+1}' for i in range(max(c.normal_sizes))];weights=np.full(len(members),1/len(members))
    power_bounds=history_power_bounds(c,models,members,weights,public['n'])
    error=[];details=[]
    for name,w in zip(members,weights):
        index=names.index(name);scores=np.empty(public['n']);p=models[name]
        for event in range(public['J']):
            mask=public['call_ids']==event
            innovations=frozen[mask] if index==0 else np.random.default_rng(
                np.random.SeedSequence([c.history_power_seed,index,event])).standard_normal(mask.sum())
            z=innovations.copy()
            for j in range(1,len(z)):z[j]=p['rho']*z[j-1]+np.sqrt(1-p['rho']**2)*innovations[j]
            scores[mask]=z
        potential=c.nomination_mw*inputs.inverse(p,scores,c)
        untreated=untreated_load(c,public,index);delivered=untreated-meter[index]
        lost=np.maximum(potential*availability-delivered,0)
        power_retained=bool(np.all(w*delivered<=power_bounds[len(details)]*availability+c.solver_tolerance))
        details.append(dict(member=name,clipped_points=int(np.sum(potential>untreated)),
            reduced_energy_MWh=float(w*.25*lost.sum()),true_power_path_retained=power_retained))
        error.append(w*(baseline[index]-untreated-c.error_mean_rate*c.load_mw))
    error=np.array([public['W']@e for e in error]);calibration=case['calibration']
    critical=np.where(public['error_families']=='global',calibration['global_']['multiplier'],calibration['other']['multiplier'])
    bound=public['error_stat_SD_MW']*critical
    unit=float(np.max(abs(error)/(weights[:,None]*bound[None,:])))
    aggregate=float(np.max(abs(error.sum(axis=0))/(bound*np.sqrt(POLICY.common_variance_fraction+
        (1-POLICY.common_variance_fraction)*float(weights@weights)))))
    # 原核验允许每户均值偏差；直接检验其可行区间，不能把零偏差当成唯一解释。
    coefficients=np.asarray(public['W'].sum(axis=1)).ravel()
    bias=c.load_mw*c.bias_bound_rate*weights
    lower=np.maximum(-bias,((error-weights[:,None]*bound)/coefficients).max(axis=1))
    upper=np.minimum(bias,((error+weights[:,None]*bound)/coefficients).min(axis=1))
    aggregate_bound=bound*np.sqrt(POLICY.common_variance_fraction+(1-POLICY.common_variance_fraction)*float(weights@weights))
    aggregate_lower=float(((error.sum(axis=0)-aggregate_bound)/coefficients).max())
    aggregate_upper=float(((error.sum(axis=0)+aggregate_bound)/coefficients).min())
    retained=bool(np.all(lower<=upper+c.solver_tolerance) and
        max(lower.sum(),aggregate_lower)<=min(upper.sum(),aggregate_upper)+c.solver_tolerance)
    case['history_physical_audit']=dict(members=details,maximum_zero_bias_individual_error_ratio=unit,
        maximum_zero_bias_aggregate_error_ratio=aggregate,zero_bias_path_retained=bool(max(unit,aggregate)<=1+c.solver_tolerance),
        admissible_member_bias_intervals_MW=np.c_[lower,upper].tolist(),
        admissible_aggregate_bias_interval_MW=[aggregate_lower,aggregate_upper],true_error_path_retained=retained,
        true_physical_history_retained=bool(retained and all(r['true_power_path_retained'] for r in details)),
        changed_history_delivery_points=sum(x['clipped_points'] for x in details),
        reduced_history_energy_MWh=sum(x['reduced_energy_MWh'] for x in details),future_power_paths_unchanged=True)
    write_json(DEST/'results.json',result)
    print('共同误差历史物理核对：',case['history_physical_audit'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=CORES,help='与独立核心补算并行时分配的核心数')
    args=parser.parse_args();threads=min(CORES,args.threads)
    DEST.mkdir(exist_ok=True);threads_for_worker(threads);inputs.CORES=threads
    # 共同误差比较沿用实验三既有功率轨迹及积分精度，两边一致；基准全年结果直接复用。
    base=parameters()
    original=base.common_error_amplitude**2/(base.common_error_amplitude**2+base.individual_error_amplitude**2)
    saved=DEST/'results.json';result=read(saved) if saved.exists() else dict(cases={})
    for gamma in (original,POLICY.common_variance_fraction):
        key=f'{gamma:.8f}'
        if gamma==original:
            previous=read(AUDIT/'portfolio_sequence_capacity.json')['groups']['正态4户']
            history=read(AUDIT/'portfolio_history.json')['groups']['正态4户']
            full=read(AUDIT/'portfolio_capacity.json')['groups']['正态4户']['full_information']
            result['cases'][key]=dict(common_variance_fraction=gamma,parameters=base.values(),history=history['cases'],
                capacities=[r for r in previous['cases'] if r['count'] in (1,20)],full_information=full,
                calibration=history['calibration'],normal_single_history_unchanged=True,
                source='portfolio_sequence_capacity.json/正态4户：同一64条全年功率、32个分位节点的定稿端点')
            write_json(saved,result)
            print('共同误差基准：复用实验三四户既有1次、20次及完整信息容量。',flush=True)
            continue
        cached=result['cases'].get(key,{})
        if len(cached.get('capacities',[]))==2:continue
        # 两个系数仅分配共同/个体方差；总标准差仍由原14.44%误差率校准。
        c=replace(base,common_error_amplitude=float(np.sqrt(gamma)),individual_error_amplitude=float(np.sqrt(1-gamma)))
        inputs.DEST=DEST;inputs.INPUT=DEST/'portfolio_inputs.npz';inputs.AUDIT_FILE=DEST/'input_check.json'
        public,models,data,_=inputs.make_inputs(c)
        group=next(g for g in inputs.definitions(c) if g['name']=='正态4户')
        members=group['members'];weights=np.asarray(group['weights']);indices=[data['names'].index(n) for n in members]
        maxima=calibration_maxima(c,public);critical,calibration=multipliers(c,public,len(members),maxima)
        P=history_power_bounds(c,models,members,weights,public['n'])
        history_path=DEST/f'history_{key}.json'
        history=read(history_path)['cases'] if history_path.exists() else []
        if not history and gamma==original:
            # 基准历史约束与未来积分精度无关，直接复用已完成的四户1次、20次历史核验。
            history=read(AUDIT/'portfolio_history.json')['groups']['正态4户']['cases']
            write_json(history_path,dict(cases=history))
        if gamma!=original:
            for count in (1,20):  # 用户确认的两个资料端点；不重算全部逐次曲线。
                if any(h['count']==count for h in history):continue
                h=WarmPortfolioHistory(c,public,data['baseline'][indices]*weights[:,None],data['meter'][indices]*weights[:,None],
                    weights,P,count,critical,gamma,threads=threads)
                if not h.solve(np.full(len(members),c.true_duration_hours)).success:
                    raise RuntimeError('共同误差情景真实持续能力未被保留')
                thresholds,candidates=h.envelope()
                history.append(dict(count=count,thresholds=thresholds,candidates=candidates.tolist()))
                write_json(history_path,dict(cases=history))
                print(f'共同误差{gamma:.0%}：{count}次历史核验完成。',flush=True)
        choice=load_choice(c);power=np.array([data['power'][n]*w for n,w in zip(members,weights)])
        original_full=read(AUDIT/'portfolio_capacity.json')['groups']['正态4户']['full_information']
        engine=SparsePortfolioAnnual(c,choice,power,original_full['total_upper_MW'])
        reference_case=result['cases'].get(f'{original:.8f}',{})
        if 'full_information' in cached:full=cached['full_information']
        elif gamma!=original and 'full_information' in reference_case:full=reference_case['full_information']
        else:full=engine.capacity(engine.duration_segments(np.full(len(members),c.true_duration_hours)),True,
            lower=original_full['total_lower_MW'],upper=original_full['total_upper_MW'])
        raw=cached.get('capacities',[])
        original_cases={r['count']:r for r in read(AUDIT/'portfolio_sequence_capacity.json')['groups']['正态4户']['cases']}
        for h in history:
            if any(r['count']==h['count'] for r in raw):continue
            candidates=np.asarray(h['candidates']);H=candidates[np.argmin(candidates@weights)]
            if gamma==original:
                H=np.asarray(original_cases[h['count']]['refined_upper_witness_hours'])
            engine.clear_geometry()
            seed=original_cases[h['count']] if gamma==original else None
            upper=engine.capacity(engine.duration_segments(H),False,
                lower=choice.engine.reference+seed['net_lower_MW'] if seed else None,
                upper=choice.engine.reference+seed['net_upper_MW'] if seed else full['total_upper_MW'])
            thresholds={int(k):v for k,v in h['thresholds'].items()}
            segments=engine.envelope_segments(thresholds)
            probe=upper['net_upper_MW']-WIDTH
            value=engine.risk(probe+choice.engine.reference,segments,False)
            if value['upper_MWh']<=c.risk_budget_mwh:lower=dict(net_lower_MW=probe)
            else:lower=engine.capacity(segments,False,upper=upper['total_upper_MW'])
            raw.append(dict(count=h['count'],net_lower_MW=lower['net_lower_MW'],net_upper_MW=upper['net_upper_MW'],
                witness_duration_hours=H.tolist()))
            result['cases'][key]=dict(common_variance_fraction=gamma,parameters=c.values(),history=history,capacities=raw,
                full_information=full,calibration=calibration,normal_single_history_unchanged=True)
            write_json(saved,result)
            print(f'共同方差{gamma:.2%}，{h["count"]}次：{raw[-1]["net_lower_MW"]:.2f}—{raw[-1]["net_upper_MW"]:.2f}兆瓦',flush=True)
        if any(r['net_upper_MW']-r['net_lower_MW']>WIDTH for r in raw):
            refined=refine_group(c,public,models,data,group,dict(cases=raw),dict(cases=history),choice,maxima,
                dict(full_information=full),history_threads=threads)
            raw=refined['cases']
        result['cases'][key]=dict(common_variance_fraction=gamma,parameters=c.values(),history=history,capacities=raw,
            full_information=full,calibration=calibration,normal_single_history_unchanged=True)
        write_json(saved,result)
        print(f'共同方差占比{gamma:.2%}：',[(r['count'],r['net_lower_MW'],r['net_upper_MW']) for r in raw],flush=True)
        del data,engine,power,choice;gc.collect()
    rows=[]
    single={r['count']:r for r in read(AUDIT/'portfolio_sequence_capacity.json')['groups']['正态1户']['cases']}
    for v in result['cases'].values():
        a,b=v['capacities'];o=v['full_information']
        rows.append(dict(共同方差占比=v['common_variance_fraction'],一次认证下界_兆瓦=a['net_lower_MW'],一次认证上界_兆瓦=a['net_upper_MW'],
            二十次认证下界_兆瓦=b['net_lower_MW'],二十次认证上界_兆瓦=b['net_upper_MW'],
            资料收益下界_兆瓦=b['net_lower_MW']-a['net_upper_MW'],资料收益上界_兆瓦=b['net_upper_MW']-a['net_lower_MW'],
            完整信息下界_兆瓦=o['net_lower_MW'],完整信息上界_兆瓦=o['net_upper_MW'],
            一次聚合增益下界_兆瓦=a['net_lower_MW']-single[1]['net_upper_MW'],
            一次聚合增益上界_兆瓦=a['net_upper_MW']-single[1]['net_lower_MW'],
            二十次聚合增益下界_兆瓦=b['net_lower_MW']-single[20]['net_upper_MW'],
            二十次聚合增益上界_兆瓦=b['net_upper_MW']-single[20]['net_lower_MW'],
            未来轨迹数=v['parameters']['future_paths'],分位节点数=v['parameters']['quadrature_nodes']))
    pd.DataFrame(rows).to_csv(DEST/'common_error_comparison.csv',index=False,encoding='utf-8-sig')
    audit_inputs()


if __name__=='__main__':main()
