"""固定20次历史和全年场景，比较真实连续调用能力4至14.5小时。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import dataclass,replace
import time
import numpy as np
import pandas as pd

from common import CORES,OUT,AUDIT,write_json
from historical_duration import HistoricalDuration
from portfolio_history import history_power_bounds
from portfolio_sparse_physics import SparsePortfolioAnnual
from run_experiment_four import future_power,untreated_load
from run_core_sensitivity import (DEST as CORE,NAMES,LABEL,FILE_CUSTOMER,parameters,public_inputs,
    critical_values,error_membership,load_choice,threads_for_worker,checked_capacity,
    read,reference_for,set_risk)


@dataclass(frozen=True)
class DurationParameters:
    baseline_hours:float=14.5  # 用户2026-11-06确定；数值等于现有20个调用日的最长指令时间，是模拟能力基准。
    minimum_hours:float=4.  # 用户确定真实连续调用能力敏感性的起点。
    step_hours:float=.5  # 作者选半小时一档；包含用户指定4至14.5小时范围，共22档。
    history_counts:tuple=(5,20)  # 沿用最近同五次和全部二十次；不增加调用日期。
    future_paths:int=128  # 沿用核心参数补算通过99%代表情景加密后的精度。
    quadrature_nodes:int=64  # 沿用核心参数补算的功率分位积分精度。


POLICY=DurationParameters()
DEST=CORE/'duration'
FIELDS=('total_lower_MW','total_upper_MW','net_lower_MW','net_upper_MW',
        'risk_lower_MWh','risk_upper_MWh','root_evaluations')


def say(text):print(time.strftime('%H:%M:%S'),text,flush=True)


def overlap(public,H):
    return np.clip((np.minimum(public['left']+.25,np.minimum(public['stops'],public['starts']+H))
        -np.maximum(public['left'],public['starts']))/.25,0.,1.)  # 原15分钟读数与指令时段的交叠，换算小时。


def compact(row):return {k:row[k] for k in FIELDS if k in row}


def customer(name,threads):
    threads_for_worker(threads)
    c=replace(parameters(),true_duration_hours=POLICY.baseline_hours,
        future_paths=POLICY.future_paths,quadrature_nodes=POLICY.quadrature_nodes)
    public,maximum,data,models,maxima=public_inputs(c)
    if public['J']!=20:raise RuntimeError('持续能力比较必须保留原20个调用日')
    critical,calibration=critical_values(c,public,maxima)
    index=list(data['prototype_names']).index(name)
    baseline=data['call_baselines'][index].copy();original=data['call_meters'][index].copy()
    untreated=untreated_load(c,public,index)
    full_share=overlap(public,POLICY.baseline_hours)
    if np.any(full_share<=0):raise RuntimeError('基准能力未覆盖既有记录时段')
    potential=(untreated-original)/full_share
    P=float(history_power_bounds(c,models,[name],[1.],public['n'])[0])
    p=dict(public,baseline=baseline,bounds=public['error_stat_SD_MW']*critical)
    original_error=baseline-untreated-c.error_mean_rate*c.load_mw
    checks={count:error_membership(public,original_error,p['bounds'],count,
        c.load_mw*c.bias_bound_rate,c.solver_tolerance) for count in POLICY.history_counts}
    if not all(v['truth_error_retained'] for v in checks.values()):raise RuntimeError('原真实误差被核验范围排除')
    old=read(CORE/f'calc_{FILE_CUSTOMER[name]}.json')
    old_central={r['history_count']:r for r in old['rows'] if r['scenario']=='central'}
    choice=load_choice(c);power=future_power(c,models,name)
    engine=SparsePortfolioAnnual(c,choice,power[None,:,:],old_central[0]['total_upper_MW'])
    # 同一进程固定客户、功率、官方全年场景和调用池，只改变真实能力及其对应的历史电表。
    path=DEST/f'calc_{FILE_CUSTOMER[name]}.json'
    complete_path=DEST/'all_results.json'
    saved=read(path) if path.exists() else (read(complete_path)['customers'].get(name,{}) if complete_path.exists() else {})
    rows=[r for r in saved.get('rows',[]) if all(v['net_upper_MW']-v['net_lower_MW']<=c.capacity_tolerance_mw
        for v in [r['full_information'],r['capacity_5'],r['capacity_20']])]
    full_references={k:v for k,v in saved.get('full_references',{}).items()
        if v['net_upper_MW']-v['net_lower_MW']<=c.capacity_tolerance_mw}
    if saved and saved.get('parameters')!=readable(c.values()):raise RuntimeError('已存持续能力比较的共同参数发生变化')
    cache={}  # 时长按10^-9小时归并，远小于原历史核验10^-6小时可行性复核容差。
    for row in old_central.values():
        if row['history_count'] and row['net_upper_MW']-row['net_lower_MW']<=c.capacity_tolerance_mw:
            cache[round(float(row['duration_hours']),9)]=compact(row)
    for row in rows:
        for count in POLICY.history_counts:cache[round(float(row[f'H{count}']),9)]=row[f'capacity_{count}']
    def save():
        write_json(path,dict(customer=name,parameters=c.values(),grid=POLICY.__dict__,rows=rows,
            full_references=full_references,calibration=calibration,history_points=public['n'],
            history_days=public['J'],history_statistics=public['W'].shape[0],
            true_error_checks=checks,historical_maximum_hours=maximum,
            baseline_meter_max_difference_MW=float(np.max(abs((untreated-potential*full_share)-original)))))
    hours=np.arange(POLICY.minimum_hours,POLICY.baseline_hours+POLICY.step_hours/2,POLICY.step_hours)
    previous_full=None
    for H in hours:
        existing=next((r for r in rows if r['true_hours']==float(H)),None)
        if existing:previous_full=existing['full_information'];continue
        meter=untreated-potential*overlap(public,H)
        history={}
        for count in POLICY.history_counts:
            h=HistoricalDuration(p,meter,count,P,bias_bound=c.load_mw*c.bias_bound_rate,
                solver_tolerance=c.solver_tolerance,duration_check_tolerance=c.duration_check_tolerance_hours)
            if not h.solve(H).success:raise RuntimeError((name,H,count,'真实持续能力没有被历史核验保留'))
            value=h.minimum(maximum,minimum=c.verified_min_duration_hours)
            if value>H+c.duration_check_tolerance_hours:raise RuntimeError((name,H,value,'确认能力超过真值'))
            history[count]=value
        engine.clear_geometry()
        seed=previous_full or dict(total_lower_MW=engine.e.reference_bracket['lower_MW'],
            total_upper_MW=old_central[0]['total_upper_MW'])
        complete=checked_capacity(engine,float(H),True,seed)
        previous_full=complete;capacities={}
        for count,Hverified in history.items():
            key=round(float(Hverified),9)
            if key in cache:capacity=cache[key]
            else:
                engine.clear_geometry()
                capacity=checked_capacity(engine,Hverified,False,dict(
                    total_lower_MW=engine.e.reference_bracket['lower_MW'],total_upper_MW=complete['total_upper_MW']))
                cache[key]=compact(capacity)
            if capacity['net_lower_MW']>complete['net_upper_MW']:
                raise RuntimeError((name,H,count,'历史认证超过完整信息'))
            capacities[count]=capacity
        if any(v['net_upper_MW']-v['net_lower_MW']>c.capacity_tolerance_mw for v in [complete,*capacities.values()]):
            raise RuntimeError((name,H,'容量根尚未达到既定区间精度'))
        if capacities[5]['net_lower_MW']>capacities[20]['net_upper_MW']:
            raise RuntimeError((name,H,'增加历史后的容量下降'))
        row=dict(true_hours=float(H),H5=history[5],H20=history[20],
            capacity_5=capacities[5],capacity_20=capacities[20],full_information=complete,
            changed_meter_points=int(np.sum(abs(meter-original)>c.solver_tolerance)),
            removed_history_delivery_MWh=float(.25*np.sum(original-meter)*-1))
        rows.append(row);save()
        say(f'{LABEL[name]} 真能力{H:g}小时：核验{history[5]:.3f}/{history[20]:.3f}小时；'
            f'容量{capacities[5]["net_lower_MW"]:.2f}/{capacities[20]["net_lower_MW"]:.2f}/'
            f'{complete["net_lower_MW"]:.2f}兆瓦')
    # 基准改变后，更新原核心参数各风险/预算档的完整信息参照；历史核验及其年度根可直接复用。
    for scenario in ('central','tail_0.9','tail_0.99','budget_0.5','budget_2.0'):
        if scenario in full_references:continue
        old_row=next(r for r in old['rows'] if r['scenario']==scenario and r['history_count']==0)
        cc=replace(c,tail_level=old_row['tail_level'],risk_budget_mwh=old_row['risk_budget_MWh'])
        reference=reference_for(engine,cc);set_risk(engine,cc,reference);engine.clear_geometry()
        if scenario=='central':value=next(r['full_information'] for r in rows if r['true_hours']==POLICY.baseline_hours)
        else:value=checked_capacity(engine,POLICY.baseline_hours,True,old_row)
        full_references[scenario]=dict(reference_MW=reference[0],**compact(value));save()
        say(f'{LABEL[name]} 14.5小时完整信息参照 {scenario}：{value["net_lower_MW"]:.2f}兆瓦')
    return name


def readable(value):
    import json
    return json.loads(json.dumps(value))


def export():
    rows=[];customers={}
    complete_path=DEST/'all_results.json'
    previous=read(complete_path).get('customers',{}) if complete_path.exists() else {}
    for name in NAMES:
        path=DEST/f'calc_{FILE_CUSTOMER[name]}.json'
        value=read(path) if path.exists() else previous.get(name)
        if value is None:continue
        customers[name]=value
        for r in value['rows']:
            a,b,o=r['capacity_5'],r['capacity_20'],r['full_information']
            rows.append(dict(客户=LABEL[name],真实持续能力_小时=r['true_hours'],
                五次核验持续能力_小时=r['H5'],二十次核验持续能力_小时=r['H20'],
                五次容量下界_兆瓦=a['net_lower_MW'],五次容量上界_兆瓦=a['net_upper_MW'],
                二十次容量下界_兆瓦=b['net_lower_MW'],二十次容量上界_兆瓦=b['net_upper_MW'],
                完整信息下界_兆瓦=o['net_lower_MW'],完整信息上界_兆瓦=o['net_upper_MW'],
                资料增量下界_兆瓦=max(0.,b['net_lower_MW']-a['net_upper_MW']),
                资料增量上界_兆瓦=b['net_upper_MW']-a['net_lower_MW'],
                尚未确认容量下界_兆瓦=max(0.,o['net_lower_MW']-b['net_upper_MW']),
                尚未确认容量上界_兆瓦=o['net_upper_MW']-b['net_lower_MW'],
                历史电表变化点数=r['changed_meter_points'],历史实际调节减少_兆瓦时=r['removed_history_delivery_MWh']))
    pd.DataFrame(rows).to_csv(DEST/'duration_sensitivity.csv',index=False,encoding='utf-8-sig')
    expected=int(round((POLICY.baseline_hours-POLICY.minimum_hours)/POLICY.step_hours))+1
    finished=len(customers)==len(NAMES) and all(len(v['rows'])==expected and len(v['full_references'])==5 for v in customers.values())
    if finished:
        for name,value in customers.items():
            tolerance=value['parameters']['capacity_tolerance_mw'];budget=value['parameters']['risk_budget_mwh']
            if value['history_days']!=20 or value['baseline_meter_max_difference_MW']>value['parameters']['solver_tolerance']:
                raise RuntimeError((name,'历史数量或14.5小时基准记录发生变化'))
            for row in value['rows']:
                for result in (row['capacity_5'],row['capacity_20'],row['full_information']):
                    if result['net_upper_MW']-result['net_lower_MW']>tolerance:
                        raise RuntimeError((name,row['true_hours'],'容量区间未达到精度'))
                    if not result['risk_lower_MWh']<=budget<result['risk_upper_MWh']:
                        raise RuntimeError((name,row['true_hours'],'风险根没有夹住预算'))
                if max(row['H5'],row['H20'])>row['true_hours']+value['parameters']['duration_check_tolerance_hours']:
                    raise RuntimeError((name,row['true_hours'],'历史能力高于真实能力'))
                if row['capacity_5']['net_lower_MW']>row['capacity_20']['net_upper_MW'] or row['capacity_20']['net_lower_MW']>row['full_information']['net_upper_MW']:
                    raise RuntimeError((name,row['true_hours'],'资料条件下的容量顺序不一致'))
        write_json(complete_path,dict(grid=POLICY.__dict__,customers=customers,completed=True,
            comparison_rows=len(rows),history_days=20,annual_atoms=40300,annual_paths=1788,annual_hours=8760))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--customer',choices=NAMES)
    parser.add_argument('--cores',type=int,default=CORES)  # 按本机22核分配；单户续算可使用其他批次空出的核心。
    args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    names=(args.customer,) if args.customer else NAMES
    complete_path=DEST/'all_results.json'
    if complete_path.exists():
        existing=read(complete_path)
        expected=readable(replace(parameters(),true_duration_hours=POLICY.baseline_hours,
            future_paths=POLICY.future_paths,quadrature_nodes=POLICY.quadrature_nodes).values())
        if existing.get('completed') and all(existing['customers'][name]['parameters']==expected for name in NAMES):
            export();say('复用已完成的四类客户、22档真实持续能力结果。');return
    workers=min(len(names),args.cores)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(customer,name,args.cores//workers+(i<args.cores%workers)) for i,name in enumerate(names)]
        for task in as_completed(pending):
            try:say('完成客户：'+task.result());export()
            except Exception:
                import traceback
                say(traceback.format_exc());raise
    export()


if __name__=='__main__':main()
