"""实验一余量敏感性：每档固定同一调用，比较到点停止与续调四小时。"""
from functools import lru_cache
import argparse
import numpy as np
import pandas as pd
from numba import set_num_threads
from common import DATA, OUT, AUDIT, CORES, cvar, monotone_root, read_json, write_json
from experiment_one_parameters import ExperimentOneParameters
from experiment_one_physics import losses, clock_account
from call_schedule_physics import call_formats, onset_hours, tail_weights
from system_reference import _sort_zero_background, _zero_loss
from sensitivity_parameters import SensitivityParameters
import run_experiment_one as original

DEST=OUT/'exp1'
POLICY=SensitivityParameters()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=CORES,help='与其他独立补算并行时分配的核心数')
    args=parser.parse_args();set_num_threads(min(CORES,args.threads))
    old=read_json(AUDIT/'exp1_check.json');c=ExperimentOneParameters(**old['parameters'])
    with np.load(DATA/'annual/exact_path_groups.npz') as z:
        profiles,weights,ptr,hours,values=(z[k] for k in ('profile_ids','weights','ptr','hours','values'))
    with np.load(DATA/'annual/background_arrays.npz') as z:g=z['g_committed_median_allnumeric']
    original.HOURS=g.shape[1]
    lengths=np.array([x[2] for x in call_formats()]);probability=np.repeat(weights/len(lengths),len(lengths))
    extensions=np.asarray([c.extension_hours[0],c.extension_hours[-1]])  # 用户要求仅比较定稿0小时及4小时。
    saved=DEST/'margin_sensitivity.json';result=read_json(saved) if saved.exists() else dict(cases={})
    all_clocks=[]
    for scale in POLICY.surplus_scales_mw:
        key=f'{scale:g}'
        clock_path=DEST/f'plot/margin_{key}_common_tail.csv'
        if key in result['cases']:
            if clock_path.exists():all_clocks.append(pd.read_csv(clock_path))
            continue
        if scale==c.system_surplus_mw:
            a,b=old['capacity_rows'][0],old['capacity_rows'][-1]
            rows=[dict(extension_hours=r['最多续调_小时'],net_lower_MW=r['下界_兆瓦']-old['no_resource_reference_MW'],
                net_upper_MW=r['上界_兆瓦']-old['no_resource_reference_MW'],total_lower_MW=r['下界_兆瓦'],
                total_upper_MW=r['上界_兆瓦'],risk_lower_MWh=r['下界风险_兆瓦时'],risk_upper_MWh=r['上界风险_兆瓦时']) for r in (a,b)]
            reference=old['no_resource_reference_MW'];share=old['diagnostic_interval_share'];saving=old['common_tail_saving_MWh']
            clock=pd.read_csv(DEST/'plot/shortfall_timing.csv')
            clock['余量尺度_兆瓦']=scale;clock['共同增载_兆瓦']=old['common_load_MW']
        else:
            gap=-scale*g[profiles];s0=np.zeros(len(weights))
            for j in range(len(weights)):
                sl=slice(ptr[j],ptr[j+1]);gap[j,hours[sl]]=values[sl];s0[j]=values[sl].sum()
            ordered,prefix=_sort_zero_background(g,scale)
            @lru_cache(maxsize=128)
            def baseline(q):return _zero_loss(q,ordered,prefix,g,profiles,ptr,hours,values,s0,scale)
            reference,refbr=monotone_root(lambda q:cvar(baseline(q)[0],weights,c.risk_level),
                c.risk_budget_mwh,tol=c.reference_tolerance_mw)
            starts,counts=onset_hours(gap,reference)
            @lru_cache(maxsize=128)
            def annual(q):
                return losses(q,reference,gap,starts,counts,lengths,baseline(q)[1],s0,
                    extensions,c.power_mw,c.invitation_extension_hours)
            rows=[]
            for i,H in enumerate(extensions):
                root,bracket=monotone_root(lambda q:cvar(annual(q)[i],probability,c.risk_level),c.risk_budget_mwh,
                    hi=refbr['upper_MW']+c.power_mw,tol=POLICY.capacity_tolerance_mw)
                lo,hi=bracket['lower_MW'],bracket['upper_MW']
                low=cvar(annual(lo)[i],probability,c.risk_level);high=cvar(annual(hi)[i],probability,c.risk_level)
                if not low<=c.risk_budget_mwh<high:raise RuntimeError('实验一余量敏感性风险根未夹住预算')
                rows.append(dict(extension_hours=H,net_lower_MW=lo-reference,net_upper_MW=hi-reference,
                    total_lower_MW=lo,total_upper_MW=hi,risk_lower_MWh=low,risk_upper_MWh=high))
            q=(rows[-1]['total_lower_MW']+rows[-1]['total_upper_MW'])/2
            tail=tail_weights(annual(q)[0],probability,c.risk_level)
            clock=original.hourly_table(clock_account(q,reference,gap,starts,counts,lengths,tail,
                profiles//original.ROTATIONS,extensions[-1],c.power_mw,c.invitation_extension_hours)).reset_index()
            saving=float(clock['续调少缺供_兆瓦时'].sum())
            hour=clock['模型月日时刻'].str[6:8].astype(int)
            evening=float(clock.loc[(hour>=c.diagnostic_start_hour)&(hour<c.diagnostic_end_hour),'续调少缺供_兆瓦时'].sum())
            share=evening/saving if saving else 0.
            clock['余量尺度_兆瓦']=scale;clock['共同增载_兆瓦']=q
            del gap
        clock.to_csv(clock_path,index=False,encoding='utf-8-sig');all_clocks.append(clock)
        result['cases'][key]=dict(scale_MW=scale,reference_MW=reference,capacities=rows,
            gain_lower_MW=rows[-1]['net_lower_MW']-rows[0]['net_upper_MW'],
            gain_upper_MW=rows[-1]['net_upper_MW']-rows[0]['net_lower_MW'],
            common_tail_saving_MWh=saving,evening_share=share)
        write_json(saved,result)
        print(f'实验一余量{scale:g}MW：续调容量增量{result["cases"][key]["gain_lower_MW"]:.2f}—'
            f'{result["cases"][key]["gain_upper_MW"]:.2f}MW，晚间少缺供占比{share:.2%}',flush=True)
    table=[]
    for x in result['cases'].values():
        a,b=x['capacities']
        table.append(dict(余量尺度_兆瓦=x['scale_MW'],无客户基准_兆瓦=x['reference_MW'],
            停止容量下界_兆瓦=a['net_lower_MW'],停止容量上界_兆瓦=a['net_upper_MW'],
            续调四小时下界_兆瓦=b['net_lower_MW'],续调四小时上界_兆瓦=b['net_upper_MW'],
            容量增量下界_兆瓦=x['gain_lower_MW'],容量增量上界_兆瓦=x['gain_upper_MW'],
            晚间少缺供占比=x['evening_share']))
    pd.DataFrame(table).sort_values('余量尺度_兆瓦').to_csv(DEST/'margin_sensitivity.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_clocks,ignore_index=True).to_csv(DEST/'plot/margin_shortfall_timing.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':main()
