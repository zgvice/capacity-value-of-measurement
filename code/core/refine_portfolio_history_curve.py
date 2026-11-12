"""逐次容量收紧：核对其他可行客户状态，保留容量上下界及每次增量的数值范围。"""
from __future__ import annotations
import argparse,json,copy,time
import numpy as np
from common import AUDIT,CORES,write_json
from portfolio_inputs import parameters,make_inputs,definitions,PortfolioParameters
from portfolio_history import calibration_maxima,multipliers,history_power_bounds
from portfolio_warm_history import WarmPortfolioHistory
from portfolio_annual import frozen_choice
from portfolio_sparse_physics import SparsePortfolioAnnual
from run_portfolio_history_curve import RESULT,HISTORY,export


REFINED=AUDIT/'portfolio_sequence_refined.json'
WIDTH=.5  # 图以占申报量0.1个百分点显示；把单点容量数值范围收紧至半兆瓦，风险标准保持原值。


def say(message):print(time.strftime('%H:%M:%S'),message,flush=True)


def refine_group(c,public,models,data,group,raw,hist,choice,maxima,anchor,history_threads=None):
    name=group['name'];w=np.asarray(group['weights']);M=len(w);idx=[data['names'].index(n) for n in group['members']]
    critical,_=multipliers(c,public,M,maxima);P=history_power_bounds(c,models,group['members'],w,public['n'])
    power=np.array([data['power'][n]*weight for n,weight in zip(group['members'],w)])
    engine=SparsePortfolioAnnual(c,choice,power,anchor['full_information']['total_upper_MW']);answer=[];previous=None
    hist={r['count']:r for r in hist['cases']}
    for original in sorted(raw['cases'],key=lambda x:x['count']):
        case=copy.deepcopy(original);count=case['count'];lo=case['net_lower_MW'];hi=case['net_upper_MW'];witness=case.get('witness_duration_hours')
        if previous:lo=max(lo,previous['net_lower_MW'])
        if count>0 and hi-lo>WIDTH:
            engine.clear_geometry()
            h=WarmPortfolioHistory(c,public,data['baseline'][idx]*w[:,None],data['meter'][idx]*w[:,None],w,P,
                count,critical,data['gamma'],threads=max(1,CORES//11) if history_threads is None else history_threads)
            # 历史规划在同一进程内保持固定线程数，避免复用求解器时改变全局调度。
            old_witness=previous.get('refined_upper_witness_hours') if previous else None
            retained=False
            if old_witness is not None and h.solve(old_witness).success and previous['net_upper_MW']<hi:
                hi=previous['net_upper_MW'];witness=old_witness;retained=True
                case['refinement_source']='原限制候选通过新增历史的全部约束，复用其年度容量上界'
            if hi-lo>WIDTH and not retained:
                state=hist[count];candidates=np.asarray(state['candidates'])
                if state.get('reused_from_count') is not None:
                    thresholds,candidates=h.envelope()
                else:thresholds={int(k):v for k,v in state['thresholds'].items()}
                _,indices=np.unique(np.round(candidates,4),axis=0,return_index=True)
                candidates=candidates[np.sort(indices)]  # 只省去几乎重合的试探状态；每个保留状态仍单独核对可行性。
                candidates=sorted(candidates,key=lambda H:(float(np.ptp(H)),float(H@w)))
                probe=lo+WIDTH;best=None;best_risk=-np.inf;closed=False
                for H in candidates:
                    if not h.solve(H).success:continue
                    risk=engine.risk(probe+engine.e.reference,engine.duration_segments(H),False)
                    if risk['lower_MWh']>best_risk:best_risk=risk['lower_MWh'];best=H
                    if risk['lower_MWh']>c.risk_budget_mwh:
                        hi=min(hi,probe);witness=H.tolist();closed=True
                        case['refinement_source']='另一完整可行状态在上界处超过原风险预算'
                        case['refined_upper_risk_MWh']=risk['lower_MWh'];break
                if not closed and best is not None:
                    upper=engine.capacity(engine.duration_segments(best),False,lower=lo+engine.e.reference,
                        upper=hi+engine.e.reference)
                    if upper['net_upper_MW']<hi:hi=upper['net_upper_MW'];witness=best.tolist()
                if hi-lo>WIDTH:
                    probe=hi-WIDTH
                    risk=engine.risk(probe+engine.e.reference,engine.envelope_segments(thresholds),False)
                    if risk['upper_MWh']<=c.risk_budget_mwh:
                        lo=max(lo,probe);case['refinement_source']='完整历史下的统一功率下界满足原年度风险预算'
                        case['refined_lower_risk_MWh']=risk['upper_MWh']
            case.update(net_lower_MW=lo,net_upper_MW=hi,refined_upper_witness_hours=witness)
            say(f'范围收紧 {name}，{count}次：{lo:.3f}—{hi:.3f}兆瓦')
        elif witness is not None:case['refined_upper_witness_hours']=witness
        if hi<lo-c.solver_tolerance:raise RuntimeError((name,count,lo,hi))
        case['net_lower_MW']=lo;case['net_upper_MW']=hi;answer.append(case);previous=case
    return dict(**group,cases=answer,refinement_checked=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--normal-only',action='store_true');parser.add_argument('--merge',action='store_true');args=parser.parse_args()
    base=json.loads(RESULT.read_text(encoding='utf-8'));c=PortfolioParameters(**base['parameters'])
    results={k:v for k,v in base['groups'].items() if v.get('refinement_checked')}
    if REFINED.exists():
        saved=json.loads(REFINED.read_text(encoding='utf-8'))
        if saved['parameters']==base['parameters']:results.update(saved['groups'])
    if args.merge:
        for group in results.values():group['refinement_checked']=True
        base['groups'].update(results);base['refinement_target_MW']=WIDTH
        base['maximum_capacity_interval_MW']=max(r['net_upper_MW']-r['net_lower_MW'] for g in base['groups'].values() for r in g['cases'])
        write_json(RESULT,base);export(c,base['groups']);return
    public,models,data,_=make_inputs(c);history=json.loads(HISTORY.read_text(encoding='utf-8'))['groups']
    source=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))['invitation_comparison']
    anchors=json.loads((AUDIT/'portfolio_capacity.json').read_text(encoding='utf-8'))['groups']
    maxima=calibration_maxima(c,public);choice=frozen_choice(c,source)
    for group in definitions(c):
        name=group['name']
        if args.normal_only and group['kind']!='normal':continue
        if name in results:continue
        if name not in base['groups'] or len(base['groups'][name]['cases'])!=public['J']+1:continue
        results[name]=refine_group(c,public,models,data,group,base['groups'][name],history[name],choice,maxima,anchors[name])
        write_json(REFINED,dict(parameters=c.values(),target_width_MW=WIDTH,groups=results))


if __name__=='__main__':main()
