"""逐次提交既有历史：11组客户、每户0—20次完整记录，复用全部年度场景。"""
from __future__ import annotations
import argparse,json,time,copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from numba import set_num_threads
from common import AUDIT,CORES,write_json
from portfolio_inputs import parameters,make_inputs,definitions,DEST,PortfolioParameters
from portfolio_history import calibration_maxima,multipliers,history_power_bounds
from portfolio_warm_history import WarmPortfolioHistory
from portfolio_annual import frozen_choice
from portfolio_sparse_physics import SparsePortfolioAnnual


HISTORY=AUDIT/'portfolio_sequence_history.json'
RESULT=AUDIT/'portfolio_sequence_capacity.json'
ANCHOR_HISTORY=AUDIT/'portfolio_history.json'
ANCHOR_RESULT=AUDIT/'portfolio_capacity.json'
TABLE=DEST/'portfolio_sequence.csv'


def say(message):print(time.strftime('%H:%M:%S'),message,flush=True)


def history_group(c,public,models,data,group,maxima,anchors,normal_curve,threads):
    set_num_threads(threads);members=group['members'];w=np.asarray(group['weights']);M=len(w)
    index=[data['names'].index(name) for name in members];critical,calibration=multipliers(c,public,M,maxima)
    P=history_power_bounds(c,models,members,w,public['n']);cases=[]
    for count in range(1,public['J']+1):
        if count in anchors:
            case=copy.deepcopy(anchors[count]);case['reused_from_count']=None;case['lower_certificate_count']=count
        elif group['name']=='正态1户' and normal_curve is not None:
            H=normal_curve[count]['确认持续时长_小时'];margin=c.duration_check_tolerance_hours
            case=dict(count=count,thresholds={0:max(0.,H-margin),1:0.},candidates=[[H+margin]],
                lower_certificate_count=count,reused_from_count=None,source='实验2已经核验的同一正态历史持续能力')
        else:
            h=WarmPortfolioHistory(c,public,data['baseline'][index]*w[:,None],data['meter'][index]*w[:,None],
                w,P,count,critical,data['gamma'],threads=threads)
            previous=cases[-1];candidates=np.asarray(previous['candidates']);witness=candidates[np.argmin(candidates@w)]
            if h.solve(witness).success:
                case=copy.deepcopy(previous);case.update(count=count,reused_from_count=previous['count'],candidates=[witness.tolist()])
                case['source']='新增记录仍保留原上界候选；历史集合嵌套，沿用原下界证书'
            else:
                thresholds,candidates=h.envelope()
                case=dict(count=count,thresholds=thresholds,candidates=candidates.tolist(),lower_certificate_count=count,
                    reused_from_count=None,linear_programs=h.check_count,source='完整15分钟及跨调用约束重新求解')
        cases.append(case)
        say(f'历史核验 {group["name"]}：{count}/{public["J"]}，'+('复用有效候选' if case['reused_from_count'] else '边界已更新'))
    return dict(**group,calibration=calibration,power_bounds_MW=P.tolist(),cases=cases)


def histories(c,public,models,data):
    old_saved=json.loads(ANCHOR_HISTORY.read_text(encoding='utf-8'))
    if old_saved['parameters']!=json.loads(json.dumps(c.values())):raise RuntimeError('逐次核验须沿用同一参数的单次与20次端点')
    old=old_saved['groups'];answer={}
    if HISTORY.exists():
        saved=json.loads(HISTORY.read_text(encoding='utf-8'))
        if saved['parameters']==json.loads(json.dumps(c.values())):answer=saved['groups']
    normal_source=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))
    matching=all(getattr(c,k)==v for k,v in normal_source['invitation_comparison']['parameters'].items())
    normal={r['历史调用次数']:r for r in normal_source['history_curve']} if matching else None
    maxima=calibration_maxima(c,public)
    pending=[g for g in definitions(c) if g['name'] not in answer]
    if not pending:return answer
    workers=min(len(pending),max(1,CORES//2));threads=max(1,CORES//workers)  # 独立客户组合分用全部核心，每组复用线性规划基。
    compact={key:data[key] for key in ('names','baseline','meter','gamma')}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(history_group,c,public,models,compact,g,maxima,
            {r['count']:r for r in old[g['name']]['cases']},normal,threads):g['name'] for g in pending}
        for future in as_completed(futures):
            name=futures[future];answer[name]=future.result();write_json(HISTORY,dict(parameters=c.values(),groups=answer))
    return answer


def export(c,groups,path=TABLE):
    rows=[]
    for group in definitions(c):
        if group['name'] not in groups:continue
        cases=sorted(groups[group['name']]['cases'],key=lambda x:x['count'])
        # 历史可行集逐次收缩：向前累积下界、向后累积上界，仍是同一真实容量的有效数值界。
        lower=np.maximum.accumulate([r['net_lower_MW'] for r in cases])
        upper=np.minimum.accumulate(np.asarray([r['net_upper_MW'] for r in cases])[::-1])[::-1]
        if np.any(lower>upper+c.solver_tolerance):raise RuntimeError((group['name'],'逐次容量上下界冲突'))
        for i,(case,lo,hi) in enumerate(zip(cases,lower,upper)):
            prev_lo=lower[i-1] if i else lo;prev_hi=upper[i-1] if i else hi
            rows.append(dict(组合=group['name'],户数=len(group['members']),每户历史次数=case['count'],
                认证下界_兆瓦=lo,认证上界_兆瓦=hi,认证约值_兆瓦=(lo+hi)/2,
                占总申报量下界_百分比=100*lo/c.nomination_mw,占总申报量上界_百分比=100*hi/c.nomination_mw,
                占总申报量约值_百分比=50*(lo+hi)/c.nomination_mw,
                本次新增下界_兆瓦=max(0.,lo-prev_hi),本次新增上界_兆瓦=max(0.,hi-prev_lo),
                本次新增约值_兆瓦=(lo+hi-prev_lo-prev_hi)/2,
                本次新增约值_百分点=50*(lo+hi-prev_lo-prev_hi)/c.nomination_mw,
                计算依据=case.get('source','全年风险根'),复用自次数=case.get('reused_from_count')))
    table=pd.DataFrame(rows)
    if path is not None:table.to_csv(path,index=False,encoding='utf-8-sig')
    return table


def annual(c,data,history):
    anchored=json.loads(ANCHOR_RESULT.read_text(encoding='utf-8'))['groups']
    source=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))['invitation_comparison']
    choice=frozen_choice(c,source);results={}
    if RESULT.exists():
        saved=json.loads(RESULT.read_text(encoding='utf-8'))
        if saved['parameters']==json.loads(json.dumps(c.values())):results=saved['groups']
    def save():
        write_json(RESULT,dict(parameters=c.values(),groups=results,annual_atoms=len(choice.engine.atoms),
            annual_groups=len(choice.engine.weights),record_order='从最近一次起，逐次补充更早的既有记录；与原1次、20次端点一致'))
        export(c,results)
    for group in definitions(c):
        name=group['name'];done=results.get(name,dict(**group,cases=[]));cases={r['count']:r for r in done['cases']}
        if len(cases)==len(history[name]['cases'])+1:continue
        P=np.asarray([data['power'][user]*w for user,w in zip(group['members'],group['weights'])])
        upper=anchored[name]['full_information']['total_upper_MW'];engine=SparsePortfolioAnnual(c,choice,P,upper)
        if 0 not in cases:
            if c.verified_min_duration_hours==0:
                zero=dict(net_lower_MW=0.,net_upper_MW=0.,total_lower_MW=engine.e.reference_bracket['lower_MW'],total_upper_MW=engine.e.reference_bracket['upper_MW'])
            else:zero=engine.capacity(engine.duration_segments(np.full(len(P),c.verified_min_duration_hours)),False)
            cases[0]=dict(count=0,source='未提供历史时，沿用已核定工程持续能力下限',**zero)
        anchor_cases={r['count']:r for r in anchored[name]['history_cases']}
        for hist in history[name]['cases']:
            count=hist['count']
            if count in cases:continue
            if name=='正态1户' and count>1 and count not in anchor_cases:
                earlier=history[name]['cases'][count-2]
                if hist['thresholds']==earlier['thresholds'] and hist['candidates']==earlier['candidates']:
                    hist=dict(hist,reused_from_count=count-1)  # 单户标量持续能力完全相同时，未来物理输入及认证结果相同。
            if count in anchor_cases:
                case=copy.deepcopy(anchor_cases[count]);case['source']='复用已定稿的1次或20次年度容量端点'
            elif hist['reused_from_count'] is not None:
                case=copy.deepcopy(cases[hist['reused_from_count']]);case.update(count=count,reused_from_count=hist['reused_from_count'],
                    source='新增记录仍保留原限制候选，复用已夹住的年度容量范围')
            else:
                engine.clear_geometry();previous=cases[count-1];future=anchor_cases[max(anchor_cases)]
                candidate=np.asarray(hist['candidates']);H=candidate[np.argmin(candidate@np.asarray(group['weights']))]
                segments=engine.duration_segments(H);at_upper=engine.risk(future['net_upper_MW']+engine.e.reference,segments,False)
                if at_upper['lower_MWh']>c.risk_budget_mwh:
                    high=engine.capacity(segments,False,lower=previous['net_lower_MW']+engine.e.reference,
                        upper=future['net_upper_MW']+engine.e.reference)
                    cap_upper=min(high['net_upper_MW'],future['net_upper_MW']);upper_source='完整历史下可行候选的年度风险根'
                else:
                    high=None;cap_upper=future['net_upper_MW'];upper_source='全部20次历史的既有容量上界；可行集嵌套'
                thresholds={int(k):v for k,v in hist['thresholds'].items()}
                low=engine.capacity(engine.envelope_segments(thresholds),False,
                    lower=previous['net_lower_MW']+engine.e.reference,upper=cap_upper+engine.e.reference)
                cap_lower=max(low['net_lower_MW'],previous['net_lower_MW'])
                if cap_lower>cap_upper+c.solver_tolerance:raise RuntimeError((name,count,cap_lower,cap_upper))
                case=dict(count=count,net_lower_MW=cap_lower,net_upper_MW=cap_upper,lower_power_capacity=low,
                    feasible_candidate_capacity=high,upper_source=upper_source,witness_duration_hours=H.tolist(),source='同一全年场景逐次核验求根')
            cases[count]=case;done['cases']=sorted(cases.values(),key=lambda x:x['count']);results[name]=done;save()
            say(f'逐次容量 {name}：{count}次，{case["net_lower_MW"]:.2f}—{case["net_upper_MW"]:.2f}兆瓦')
        engine.clear_geometry()
    save();return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--history-only',action='store_true');parser.add_argument('--reuse-history',action='store_true');args=parser.parse_args()
    set_num_threads(CORES)
    c=PortfolioParameters(**json.loads(ANCHOR_RESULT.read_text(encoding='utf-8'))['parameters'])  # 逐次计算跟随当前端点参数，敏感性分析复用同一入口。
    public,models,data,_=make_inputs(c)
    history=json.loads(HISTORY.read_text(encoding='utf-8'))['groups'] if args.reuse_history else histories(c,public,models,data)
    if args.history_only:return
    annual(c,data,history)


if __name__=='__main__':main()
