"""实验三：同一总申报量下，三户实测组合及正态同类客户组合。"""
from __future__ import annotations
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from numba import set_num_threads
from common import AUDIT,CORES,write_json
from portfolio_inputs import parameters,make_inputs,definitions,DEST,PortfolioParameters,INPUT
from portfolio_history import calibration_maxima,multipliers,history_power_bounds,PortfolioHistory
from portfolio_annual import frozen_choice,PortfolioAnnual


HISTORY=AUDIT/'portfolio_history.json'
RESULT=AUDIT/'portfolio_capacity.json'
COUNTS=(1,20)  # 已有实验二的最近1次和全部20次调用；每户均保留全部15分钟点。


def say(message):print(time.strftime('%H:%M:%S'),message,flush=True)


def history_group(c,public,model,data,group,maxima,threads):
    set_num_threads(threads)
    members=group['members'];weights=np.asarray(group['weights']);M=len(members)
    index=[data['names'].index(name) for name in members]
    critical,calibration=multipliers(c,public,M,maxima)
    P=history_power_bounds(c,model,members,weights,public['n'])
    baseline=data['baseline'][index]*weights[:,None];meter=data['meter'][index]*weights[:,None]
    cases=[]
    for count in COUNTS:
        h=PortfolioHistory(c,public,baseline,meter,weights,P,count,critical,data['gamma'],threads=threads)
        if not h.solve(np.full(M,c.true_duration_hours)).success:
            raise RuntimeError(f'{group["name"]}真实持续能力被历史约束排除')
        thresholds,candidates=h.envelope()
        cases.append(dict(count=count,thresholds=thresholds,candidates=candidates.tolist(),
            longest_historical_call_hours=h.maximum,linear_programs=h.check_count))
        say(f'{group["name"]}，每户{count}次历史：边界候选{len(candidates)}个，阈值{thresholds}')
    # 真值只用于核对核验区间覆盖；不进入ISO历史约束。
    errors=np.asarray([data['actual_error'][i]*w for i,w in zip(index,weights)])
    stdev=public['error_stat_SD_MW'];truth_error=np.array([public['W']@v for v in errors])
    unit_ratio=float(np.max(abs(truth_error)/(weights[:,None]*stdev[None,:]*critical[None,:])))
    agg_ratio=float(np.max(abs(truth_error.sum(axis=0))/(stdev*critical*np.sqrt(data['gamma']+(1-data['gamma'])*float(weights@weights)))))
    return dict(**group,calibration=calibration,power_bounds_MW=P.tolist(),cases=cases,
        realized_error_to_verified_bound_max=unit_ratio,realized_aggregate_error_to_verified_bound_max=agg_ratio)


def historical(c,public,model,data):
    maxima=calibration_maxima(c,public);answer={}
    if HISTORY.exists():
        saved=json.loads(HISTORY.read_text(encoding='utf-8'))
        if saved['parameters']==json.loads(json.dumps(c.values())):answer=saved['groups']
    remaining=[g for g in definitions(c) if g['name'] not in answer]
    if not remaining:return answer
    workers=min(len(remaining),max(1,CORES//2));threads=max(1,CORES//workers)  # 并行组合分用全部22核，避免每个小型规划重复占满线程池。
    compact={k:data[k] for k in ('names','baseline','meter','actual_error','gamma')}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(history_group,c,public,model,compact,g,maxima,threads):g['name'] for g in remaining}
        for task in as_completed(pending):
            answer[pending[task]]=task.result()
            write_json(HISTORY,dict(parameters=c.values(),groups=answer))
    return answer


def annual(c,data,history):
    source=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))['invitation_comparison']
    choice=frozen_choice(c,source);results={}
    if RESULT.exists():
        saved=json.loads(RESULT.read_text(encoding='utf-8'))
        if saved['parameters']==json.loads(json.dumps(c.values())):results=saved['groups']
    def save():
        write_json(RESULT,dict(parameters=c.values(),groups=results,plan_thresholds_MW=[p['threshold_MW'] for p in choice.plans],
            annual_atoms=len(choice.engine.atoms),annual_groups=len(choice.engine.weights),cores=CORES))
    for group in definitions(c):
        name=group['name']
        if name in results and len(results[name].get('history_cases',[]))==len(COUNTS) and 'physical_state_check' in results[name]:continue
        if name not in history:history=json.loads(HISTORY.read_text(encoding='utf-8'))['groups']
        if name not in history:continue
        P=np.asarray([data['power'][user]*w for user,w in zip(group['members'],group['weights'])])
        engine=PortfolioAnnual(c,choice,P);M=len(P);cached=results.get(name,{});cases=cached.get('history_cases',[])
        say(f'全年计算：{name}')
        if name=='正态1户':
            complete=source['capacity_cases'][-1];known=source['capacity_cases'][-2]
        else:
            complete=cached.get('full_information')
            if complete is None:
                seed=source['capacity_cases'][-1]
                complete=engine.capacity(engine.duration_segments(np.full(M,c.true_duration_hours)),True,
                    lower=seed['total_lower_MW'],upper=seed['total_upper_MW'])
                results[name]=dict(**group,full_information=complete,history_cases=cases);save()
            # 完整信息与最坏天气配对分别计算；复用刚得到的容量括区间来开始搜索。
            known=cached.get('known_duration')
            if known is None:
                known=engine.capacity(engine.duration_segments(np.full(M,c.true_duration_hours)),False,
                    lower=complete['total_lower_MW']-c.capacity_tolerance_mw,upper=complete['total_upper_MW'])
                results[name]=dict(**group,full_information=complete,known_duration=known,history_cases=cases);save()
        for case in history[name]['cases']:
            count=case['count'];thresholds={int(k):float(v) for k,v in case['thresholds'].items()}
            if any(row['count']==count for row in cases):continue
            if name=='正态1户' and count==20:
                low=high=source['capacity_cases'][0];witness=case['candidates'][0]
            else:
                candidates=np.asarray(case['candidates'])
                witness=candidates[np.argmin(candidates@np.asarray(group['weights']))]
                high=engine.capacity(engine.duration_segments(witness),False,upper=known['total_upper_MW'])
                segments=engine.envelope_segments(thresholds)
                low=engine.capacity(segments,False,
                    lower=high['total_lower_MW']-c.capacity_tolerance_mw if M==1 else None,upper=high['total_upper_MW'])
            cases.append(dict(count=count,net_lower_MW=low['net_lower_MW'],net_upper_MW=high['net_upper_MW'],
                lower_power_capacity=low,feasible_candidate_capacity=high,witness_duration_hours=list(witness)))
            results[name]=dict(**group,full_information=complete,known_duration=known,history_cases=cases);save()
            say(f'{name}，每户{count}次历史：认证{low["net_lower_MW"]:.2f}—{high["net_upper_MW"]:.2f}兆瓦')
        # κ=1、r=0时，全年累计消耗不超过初始状态即满足所有逐期状态约束。
        state=[]
        for i,(member,w) in enumerate(zip(group['members'],group['weights'])):
            duration_sum=[]
            for plan in choice.plans:
                use=np.minimum(plan['calls'][:,2]-plan['calls'][:,0],c.true_duration_hours)
                total=np.r_[0.,np.cumsum(use)];duration_sum.append(float(np.max(total[plan['ptr'][1:]]-total[plan['ptr'][:-1]])))
            initial=c.state_power_ratio*c.nomination_mw*w*8760.  # 原普通负荷嵌入；8760为年度小时数。
            consumption=max(duration_sum)*float(P[i].max())
            if consumption>initial:raise RuntimeError(f'{name}中{member}的全年状态约束需要显式求解')
            state.append(dict(user=member,initial_state_MWh=initial,annual_consumption_upper_MWh=consumption,
                minimum_state_lower_MWh=initial-consumption))
        results[name]=dict(**group,full_information=complete,known_duration=known,history_cases=cases,physical_state_check=state)
        save()
        report(c,results)
    return results


def report(c,results):
    rows=[]
    for group in definitions(c):
        if group['name'] not in results:continue
        item=results[group['name']]
        if len(item.get('history_cases',[]))<len(COUNTS):continue
        a,b=item['history_cases'];oracle=item['full_information']
        rows.append(dict(组合=group['name'],户数=len(group['members']),每户申报量_兆瓦=c.nomination_mw/len(group['members']),
            一次历史认证下界_兆瓦=a['net_lower_MW'],一次历史认证上界_兆瓦=a['net_upper_MW'],
            二十次历史认证下界_兆瓦=b['net_lower_MW'],二十次历史认证上界_兆瓦=b['net_upper_MW'],
            历史资料增量下界_兆瓦=b['net_lower_MW']-a['net_upper_MW'],历史资料增量上界_兆瓦=b['net_upper_MW']-a['net_lower_MW'],
            完整信息容量下界_兆瓦=oracle['net_lower_MW'],完整信息容量上界_兆瓦=oracle['net_upper_MW'],
            一次历史信息折减下界_兆瓦=max(0.,oracle['net_lower_MW']-a['net_upper_MW']),
            一次历史信息折减上界_兆瓦=oracle['net_upper_MW']-a['net_lower_MW'],
            二十次历史信息折减下界_兆瓦=max(0.,oracle['net_lower_MW']-b['net_upper_MW']),
            二十次历史信息折减上界_兆瓦=oracle['net_upper_MW']-b['net_lower_MW'],
            二十次历史认证占申报百分比下界=b['net_lower_MW']/c.nomination_mw*100,
            二十次历史认证占申报百分比上界=b['net_upper_MW']/c.nomination_mw*100))
    table=pd.DataFrame(rows);table.to_csv(DEST/'portfolio_capacity.csv',index=False,encoding='utf-8-sig')
    def number(a,b):
        return f'{(a+b)/2:.1f}' if b-a<=.5 else f'{a:.1f}—{b:.1f}'  # 小于半兆瓦的已确定范围显示约值；完整上下界保留在数值表。
    readable=[]
    for row in rows:
        readable.append(dict(组合=row['组合'],每户只有1次历史=number(row['一次历史认证下界_兆瓦'],row['一次历史认证上界_兆瓦']),
            每户已有20次历史=number(row['二十次历史认证下界_兆瓦'],row['二十次历史认证上界_兆瓦']),
            增加历史带来的容量=number(row['历史资料增量下界_兆瓦'],row['历史资料增量上界_兆瓦']),
            距完整信息容量=number(row['二十次历史信息折减下界_兆瓦'],row['二十次历史信息折减上界_兆瓦'])))
    display=pd.DataFrame(readable);normal=display['组合'].str.startswith('正态') if len(display) else []
    text='# 实验三：聚合与逐次增加的历史资料\n\n'
    text+=f'所有组合的总申报量固定为{c.nomination_mw:g}兆瓦，总正常负荷固定为{c.load_mw:g}兆瓦。两户各承担一半，三户各承担三分之一。\n\n'
    curve_file=DEST/'portfolio_sequence.csv';curve_audit=AUDIT/'portfolio_sequence_capacity.json'
    if curve_file.exists() and curve_audit.exists():
        curve_saved=json.loads(curve_audit.read_text(encoding='utf-8'));curves=pd.read_csv(curve_file)
        ready=curve_saved['parameters']==json.loads(json.dumps(c.values())) and len(curves)==len(definitions(c))*21
        normal_image=DEST/'normal_portfolio_sequence.png';empirical_image=DEST/'empirical_portfolio_sequence.png'
        if ready and normal_image.exists() and empirical_image.exists():
            text+='## 每增加一次历史记录，容量怎样变化\n\n'
            text+=f'每组总申报量都是{c.nomination_mw:g}兆瓦，组内等分。横轴是每户提交的历史调用记录数，例如横轴5表示每户各交5次记录，4户聚合共提供20份。沿一条曲线向右看，是同一组客户增加资料；同一横轴位置比较不同曲线，是每户记录次数相同时的聚合比较。\n\n'
            text+=f'![正态1—4户的逐次认证及增量](<{normal_image.name}>)\n\n'
            values=[float(curves[(curves['组合']==f'正态{i}户')&(curves['每户历史次数']==5)]['认证约值_兆瓦'].iloc[0]) for i in range(1,5)]
            text+='同样提供5次历史，正态1—4户分别支持约'+ '、'.join(f'{v:.1f}' for v in values)+'兆瓦。\n\n'
            text+=f'![三户拟合的单户及组合逐次结果](<{empirical_image.name}>)\n\n'
            text+='三个两户组合和一个三户组合分别与组成单户比较。每条记录完整保留15分钟读数，按最近1次、最近2次至最近20次依次提供。\n\n'
            text+=f'上图纵轴表示ISO认可容量占总申报量的比例；总申报{c.nomination_mw:g}兆瓦时，20%就是认可{c.nomination_mw*.2:g}兆瓦。下表用兆瓦列出相邻两次提交资料后增加的认证容量，例如第5列对应每户从4次记录增加到5次。数字取整，“≤1”表示新增不超过1兆瓦。阴影和数值区间表示计算上下界。\n\n'
            text+='[逐次容量、增量及上下界](portfolio_sequence.csv)\n\n'
    text+='## 三户实测拟合的组合\n\n三个两户组合，加上一个三户组合；单户结果列在前面，便于对照。容量单位均为兆瓦，表内取约值。\n\n'
    text+=display.loc[~normal].to_markdown(index=False)+'\n\n'
    text+='## 正态同类客户参照\n\n各户来自同一种正态分布，比较户数增加的作用。四户组合也保持总申报量不变，每户占四分之一。\n\n'
    text+=display.loc[normal].to_markdown(index=False)+'\n\n'
    if len(results)==len(definitions(c)):
        text+='## 结果说明\n\n'
        text+='正态四户在各户只有1次历史时，只能认约25兆瓦；单户已有20次历史时，可以认约124兆瓦。增加户数提高了部分认证容量，增加历史资料仍有独立作用。\n\n'
        text+='三户实测组合在各户1次历史时支持22.9—23.5兆瓦，20次历史时支持约274.3兆瓦。三个两户组合在20次历史时也接近274兆瓦。\n\n'
        reference=results['正态1户']['full_information']
        text+=f'客户真实连续调节能力为{c.true_duration_hours:g}小时，正态单户完整信息容量为{number(reference["net_lower_MW"],reference["net_upper_MW"])}兆瓦。各组合完整信息容量及20次资料后尚未确认的容量均按同一能力重新计算，见完整容量表。\n\n'
    text+='## 组合是怎样算的\n\n'
    text+='LSE A、LSE B、LSE C的功率比例分别来自原表实际值/目标值的经验分布，保留各自的相邻时段关系。每户独立生成可调功率；在每个15分钟时点，将各户功率乘各自份额后相加。基线误差保留共同部分和各户独立部分。\n\n'
    text+='每户分别受到功率、持续时间和逐期状态约束；其中一户不能继续时，其余客户按各自能力继续。合计调节量进入完整年度调度，再计算容量。\n\n'
    text+='## ISO两次拿到什么\n\n'
    text+='两边都掌握各户已拟合的功率分布、同样的误差规则和运行约束。第一边提供各户最近1次调用的完整15分钟电表、结算基线和基线输入日记录；第二边增加至各户全部20次既有调用。\n\n'
    text+='ISO资料文件按单户申报1000兆瓦、正常负荷2000兆瓦保存电表与基线，portfolio_weights列出各组合对各户采用的份额。\n\n'
    text+=f'客户真实连续调节能力固定为{c.true_duration_hours:g}小时。两种资料条件使用同一批客户、同一组真实能力、同一套未来调用和2027—2028年度场景。历史曲线共同约束基线误差与客户能力，再把得到的能力范围带入全年计算。\n\n'
    text+='## 全年计算及完整数值\n\n'
    text+=f'全年保留40300个官方场景原子及其1788组精确年度路径，每年8760小时、35040个15分钟点。风险要求沿用最严重{100*(1-c.tail_level):g}%情形的平均新增缺供不超过{c.risk_budget_mwh:g}兆瓦时。容量已扣除相同标准下无新增客户的系统容量。\n\n'
    text+='未来调用采用实验二的追加邀约情景，共用已归档的三套安排；其中正态单户20次历史对应123.8兆瓦，作为这里的单户起点。客户持续能力与ISO邀约分别计算，追加邀约按原开始时刻连续计时。\n\n'
    text+='完整数值表同时给出认证上下界、资料增量、申报量比例及完整信息容量。认证下界根据全部历史约束给出；上界由满足全部历史约束的客户候选给出。\n\n'
    text+='固定组合后增加记录，对应历史核验的容量价值；固定每户记录数量再比较正态1—4户，对应规模与资料改善的比较。完整信息容量减去历史认证容量，就是聚合部分定义的信息折减。三个实测两户组合和一个三户组合列出不同组成下的结果。\n\n'
    text+='[完整容量表](portfolio_capacity.csv) · [组合功率分布](portfolio_power.csv) · [参数及来源](parameters.csv) · [ISO的15分钟资料](portfolio_iso_inputs.npz)\n'
    (DEST/'report.md').write_text(text,encoding='utf-8')
    notes={}
    for filename in ('normal_case_parameters.py','portfolio_inputs.py'):
        for line in Path(__file__).with_name(filename).read_text(encoding='utf-8').splitlines():
            if ' = ' in line and ' # ' in line and ':' in line:
                notes[line.strip().split(':',1)[0]]=line.split(' # ',1)[1]
    pd.DataFrame([dict(参数=k,取值=v,来源=notes[k]) for k,v in c.values().items()]).to_csv(DEST/'parameters.csv',index=False,encoding='utf-8-sig')


def check_results():
    saved=json.loads(RESULT.read_text(encoding='utf-8'));c=PortfolioParameters(**saved['parameters'])
    if set(saved['groups'])!={g['name'] for g in definitions(c)}:raise RuntimeError('组合容量表尚未覆盖全部约定组合')
    for name,g in saved['groups'].items():
        if len(g['history_cases'])!=len(COUNTS):raise RuntimeError((name,'历史条件未完成'))
        if not np.isclose(sum(g['weights']),1.):raise RuntimeError((name,'组合总申报量未保持一致'))
        a,b=g['history_cases']
        if b['net_upper_MW']<a['net_lower_MW']:raise RuntimeError((name,'增加历史后容量下降'))
        if b['net_lower_MW']>g['full_information']['net_upper_MW']:raise RuntimeError((name,'历史认证超过完整信息容量'))
        if min(s['minimum_state_lower_MWh'] for s in g['physical_state_check'])<0:raise RuntimeError((name,'逐户状态约束不满足'))
        for r in [g['full_information'],g['known_duration']]+[value[key] for value in g['history_cases'] for key in ('lower_power_capacity','feasible_candidate_capacity')]:
            low=r.get('risk_lower_MWh',r.get('risk_lower_boundary_MWh'));high=r.get('risk_upper_MWh',r.get('risk_upper_boundary_MWh'))
            if low is not None and not low<=c.risk_budget_mwh<high:raise RuntimeError((name,'年度风险根未夹住预算'))
    kernel=json.loads((AUDIT/'portfolio_kernel_check.json').read_text(encoding='utf-8'))
    if max(kernel['single_annual_loss_max_difference_MWh'],kernel['heterogeneous_duration_direct_integration_max_difference_MWh'])>1e-5:
        raise RuntimeError('组合全年计算与直接积分不一致')  # 沿用核函数独立核对的1e-5兆瓦时精度。
    observed={'prototype_names','portfolio_names','portfolio_weights','call_baselines','call_meters','original_meters',
        'original_days','dates','call_ids','call_left','call_starts','call_stops'}
    with np.load(INPUT) as z:
        if set(z.files)!=observed:raise RuntimeError('ISO资料须为电表、基线、时间和组合份额')
        if not np.allclose(z['portfolio_weights'].sum(axis=1),1.):raise RuntimeError('ISO资料中的组合份额未合计为一')
    say('全部11组、两种历史资料、年度容量边界和逐户物理约束核对完成')


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--history-only',action='store_true');parser.add_argument('--reuse-history',action='store_true')
    parser.add_argument('--config',help='覆盖集中参数入口的JSON文件')
    parser.add_argument('--report',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:check_results();return
    if args.report:
        result=json.loads(RESULT.read_text(encoding='utf-8'));report(PortfolioParameters(**result['parameters']),result['groups']);return
    set_num_threads(CORES);c=parameters(args.config);public,model,data,distribution=make_inputs(c)
    if args.reuse_history:
        saved=json.loads(HISTORY.read_text(encoding='utf-8'))
        if saved['parameters']!=json.loads(json.dumps(c.values())):raise RuntimeError('已存历史核验参数与当前入口不一致')
        history=saved['groups']
    else:history=historical(c,public,model,data)
    if args.history_only:return
    results=annual(c,data,history)
    missing={g['name'] for g in definitions(c)}-set(results)
    if missing:raise RuntimeError(('历史核验尚未齐备的组合',sorted(missing)))
    report(c,results)


if __name__=='__main__':run()
