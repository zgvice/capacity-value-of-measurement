"""整理四个实验的展示数据，补齐过程；沿用当前参数、记录及年度结果。"""
from __future__ import annotations
import argparse,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import truncnorm
from numba import set_num_threads
from common import ROOT,DATA,OUT,AUDIT,CORES,write_json

QUARTER=.25  # 当前计量网格：15分钟折算为小时。
SUMMARY=AUDIT/'plot_metadata.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(folder,name,rows):
    folder=folder/'plot';folder.mkdir(exist_ok=True)
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(folder/name,index=False,encoding='utf-8-sig')
    return frame


def record(stage,value):
    all_results=read(SUMMARY) if SUMMARY.exists() else {}
    all_results[stage]=value;write_json(SUMMARY,all_results)
    print(stage,json.dumps(value,ensure_ascii=False),flush=True)






def data():
    from normal_case_parameters import NormalCaseParameters
    c=NormalCaseParameters();dest=OUT/'hubei'
    raw=pd.read_csv(dest/'meter_series.csv',dtype={'event':str})
    fit=pd.read_csv(dest/'fitted_profiles.csv');names=list(fit['用户']);points=[];description=[]
    for row in fit.to_dict('records'):
        name=row['用户'];part=raw[raw.user==name].copy();ratio=part.actual_kW/part.target_kW
        points.extend(dict(客户=name,场次=r.event,记录时刻=r.time,功率占申报百分比=100*v,数据='原表比例映射')
            for r,v in zip(part.itertuples(),ratio))
        description.append(dict(客户=name,类型={'LSE A':'钢铁','LSE B':'水泥','LSE C':'建材'}[name],
            调用场次=row['调用场次'],记录数=row['记录数'],均值百分比=100*row['平均功率比例'],
            标准差百分比=100*row['比例标准差'],相邻15分钟关联=row['相邻15分钟关联'],
            最小百分比=100*ratio.min(),最大百分比=100*ratio.max(),仿真申报量_兆瓦=c.nomination_mw))
    # 201个等概率分位为绘图采样分辨率；参照分布仍为既定非负正态。
    probability=(np.arange(201)+.5)/201
    normal=truncnorm.ppf(probability,-c.response_mean_ratio/c.response_sd_ratio,np.inf,
        loc=c.response_mean_ratio,scale=c.response_sd_ratio)
    points.extend(dict(客户='正态参照',场次='',记录时刻='',功率占申报百分比=100*v,数据='等概率分位点，非实测样本') for v in normal)
    description.append(dict(客户='正态参照',类型='模拟参照',调用场次=0,记录数=0,
        均值百分比=100*c.response_mean_ratio,标准差百分比=100*c.response_sd_ratio,
        相邻15分钟关联=c.response_rho15,最小百分比=np.nan,最大百分比=np.nan,仿真申报量_兆瓦=c.nomination_mw))
    save(dest,'customer_profiles.csv',points);save(dest,'customer_notes.csv',description)
    record('数据处理',dict(empirical_users=names,empirical_points=sum(r['记录数'] for r in description),
        normal_plot_quantiles=len(normal),source='meter_series.csv、fitted_profiles.csv及正态参数'))


def experiment_one():
    from experiment_one_parameters import ExperimentOneParameters
    from system_reference import _sort_zero_background,_zero_loss
    from call_schedule_physics import call_formats,onset_hours,call_ends,tail_weights
    from experiment_one_physics import losses,interval_saving
    dest=OUT/'exp1';old=read(AUDIT/'exp1_check.json');c=ExperimentOneParameters(**old['parameters'])
    cap=pd.read_csv(dest/'capacity.csv');cap['容量占交付功率_百分比']=100*cap['净增加_兆瓦']/c.power_mw
    save(dest,'capacity_comparison.csv',cap)
    hourly=pd.read_csv(dest/'hourly_shortfall.csv');hourly=hourly[hourly['核对口径']=='共同增载和共同尾部'].copy()
    hourly['月日']=hourly['模型月日时刻'].str[:5];hourly['小时']=hourly['模型月日时刻'].str[6:8].astype(int)
    hourly['少缺供占比_百分比']=100*hourly['续调少缺供_兆瓦时']/old['common_tail_saving_MWh']
    save(dest,'shortfall_timing.csv',hourly)
    # 代表过程取共同尾部中对续调收益贡献最大的同一年度路径和同一调用格式。
    with np.load(DATA/'annual/exact_path_groups.npz') as z:
        profiles,weights,ptr,hours,values=(z[k] for k in ['profile_ids','weights','ptr','hours','values'])
    with np.load(DATA/'annual/background_arrays.npz') as z:g=z['g_committed_median_allnumeric']
    gap=-c.system_surplus_mw*g[profiles];original=np.zeros(len(weights))
    for j in range(len(weights)):
        sl=slice(ptr[j],ptr[j+1]);gap[j,hours[sl]]=values[sl];original[j]=values[sl].sum()
    formats=call_formats();lengths=np.array([x[2] for x in formats]);q=old['common_load_MW'];reference=old['no_resource_reference_MW']
    ordered,prefix=_sort_zero_background(g,c.system_surplus_mw)
    total=_zero_loss(q,ordered,prefix,g,profiles,ptr,hours,values,original,c.system_surplus_mw)[1]
    starts,counts=onset_hours(gap,reference)
    annual=losses(q,reference,gap,starts,counts,lengths,total,original,np.asarray(c.extension_hours),c.power_mw,c.invitation_extension_hours)
    tail=tail_weights(annual[0],np.repeat(weights/len(lengths),len(lengths)),c.risk_level)
    benefit=tail*(annual[0]-annual[-1]);difference=abs(float(benefit.sum())-old['common_tail_saving_MWh'])
    if difference>c.accounting_tolerance_mwh:raise RuntimeError(('共同尾部缺供分账不符',difference))
    atom=int(np.argmax(benefit));j,variant=divmod(atom,len(lengths))
    calls=call_ends(gap[j],starts[j],counts[j],lengths,variant,reference,c.invitation_extension_hours)
    benefit_by_call=[interval_saving(gap[j],q,b,min(e,b+c.extension_hours[-1]),c.power_mw) for a,b,e in calls]
    call_id=int(np.argmax(benefit_by_call));a,b,e=calls[call_id]
    # 前后一小时仅为展示窗口；每个时点使用原小时缺口和15分钟交叠积分。
    left=np.arange(max(0.,np.floor(a)-1),min(len(gap[j]),np.ceil(e)+1),QUARTER)
    trace=[]
    for extension in c.extension_hours:
        finish=min(e,b+extension);sum_energy=0.
        for t in left:
            before=max(gap[j,int(t)]+q,0.)
            active=max(0.,min(t+QUARTER,finish)-max(t,a))/QUARTER
            response=min(before,c.power_mw)*active;sum_energy+=response*QUARTER
            trace.append(dict(最多续调_小时=extension,年度路径组=j,调用格式编号=variant,调用序号=call_id,
                年度时刻_小时=t,调用开始后_小时=t-a,系统缺口_兆瓦=before,
                最优调节_兆瓦=response,剩余缺供_兆瓦=before-response,
                原邀约开始_小时=a,原邀约结束_小时=b,延续邀约结束_小时=e,
                在原结束前=t<b,共同增载_兆瓦=q))
        expected=interval_saving(gap[j],q,a,finish,c.power_mw)
        if abs(sum_energy-expected)>c.accounting_tolerance_mwh:raise RuntimeError('代表过程积分与原年度核函数不符')
    save(dest,'call_trace.csv',trace)
    record('实验一',dict(reference_MW=reference,capacities_MW=cap['净增加_兆瓦'].tolist(),
        tail_saving_MWh=float(benefit.sum()),evening_share=old['diagnostic_interval_share'],
        selected_annual_group=j,selected_template=variant,selected_call_start=float(a),
        selected_original_end=float(b),selected_invitation_end=float(e),tail_accounting_error_MWh=difference))


def normal_history():
    from normal_case_parameters import NormalCaseParameters
    from normal_customer import build_history
    old=read(AUDIT/'exp2_check.json');c=NormalCaseParameters(**old['parameters'])
    public,meter,power,maximum,truth,records,source=build_history(c)
    critical=old['history_calibration']['calibration']
    public['bounds']=public['error_stat_SD_MW']*np.where(public['error_families']=='global',
        critical['global']['joint_multiplier'],critical['other']['joint_multiplier'])
    return old,c,public,meter,power,maximum


def _duration(c,p,y,P,count):
    from historical_duration import HistoricalDuration
    return HistoricalDuration(p,y,count,P,bias_bound=c.load_mw*c.bias_bound_rate,
        solver_tolerance=c.solver_tolerance,duration_check_tolerance=c.duration_check_tolerance_hours)


def _labels(p):
    ids=p['call_ids'];n=p['n'];J=p['J'];hour=np.floor(np.maximum(p['left']-p['starts'],0)).astype(int)
    dates=[str(d['task']['date']) for d in p['designs']]
    rows=[f'{dates[int(ids[i])]}，开始后{p["left"][i]-p["starts"][i]:g}小时的15分钟读数' for i in range(n)]
    for j in range(J):
        rows.append(f'{dates[j]}调用全程')
        rows.extend(f'{dates[j]}，开始后{h}至{h+1}小时' for h in np.unique(hour[ids==j]))
    for count in range(1,J+1):
        known=ids>=J-count;rows.append(f'最近{count}次调用全程合用')
        rows.extend(f'最近{count}次调用，开始后{h}至{h+1}小时合用' for h in np.unique(hour[known]))
    if len(rows)!=p['W'].shape[0]:raise RuntimeError('过程说明与原统计行数不符')
    return np.asarray(rows)


def historical_call_sources(dest):
    """把每个历史调用日的日期出处和实际采用的指令放在同一张来源表。"""
    from normal_case_parameters import NormalCaseParameters
    task_file=DATA/'frozen/call_tasks.csv'
    tasks=pd.read_csv(task_file,keep_default_na=False)
    selected=tasks.sort_values(['duration_hours','task_id'],ascending=[False,True]).drop_duplicates('date').sort_values('date')
    raw=pd.read_csv(DATA/'sources/public/dispatch_records.csv',keep_default_na=False)
    tolerance=NormalCaseParameters().time_tolerance_hours  # 沿用历史记录生成时的15分钟边界容差。
    rows=[]
    for i,task in enumerate(selected.to_dict('records')):
        source=raw[raw['date']==task['date']]
        book=source[source.source_file.str.endswith('.xlsx')]
        if len(book):
            first=book.loc[pd.to_numeric(book.source_row).idxmin()]
            kind='工作簿';position=f'{first["source_sheet"]}!D{int(first["source_row"])}'
        else:
            first=source.iloc[0];kind='10月调用报告'
            pages=sorted(set(pd.to_numeric(source.PDF_page_one_based).astype(int)))
            position='第'+'、'.join(str(page) for page in pages)+'页'
        points=int(np.ceil(task['end_hour']/QUARTER-tolerance)-np.floor(task['start_hour']/QUARTER+tolerance))
        rows.append(dict(调用日期=task['date'],年份=int(task['date'][:4]),日期来源=kind,
            日期来源原件=first['archived_source_file'],原件位置=position,
            当日区域及产品任务数=int((tasks.date==task['date']).sum()),
            所选任务编号=task['task_id'],调用开始_小时=task['start_hour'],调用结束_小时=task['end_hour'],
            调用持续时间_小时=task['duration_hours'],十五分钟读数点数=points,
            ISO提交顺序=len(selected)-i,调用时刻来源=task['archived_source_file'],
            所选区域=task['zone_literal'],所选产品=task['product'],提前通知类型=task['lead_time']))
    frame=save(dest,'call_dates_sources.csv',rows)
    if len(frame)!=len(selected) or int(frame['当日区域及产品任务数'].sum())!=len(tasks):
        raise RuntimeError('调用日期来源表未覆盖全部选用日期与任务')
    return frame


def experiment_two():
    dest=OUT/'exp2';old,c,p,y,P,maximum=normal_history();J=p['J']
    historical_call_sources(dest)
    curve=pd.read_csv(OUT/'exp3/portfolio_sequence.csv');curve=curve[curve['组合']=='正态1户'].sort_values('每户历史次数')
    other=read(AUDIT/'portfolio_sequence_capacity.json')['parameters']
    if any(other[k]!=v for k,v in old['invitation_comparison']['parameters'].items()):raise RuntimeError('延续邀约曲线参数不一致')
    process=[];limits=[];grid=[];labels=_labels(p);previous=c.verified_min_duration_hours
    # 网格跟随原15分钟刻度，增加精确能力边界；不重新逐格求解同一单调可行性问题。
    durations=[r['确认持续时长_小时'] for r in old['history_curve']]
    display_H=np.unique(np.r_[np.arange(0,c.true_duration_hours+QUARTER,QUARTER),durations])
    for count,H in enumerate(durations,1):
        model=_duration(c,p,y,P,count);fit_old=model.solve(previous);fit_current=model.solve(H+c.duration_check_tolerance_hours)
        if not fit_current.success:raise RuntimeError(('已存历史能力边界不符',count,H))
        selected=p['required']<=count;stat_labels=labels[selected];families=p['error_families'][selected]
        violation=0.;strong=''
        if not fit_old.success:
            # δ只诊断旧能力解释距原可行集有多远，单位MW；δ=0等价于原历史约束可行。
            A=sparse.hstack([model.A,-np.ones((model.A.shape[0],1))],format='csr')
            bounds=list(zip(np.zeros(model.n),model.cap(previous)))+[(-model.bias_bound,model.bias_bound),(0,None)]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options')
                solution=linprog(np.r_[np.zeros(model.n+1),1.],A_ub=A,b_ub=model.b,bounds=bounds,method='highs',
                    options={'threads':CORES,'primal_feasibility_tolerance':c.solver_tolerance,'dual_feasibility_tolerance':c.solver_tolerance})
            if not solution.success:raise RuntimeError(solution.message)
            violation=float(solution.x[-1]);importance=abs(solution.ineqlin.marginals)
            important=np.flatnonzero(importance>c.solver_tolerance)
            # 仅输出有正对偶权重的实际限制，不按展示需要另选候选或修改参数。
            for at in important:
                row=int(at%len(stat_labels));full_row=np.flatnonzero(selected)[row]
                events=np.unique(p['call_ids'][p['W'][full_row].indices])
                dates='；'.join(str(p['designs'][int(event)]['task']['date']) for event in events)
                limits.append(dict(历史次数=count,原确认时长_小时=previous,
                    核验内容=stat_labels[row],统计类别={'interval':'逐点','event':'一次调用内','global':'跨调用统计'}[families[row]],
                    实际涉及调用次数=len(events),实际涉及日期=dates,对偶权重=float(importance[at]),
                    最小约束偏差_兆瓦=violation))
            strong=stat_labels[int(np.argmax(importance)%len(stat_labels))]
        design=p['designs'][J-count];entry=curve[curve['每户历史次数']==count].iloc[0]
        prev=curve[curve['每户历史次数']==count-1].iloc[0]
        process.append(dict(历史次数=count,本次增加记录日期=design['task']['date'],本次邀约时长_小时=design['task']['duration_hours'],
            已有15分钟点数=int((p['call_ids']>=J-count).sum()),原确认时长_小时=previous,确认时长_小时=H,
            新增确认时长_小时=H-previous,原能力解释仍可行=bool(fit_old.success),最小约束偏差_兆瓦=violation,
            主要限制=strong,认证下界_兆瓦=entry['认证下界_兆瓦'],认证上界_兆瓦=entry['认证上界_兆瓦'],
            占申报量_百分比=entry['占总申报量约值_百分比'],新增约值_兆瓦=entry['认证约值_兆瓦']-prev['认证约值_兆瓦']))
        grid.extend(dict(历史次数=count,候选持续能力_小时=h,能解释全部记录=bool(h+c.time_tolerance_hours>=H)) for h in display_H)
        previous=H
    save(dest,'evidence_capacity.csv',process);save(dest,'excluded_constraints.csv',limits);save(dest,'duration_by_count.csv',grid)
    families=p['error_families'];comparisons=[]
    masks=[('15分钟逐点核验',families=='interval'),('逐点和每次调用核验',families!='global'),
           ('跨调用统计核验',families=='global'),('全部联合核验',np.ones(len(families),bool))]
    for label,mask in masks:
        part=dict(p,W=p['W'][mask],bounds=p['bounds'][mask],required=p['required'][mask])
        H=_duration(c,part,y,P,J).minimum(maximum,c.verified_min_duration_hours)
        comparisons.append(dict(核验方式=label,核验统计数=int(mask.sum()),确认持续能力_小时=H,
            共同覆盖水平=c.verification_coverage,记录次数=J,完整15分钟点数=p['n']))
    save(dest,'verification_effects.csv',comparisons)
    # 两种确认时长只补一个尚缺的年度容量根，其余直接复用延续邀约结果。
    from portfolio_inputs import PortfolioParameters
    from portfolio_annual import frozen_choice
    from portfolio_sparse_physics import SparsePortfolioAnnual
    from normal_customer import normal_power_paths
    cc=PortfolioParameters(**other);cache=dest/'plot/verification_capacity_cache.json'
    saved=read(cache) if cache.exists() else {}
    # JSON将元组存成列表，参数对照按保存格式进行。
    reused=saved.get('capacities',{}) if saved.get('parameters')==json.loads(json.dumps(cc.values())) else {}
    engine=None;rows=[]
    for r in comparisons:
        H=r['确认持续能力_小时'];key=repr(H)
        if abs(H-durations[-1])<=c.duration_check_tolerance_hours:
            endpoint=curve[curve['每户历史次数']==J].iloc[0]
            value=dict(net_lower_MW=endpoint['认证下界_兆瓦'],net_upper_MW=endpoint['认证上界_兆瓦'])
        elif key in reused:value=reused[key]
        else:
            if engine is None:
                choice=frozen_choice(cc,old['invitation_comparison']);power=normal_power_paths(cc)
                engine=SparsePortfolioAnnual(cc,choice,power[None,:,:],choice.engine.reference+float(power.max()))
            known=np.r_[0.,durations];lower=curve[curve['每户历史次数']==int(np.flatnonzero(known<=H)[-1])].iloc[0]
            upper=curve[curve['每户历史次数']==int(np.flatnonzero(known>=H)[0])].iloc[0]
            value=engine.capacity(engine.duration_segments([H]),False,
                lower=engine.e.reference+lower['认证下界_兆瓦'],upper=engine.e.reference+upper['认证上界_兆瓦'])
            reused[key]=value;write_json(cache,dict(parameters=cc.values(),capacities=reused))
        rows.append(dict(**r,认证下界_兆瓦=value['net_lower_MW'],认证上界_兆瓦=value['net_upper_MW'],
            占申报量下界_百分比=100*value['net_lower_MW']/c.nomination_mw))
    save(dest,'verification_capacity.csv',rows)
    capacity=old['invitation_comparison']['capacity_cases'];allocation=[]
    for index,(label,value) in enumerate(zip(['全部20次历史','持续能力已知，未来配对未知','完整信息'],capacity)):
        if index==0:
            point=curve[curve['每户历史次数']==J].iloc[0]
            value=dict(net_lower_MW=max(value['net_lower_MW'],point['认证下界_兆瓦']),net_upper_MW=min(value['net_upper_MW'],point['认证上界_兆瓦']))
        allocation.append(dict(资料=label,净容量下界_兆瓦=value['net_lower_MW'],净容量上界_兆瓦=value['net_upper_MW'],
            占申报量下界_百分比=100*value['net_lower_MW']/c.nomination_mw,
            占申报量上界_百分比=100*value['net_upper_MW']/c.nomination_mw))
    save(dest,'call_extension_comparison.csv',allocation)
    record('实验二',dict(progressive_counts=J,step_counts=[r['历史次数'] for r in process if r['新增确认时长_小时']>c.duration_check_tolerance_hours],
        statistics_effect=rows,main_capacity_rows=allocation,
        strict_complementarity_in_this_comparison=comparisons[-1]['确认持续能力_小时']>max(comparisons[1]['确认持续能力_小时'],comparisons[2]['确认持续能力_小时'])+c.duration_check_tolerance_hours,
        source='原20次完整记录、原联合校准；容量复用同一客户的延续邀约曲线'))




def experiment_three():
    from portfolio_inputs import PortfolioParameters,profiles,definitions
    from normal_customer import normal_power_paths
    from customer_profiles import _ar_scores,_reorder_nodes
    dest=OUT/'exp3';old=read(AUDIT/'portfolio_capacity.json');c=PortfolioParameters(**old['parameters'])
    models=profiles(c);groups=definitions(c)
    names=['正态1',*c.empirical_users,*[f'正态{i+1}' for i in range(1,max(c.normal_sizes))]]
    power={'正态1':normal_power_paths(c)}
    for i,name in enumerate(names[1:],1):
        # 与portfolio_inputs.make_inputs完全相同的分流编号、种子及全年时序。
        innovations=np.random.default_rng(np.random.SeedSequence([c.future_seed,i])).standard_normal((c.future_paths,35040))
        score=_ar_scores(innovations,models[name]['rho']);power[name]=c.nomination_mw*_reorder_nodes(score,models[name]['nodes'])
    membership=[];distribution=[];shapes=[]
    # 分位网格仅用于显示分布；全部64条、35040点功率都参与分位计算。
    probability=np.linspace(0,1,201)
    expected=pd.read_csv(dest/'portfolio_power.csv').set_index('组合');maximum_error=0.
    for group in groups:
        total=sum(w*power[name] for name,w in zip(group['members'],group['weights']))
        values=np.quantile(total,probability);sd=float(total.std());mean=float(total.mean())
        maximum_error=max(maximum_error,abs(sd-expected.loc[group['name'],'生成组合标准差_兆瓦']),abs(mean-expected.loc[group['name'],'可用功率均值_兆瓦']))
        shapes.append(dict(组合=group['name'],均值_兆瓦=mean,标准差_兆瓦=sd,标准差占申报_百分比=100*sd/c.nomination_mw))
        distribution.extend(dict(组合=group['name'],分位数=p,功率_兆瓦=x,功率占申报_百分比=100*x/c.nomination_mw) for p,x in zip(probability,values))
        for name in names:
            w=group['weights'][group['members'].index(name)] if name in group['members'] else 0.
            membership.append(dict(组合=group['name'],客户=name,是否参与=w>0,份额=w,
                该户申报量_兆瓦=c.nomination_mw*w,该户正常负荷_兆瓦=c.load_mw*w))
    if maximum_error>c.solver_tolerance:raise RuntimeError(('重新生成的聚合功率不同于定稿',maximum_error))
    save(dest,'portfolio_members.csv',membership);save(dest,'portfolio_power_quantiles.csv',distribution)
    curve=pd.read_csv(dest/'portfolio_sequence.csv').merge(pd.DataFrame(shapes),on='组合',validate='many_to_one')
    complete={name:g['full_information'] for name,g in old['groups'].items()}
    curve['完整信息容量下界_兆瓦']=curve['组合'].map(lambda x:complete[x]['net_lower_MW'])
    curve['完整信息容量上界_兆瓦']=curve['组合'].map(lambda x:complete[x]['net_upper_MW'])
    curve['信息折减下界_兆瓦']=np.maximum(0,curve['完整信息容量下界_兆瓦']-curve['认证上界_兆瓦'])
    curve['信息折减上界_兆瓦']=np.maximum(0,curve['完整信息容量上界_兆瓦']-curve['认证下界_兆瓦'])
    curve['合计客户记录份数']=curve['户数']*curve['每户历史次数']
    curve['占完整信息容量_百分比']=100*curve['认证约值_兆瓦']/((curve['完整信息容量下界_兆瓦']+curve['完整信息容量上界_兆瓦'])/2)
    save(dest,'portfolio_sequence_variation.csv',curve)
    # 共同记录预算由1、2、3、4户的最小公倍数导出，直接复用现有结果。
    from math import lcm
    budget=lcm(*c.normal_sizes)
    equal=curve[curve['组合'].isin([f'正态{k}户' for k in c.normal_sizes])&(curve['合计客户记录份数']==budget)]
    save(dest,'fixed_record_budget.csv',equal)
    feasible=[];results=read(AUDIT/'portfolio_sequence_capacity.json')['groups']
    for group in groups:
        for case in results[group['name']]['cases']:
            H=case.get('refined_upper_witness_hours',case.get('witness_duration_hours'))
            if H is None:continue
            for name,h in zip(group['members'],H):
                feasible.append(dict(组合=group['name'],每户历史次数=case['count'],客户=name,
                    容量上界所用可行持续能力_小时=h,客户真实能力_小时=c.true_duration_hours))
    save(dest,'customer_feasible_duration.csv',feasible)
    record('实验三',dict(portfolios=len(groups),progressive_points=len(curve),distribution_quantiles=len(distribution),
        power_reconstruction_max_error_MW=maximum_error,
        source='各组合逐次认证、成员份额及完整年度功率路径'))


def experiment_four():
    dest=OUT/'exp4';old=read(dest/'results.json');c=old['parameters'];nomination=c['nomination_mw']
    event=pd.read_csv(dest/'event_holdout.csv');trace=pd.read_csv(dest/'event_trace.csv')
    meta=event[['客户','日期','调用起点_小时','调用终点_小时','调用时长_小时','历史确认时长_小时']]
    trace=trace.merge(meta,on=['客户','日期'],validate='many_to_one')
    trace['调用开始后_小时']=trace['时段开始_小时']-trace['调用起点_小时']
    trace['实际调节占要求_百分比']=100*trace['真实调节_兆瓦']/trace['指令功率_兆瓦']
    trace['基线电表显示调节占要求_百分比']=100*(trace['基线_兆瓦']-trace['电表_兆瓦'])/trace['指令功率_兆瓦']
    covered=np.maximum(0,np.minimum(trace['时段开始_小时']+trace['时段长度_小时'],trace['调用起点_小时']+trace['历史确认时长_小时'])-trace['时段开始_小时'])
    trace['持续时长已被历史覆盖_百分比']=100*covered/trace['时段长度_小时']
    trace['本时段不足_兆瓦时']=np.maximum(trace['指令功率_兆瓦']-trace['真实调节_兆瓦'],0)*trace['时段长度_小时']
    trace['累计不足_兆瓦时']=trace.groupby(['客户','日期'])['本时段不足_兆瓦时'].cumsum()
    rebuilt=trace.groupby(['客户','日期'])['本时段不足_兆瓦时'].sum()
    original=event.set_index(['客户','日期'])['逐时不足电量_兆瓦时']
    error=float(abs(rebuilt-original).max())
    if error>c['solver_tolerance']:raise RuntimeError(('事件15分钟过程与电量不符',error))
    save(dest,'event_trace.csv',trace)
    event['交付电量占要求_百分比']=100*event['真实调节电量_兆瓦时']/event['要求电量_兆瓦时']
    event['逐时不足占要求_百分比']=100*event['逐时不足电量_兆瓦时']/event['要求电量_兆瓦时']
    save(dest,'event_summary.csv',event)
    risk=pd.read_csv(dest/'independent_risk.csv')
    risk['容量占申报_百分比']=100*risk['核定净容量_兆瓦']/nomination
    risk['风险占预算_百分比']=100*risk['独立年度风险_兆瓦时']/risk['风险预算_兆瓦时']
    save(dest,'capacity_risk.csv',risk)
    boundary=pd.read_csv(dest/'short_duration_risk.csv')
    boundary['风险占预算_百分比']=100*boundary['独立年度风险_兆瓦时']/boundary['风险预算_兆瓦时']
    boundary['容量占申报_百分比']=100*boundary['核定净容量_兆瓦']/nomination
    save(dest,'duration_error_risk.csv',boundary)
    comparison=pd.read_csv(dest/'short_duration_trace.csv')
    normal=meta[meta['客户']=='正态1'].drop(columns='客户')
    comparison=comparison.merge(normal,on='日期',validate='many_to_one')
    comparison['调用开始后_小时']=comparison['时段开始_小时']-comparison['调用起点_小时']
    comparison=comparison.melt(id_vars=['日期','时段开始_小时','调用开始后_小时','调用时长_小时','指令功率_兆瓦'],
        value_vars=['十二小时客户实际调节_兆瓦','基准客户实际调节_兆瓦'],var_name='客户能力',value_name='实际调节_兆瓦')
    comparison['客户能力']=comparison['客户能力'].map({'十二小时客户实际调节_兆瓦':'客户可持续12小时',
        '基准客户实际调节_兆瓦':f'客户可持续{c["true_duration_hours"]:g}小时'})
    comparison['实际调节占要求_百分比']=100*comparison['实际调节_兆瓦']/comparison['指令功率_兆瓦']
    save(dest,'duration_call_trace.csv',comparison)
    sensitivity=pd.read_csv(dest/'annual_sensitivity.csv');rows=[]
    groups=[('ISO服务要求',['完整信息_原安排','邀约最长12小时','邀约最长15小时'],'完整信息_原安排'),
        ('调用后增加用电',['完整信息_原安排','调用后增加用电25%','调用后增加用电50%'],'完整信息_原安排'),
        ('固定记录改变核验范围',['固定记录_核验标准差率14.44%','固定记录_核验标准差率28.65%'],'固定记录_核验标准差率14.44%'),
        ('改变实际负荷波动',['误差标准差率7.15%','误差标准差率14.44%','误差标准差率28.65%'],'误差标准差率14.44%')]
    for family,labels,base in groups:
        reference=sensitivity[sensitivity['比较']==base].iloc[0]
        for label in labels:
            r=sensitivity[sensitivity['比较']==label].iloc[0]
            rows.append(dict(比较类别=family,情景=label,基准情景=base,净容量下界_兆瓦=r.net_lower_MW,净容量上界_兆瓦=r.net_upper_MW,
                与基准差额下界_兆瓦=r.net_lower_MW-reference.net_upper_MW,
                与基准差额上界_兆瓦=r.net_upper_MW-reference.net_lower_MW,
                认证占申报量下界_百分比=100*r.net_lower_MW/nomination))
    surplus=pd.read_csv(dest/'margin_sensitivity.csv');base=surplus[surplus['非缺电时段余量尺度_兆瓦']==c['system_surplus_mw']].iloc[0]
    for r in surplus.to_dict('records'):
        rows.append(dict(比较类别='非缺电时段余量',情景=f'余量尺度{r["非缺电时段余量尺度_兆瓦"]:g}兆瓦',基准情景=f'余量尺度{c["system_surplus_mw"]:g}兆瓦',
            净容量下界_兆瓦=r['资源净容量下界_兆瓦'],净容量上界_兆瓦=r['资源净容量上界_兆瓦'],
            与基准差额下界_兆瓦=r['资源净容量下界_兆瓦']-base['资源净容量上界_兆瓦'],
            与基准差额上界_兆瓦=r['资源净容量上界_兆瓦']-base['资源净容量下界_兆瓦'],
            认证占申报量下界_百分比=100*r['资源净容量下界_兆瓦']/nomination))
    save(dest,'sensitivity_by_group.csv',rows)
    completion=read(dest/'completion_check.json')
    record('实验四',dict(event_curves=len(event),event_quarter_rows=len(trace),event_accounting_max_error_MWh=error,
        independent_risk_rows=len(risk),maximum_risk_to_budget_percent=float(risk['风险占预算_百分比'].max()),
        coverage=completion['coverage']['true_complete_physical_history_retained'],sensitivity_comparisons=len(rows),
        mature_tool_cases=completion['mature_tool_cases'],mature_tool_max_gap_MW=completion['mature_tool_max_gap_MW'],
        source='完整真实事件15分钟过程、独立年度风险及既有物理敏感性；未重跑年度容量根'))


def update_index():
    index=[]
    folders=['hubei','exp1','exp2','exp3','exp4']
    for folder in folders:
        target=OUT/folder/'plot'
        for path in sorted(target.glob('*.csv')):
            frame=pd.read_csv(path)
            kind='当前实验结果及过程'
            index.append(dict(实验=folder,文件=str(path.relative_to(OUT)).replace('\\','/'),行数=len(frame),
                数据类别=kind,字段='；'.join(frame.columns)))
    # 核心补算按原实验目录归档；索引同时列出其比较表、风险根和原始记录核验。
    extra=[('exp4',OUT/'exp4/sens'),
        ('exp4',OUT/'exp4/sens/duration'),
        ('exp3',OUT/'exp4/sens/common'),
        ('hubei',OUT/'hubei/diag')]
    paths=[(folder,p) for folder,target in extra for p in sorted(target.glob('*.csv'))
        if p.name!='portfolio_power.csv']
    paths.append(('exp1',OUT/'exp1/margin_sensitivity.csv'))
    for folder,path in paths:
        if not path.exists():continue
        frame=pd.read_csv(path)
        index.append(dict(实验=folder,文件=str(path.relative_to(OUT)).replace('\\','/'),行数=len(frame),
            数据类别='核心参数补算及记录核验',字段='；'.join(frame.columns)))
    pd.DataFrame(index).to_csv(OUT/'plot_index.csv',index=False,encoding='utf-8-sig')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['data','1','2','3','4','all'],default='all');args=parser.parse_args()
    set_num_threads(CORES)
    stages={'data':data,'1':experiment_one,'2':experiment_two,'3':experiment_three,'4':experiment_four}
    for name,run in stages.items():
        if args.stage in (name,'all'):run()
    update_index()


if __name__=='__main__':main()
