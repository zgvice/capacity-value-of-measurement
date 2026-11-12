"""用核定额度检验独立年度风险，并用真实长调用检查持续能力边界。"""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from common import AUDIT,write_json
from run_experiment_four import inputs,subset,duration,future_power,untreated_load,say,DEST,SAVED
from historical_duration import HistoricalDuration
from portfolio_annual import frozen_choice
from portfolio_sparse_physics import SparsePortfolioAnnual


def main():
    c,p,P,maximum,data,model,calibration,old=inputs()
    saved=json.loads(SAVED.read_text(encoding='utf-8'))
    choice=frozen_choice(c,calibration['invitation_comparison'])
    independent=replace(c,future_seed=c.validation_future_seed)
    names=['正态1',*c.empirical_users];prototype=list(data['prototype_names'])
    dates=np.array([str(d['task']['date']) for d in p['designs']])
    selected=~np.isin(dates[p['call_ids']],c.winter_dates+c.summer_dates)
    rows=[];observations=[]
    for name in names:
        i=prototype.index(name);group='正态1户' if name=='正态1' else name
        full=dict(p,baseline=data['call_baselines'][i])
        bound=P if name=='正态1' else c.nomination_mw*model[name]['values'].max()
        history=HistoricalDuration(full,data['call_meters'][i],p['J'],bound,
            bias_bound=c.load_mw*c.bias_bound_rate,solver_tolerance=c.solver_tolerance,
            duration_check_tolerance=c.duration_check_tolerance_hours)
        v=saved['heldout'][name]
        observations.append(dict(客户=name,十五次确认时长_小时=v['duration_hours'],
            二十次确认时长_小时=v['all_history_duration_hours'],
            留出记录合用后原限制能力仍可行=history.solve(v['duration_hours']+c.duration_check_tolerance_hours).success,
            真实能力仍在候选集合内=history.solve(c.true_duration_hours).success))
        power=future_power(independent,model,name)
        engine=SparsePortfolioAnnual(c,choice,power[None,:,:],old['groups'][group]['full_information']['total_upper_MW'])
        for label,value,H in [('十五次历史',v['capacity'],v['duration_hours']),('二十次历史',v['all_history_capacity'],v['all_history_duration_hours'])]:
            q=value['total_lower_MW']
            actual=engine.risk(q,engine.duration_segments([c.true_duration_hours]),True)['upper_MWh']
            rows.append(dict(客户=name,资料=label,核定净容量_兆瓦=value['net_lower_MW'],
                系统总增载边界_兆瓦=q,历史确认时长_小时=H,真实持续能力_小时=c.true_duration_hours,
                独立年度风险_兆瓦时=actual,风险预算_兆瓦时=c.risk_budget_mwh,超出预算_兆瓦时=max(actual-c.risk_budget_mwh,0.)))
            say(f'{name}，{label}：核定{value["net_lower_MW"]:.2f}兆瓦，独立年度风险{actual:.2f}兆瓦时')
    pd.DataFrame(rows).to_csv(DEST/'independent_risk.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(observations).to_csv(DEST/'holdout_verification.csv',index=False,encoding='utf-8-sig')

    # 只改变原模型已有的客户持续能力。ISO继续由记录核验，作者在事后按12小时真能力核算风险。
    Htrue=c.boundary_duration_hours
    u=untreated_load(c,p,0);left=p['left'];starts=p['starts'];stops=p['stops']
    share=np.clip((np.minimum(left+.25,np.minimum(stops,starts+c.true_duration_hours))-np.maximum(left,starts))/.25,0,1) # 原15分钟重叠。
    potential=np.divide(u-data['call_meters'][0],share,out=np.zeros_like(u),where=share>0)
    short_share=np.clip((np.minimum(left+.25,np.minimum(stops,starts+Htrue))-np.maximum(left,starts))/.25,0,1)
    short_delivery=np.minimum(potential,u)*short_share;meter=u-short_delivery
    train,_=subset(p,selected);train['baseline']=data['call_baselines'][0,selected]
    full=dict(p,baseline=data['call_baselines'][0])
    Htrain=duration(c,train,meter[selected],P,maximum);Hall=duration(c,full,meter,P,maximum)
    forecast=SparsePortfolioAnnual(c,choice,future_power(c,model,'正态1')[None,:,:],old['groups']['正态1户']['full_information']['total_upper_MW'])
    train_value=saved['heldout']['正态1']['capacity'] if abs(Htrain-saved['heldout']['正态1']['duration_hours'])<=c.solver_tolerance else forecast.capacity(forecast.duration_segments([Htrain]),False)
    all_value=forecast.capacity(forecast.duration_segments([Hall]),False,lower=train_value['total_lower_MW'])
    truth=SparsePortfolioAnnual(c,choice,future_power(independent,model,'正态1')[None,:,:],old['groups']['正态1户']['full_information']['total_upper_MW'])
    compared=[]
    for label,value,Hassumed in [('按十五次记录核定',train_value,Htrain),('按二十次记录核定',all_value,Hall),
        (f'直接按{c.true_duration_hours:g}小时能力认定',old['groups']['正态1户']['full_information'],c.true_duration_hours)]:
        actual=truth.risk(value['total_lower_MW'],truth.duration_segments([Htrue]),True)['upper_MWh']
        compared.append(dict(处理=label,客户真实持续能力_小时=Htrue,核定使用的能力_小时=Hassumed,
            核定净容量_兆瓦=value['net_lower_MW'],独立年度风险_兆瓦时=actual,
            风险预算_兆瓦时=c.risk_budget_mwh,超出预算_兆瓦时=max(actual-c.risk_budget_mwh,0.)))
        say(f'12小时客户，{label}：容量{value["net_lower_MW"]:.2f}，真实年度风险{actual:.2f}')
    event_rows=[];event_trace=[]
    for j,d in enumerate(p['designs']):
        date=str(d['task']['date'])
        if date not in c.winter_dates+c.summer_dates:continue
        ids=np.flatnonzero(p['call_ids']==j);dur=d['task']['duration_hours']
        loss=.25*np.maximum(c.nomination_mw*share[ids]-short_delivery[ids],0).sum()
        event_rows.append(dict(日期=date,邀约时长_小时=dur,客户真实持续能力_小时=Htrue,
            超出客户持续能力的邀约_小时=max(dur-Htrue,0.),实际调节电量_兆瓦时=float(.25*short_delivery[ids].sum()),
            逐时不足电量_兆瓦时=float(loss)))
        for at in ids:
            event_trace.append(dict(日期=date,时段开始_小时=left[at],指令功率_兆瓦=c.nomination_mw,
                十二小时客户实际调节_兆瓦=short_delivery[at],基准客户实际调节_兆瓦=u[at]-data['call_meters'][0,at]))
    pd.DataFrame(compared).to_csv(DEST/'short_duration_risk.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(event_rows).to_csv(DEST/'short_duration_delivery.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(event_trace).to_csv(DEST/'short_duration_trace.csv',index=False,encoding='utf-8-sig')
    write_json(DEST/'validation_params.json',dict(future_power_seed=c.validation_future_seed,
        true_duration_hours=c.true_duration_hours,boundary_duration_hours=Htrue,
        annual_atoms=40300,annual_paths=1788,annual_hours=8760,quarter_points=35040,
        risk_budget_MWh=c.risk_budget_mwh,tail_level=c.tail_level,
        source='沿用原客户分布、官方年度场景及邀约；独立改变功率轨迹种子。12小时为事先选定的客户能力检验。',
        risk_evaluation='将核定净容量加回同标准系统基准，在该原认证边界核算真实物理规律下的新增缺供尾部风险。',
        true_capability_used_by='作者事后风险核对；ISO仍使用电表与基线资料。'))


if __name__=='__main__':main()
