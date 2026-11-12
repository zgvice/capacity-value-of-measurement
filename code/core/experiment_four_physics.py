"""当前正态客户的完整年度物理对照；逐期状态沿用实验三已核定的界。"""
from __future__ import annotations
import json,copy,time
from dataclasses import replace
import numpy as np
import pandas as pd
from numba import njit,prange,set_num_threads
from common import OUT,AUDIT,CORES,write_json
from run_experiment_four import inputs,duration,future_power,untreated_load,say,DEST,SAVED
from normal_customer import build_history
from portfolio_annual import frozen_choice
from portfolio_sparse_physics import SparsePortfolioAnnual,base_prefix,base_value
from annual_call_schedule import ScheduleChoice


@njit(cache=True,parallel=True)
def returns_geometry(call_ptr,calls,hours):
    # 额外用电安排在邀约结束后完整的非调用小时；r不用于恢复可调能力。
    result=np.full((len(calls),hours),-1,np.int32)
    for k in prange(len(call_ptr)-1):
        first,last=call_ptr[k],call_ptr[k+1]
        for at in range(first,last):
            h=int(np.ceil(calls[at,2]));next_call=at+1;used=0
            while h<8760 and used<hours:  # 原年度8760小时，不丢弃末端额外用电。
                while next_call<last and calls[next_call,2]<=h:next_call+=1
                if next_call>=last or calls[next_call,0]>=h+1:
                    result[at,used]=h;used+=1
                h+=1
    return result


@njit(cache=True,parallel=True)
def deficit_prefix(q,g,power,profiles,ptr,hours,values,surplus):
    result=np.zeros((len(power),len(hours)+len(profiles)))
    for rj in prange(len(power)*len(profiles)):
        r,j=rj//len(profiles),rj%len(profiles);f=profiles[j];total=0.
        for at in range(ptr[j],ptr[j+1]):
            h=hours[at];actual=max(q+values[at],0.);background=max(q-surplus*g[f,h],0.)
            for quarter in range(4):total+=.25*(min(actual,power[r,4*h+quarter])-background)  # 每小时四个15分钟。
            result[r,at+j+1]=total
    return result


@njit(cache=True)
def deficit_at(t,q,r,j,g,power,profiles,ptr,hours,values,prefix,surplus):
    h=min(int(t),8760);lo,hi=ptr[j],ptr[j+1];at=np.searchsorted(hours[lo:hi],h)
    result=prefix[r,lo+j+at]
    if h<8760 and at<hi-lo and hours[lo+at]==h:
        f=profiles[j];actual=max(q+values[lo+at],0.);background=max(q-surplus*g[f,h],0.)
        for quarter in range(4):
            length=max(0.,min(t,h+(quarter+1)/4.)-(h+quarter/4.))
            result+=length*(min(actual,power[r,4*h+quarter])-background)
    return result


@njit(cache=True,parallel=True)
def annual_payback(q,H,fraction,g,power,profiles,ptr,hours,values,s0,total,base,extra_prefix,call_ptr,calls,returns,variants,surplus):
    K=len(call_ptr)-1;R=len(power);out=np.empty((R,K))
    for k in prange(K):
        j=k//variants;f=profiles[j]
        extra=np.zeros(8760);touched=np.empty(8760,np.int32);seen=np.zeros(8760,np.int64)
        for r in range(R):
            saved=0.;used=0
            for at in range(call_ptr[k],call_ptr[k+1]):
                # 若年度末无法安排完整额外用电，禁止该次调节，保留原跨期收支。
                if returns[at,-1]<0:continue
                start=calls[at,0];stop=min(calls[at,2],start+H)
                if stop<=start:continue
                energy=base_value(stop,q,f,g,base,surplus)-base_value(start,q,f,g,base,surplus)
                energy+=deficit_at(stop,q,r,j,g,power,profiles,ptr,hours,values,extra_prefix,surplus)
                energy-=deficit_at(start,q,r,j,g,power,profiles,ptr,hours,values,extra_prefix,surplus)
                saved+=energy
                for a in range(returns.shape[1]):
                    h=returns[at,a]
                    if seen[h]!=r+1:seen[h]=r+1;touched[used]=h;used+=1;extra[h]=0.
                    extra[h]+=fraction*energy/returns.shape[1]
            cost=0.
            for a in range(used):
                h=touched[a];at=np.searchsorted(hours[ptr[j]:ptr[j+1]],h)
                gap=-surplus*g[f,h]
                if at<ptr[j+1]-ptr[j] and hours[ptr[j]+at]==h:gap=values[ptr[j]+at]
                cost+=max(gap+q+extra[h],0.)-max(gap+q,0.)
            out[r,k]=max(total[j]-saved+cost-s0[j],0.)
    return out


class PaybackAnnual(SparsePortfolioAnnual):
    def __init__(self,c,choice,power,fraction,return_hours,upper):
        super().__init__(c,choice,power[None,:,:],upper)
        self.fraction=fraction;self.H=c.true_duration_hours
        self.returns=[returns_geometry(p['ptr'],p['calls'],return_hours) for p in choice.plans]
    def losses(self,q,segments):
        # 本轮正态功率下界高于非缺电时段的最大缺口；逐节点检查后采用等价积分。
        if q>float(self.unit.min()):raise RuntimeError('需启用一般功率截断积分')
        base=base_prefix(q,self.e.g,self.e.surplus)
        prefix=deficit_prefix(q,self.e.g,self.unit[0],self.e.profiles,self.e.ptr,self.e.hours,self.e.values,self.e.surplus)
        total=self.e.baseline(q)[1];best=None
        for plan,ret in zip(self.choice.plans,self.returns):
            value=annual_payback(q,self.H,self.fraction,self.e.g,self.unit[0],self.e.profiles,self.e.ptr,self.e.hours,self.e.values,
                self.e.s0,total,base,prefix,plan['ptr'],plan['calls'],ret,self.e.variants,self.e.surplus)
            best=value if best is None else np.minimum(best,value)
        return best


def native_inputs(c,choice,power,old):
    e=choice.engine;rank=np.argsort(e.s0)
    selected=np.unique(np.r_[rank[-2:],rank[len(rank)//2]])  # 按原缺供量选择最严重两年及中位年，先于比较结果。
    qvalues=[e.reference,old['groups']['正态1户']['full_information']['total_lower_MW']]
    rows=[];meta=[];case=0
    plan=choice.plans[-1]
    for j in selected:
        f=e.profiles[j];gap=-e.surplus*e.g[f].copy();gap[e.hours[e.ptr[j]:e.ptr[j+1]]]=e.values[e.ptr[j]:e.ptr[j+1]]
        for r in (0,len(power)//2):  # 固定积分轨迹的首条及中间条，属于数值对照取样。
            steps=8760*12  # 原调用边界为5分钟整数倍；15分钟功率在三个5分钟内保持原值。
            k=int(j)*e.variants;available=np.zeros(steps)
            for at in range(plan['ptr'][k],plan['ptr'][k+1]):
                start=plan['calls'][at,0];stop=min(plan['calls'][at,2],start+c.true_duration_hours)
                qq=np.arange(int(np.floor(start*12)),min(steps,int(np.ceil(stop*12))))
                overlap=np.maximum(0.,np.minimum((qq+1)/12.,stop)-np.maximum(qq/12.,start))*12
                available[qq]=power[r,qq//3]*overlap
            for q in qvalues:
                # 200000MW是双方共同相减的账面平移，使原生工具输入非负；物理净缺口不变。
                load=np.full(steps,int(round((200000+q)*1000)),np.int64)
                supply=np.rint((200000-np.repeat(gap,12))*1000).astype(np.int64)
                resource=np.rint(available*1000).astype(np.int64)  # MW转kW，原生整数功率误差≤0.001MW。
                expected=np.maximum(load-supply-resource,0)
                rows.append(np.column_stack([np.full(steps,case),np.arange(steps),load,supply,resource,expected]))
                original=np.maximum(np.repeat(gap,12)+q-available,0.)
                meta.append(dict(case=case,annual_group=int(j),power_path=r,q_MW=q,
                    exact_original_shortfall_MWh=float(original.sum()/12),rounded_shortfall_MWh=float(expected.sum()/12000),
                    max_rounding_difference_MW=float(np.max(abs(original-expected/1000)))))
                case+=1
    np.savetxt(DEST/'pras_input.csv',np.vstack(rows),fmt='%d',delimiter=',',header='case,quarter,load_kW,background_kW,resource_kW,expected_shortfall_kW',comments='')
    write_json(DEST/'pras_input_notes.json',dict(cases=meta,quarter_points=35040,tool_steps=8760*12,tool_minutes=5,service_hours=8760,
        original_annual_atoms=40300,full_physics_source='checks/portfolio_capacity.json/正态1户/physical_state_check',
        source='当前年度缺口、同一冻结邀约、同一正态功率积分轨迹；额外状态约束沿用已证明不绑定的原界。'))


def main():
    DEST.mkdir(exist_ok=True);set_num_threads(CORES)
    c,public,bound,maximum,data,model,calibration,old=inputs();c=replace(c,capacity_tolerance_mw=c.sensitivity_tolerance_mw)
    source=calibration['invitation_comparison'];choice=frozen_choice(c,source);power=future_power(c,model,'正态1')
    path=DEST/'annual_sensitivity.json';result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(cases={})
    expected=json.loads(json.dumps(c.values()))
    physical_keys=('nomination_mw','response_mean_ratio','response_sd_ratio','response_rho15','future_paths',
        'quadrature_nodes','future_seed','true_duration_hours','maximum_extension_hours','tail_level','risk_budget_mwh','system_surplus_mw')
    before=result.get('parameters',{})
    if any(k in before and before[k]!=expected[k] for k in physical_keys):result=dict(cases={})
    rule=public['verification_rule']['settings']
    if result.get('history_rule_settings')!=rule:
        result['cases']={k:v for k,v in result['cases'].items() if not k.startswith(('误差','20次历史','固定记录'))}
    result['history_rule_settings']=rule
    result['parameters']=expected
    def save(label,H,value,**extra):
        result['cases'][label]={**value,**extra,'duration_hours':H};write_json(path,result)
        pd.DataFrame([dict(比较=k,**v) for k,v in result['cases'].items()]).to_csv(DEST/'annual_sensitivity.csv',index=False,encoding='utf-8-sig')
        say(f'{label}：净容量{value["net_lower_MW"]:.2f}—{value["net_upper_MW"]:.2f}兆瓦')
    full=old['groups']['正态1户']['full_information'];save('完整信息_原安排',c.true_duration_hours,full)
    current=json.loads(SAVED.read_text(encoding='utf-8'))['heldout']['正态1']
    save('20次历史_原安排',current['all_history_duration_hours'],current['all_history_capacity'])
    for limit in c.invitation_limits_hours:
        label=f'邀约最长{limit:g}小时'
        if label in result['cases']:continue
        altered=copy.copy(choice);altered.plans=[]
        for plan in choice.plans:
            calls=plan['calls'].copy();calls[:,2]=np.minimum(calls[:,2],calls[:,0]+limit)
            altered.plans.append(dict(plan,calls=calls))
        engine=SparsePortfolioAnnual(c,altered,power[None,:,:],full['total_upper_MW'])
        value=engine.capacity(engine.duration_segments([c.true_duration_hours]),True,upper=full['total_upper_MW'])
        save(label,c.true_duration_hours,value,invitation_limit_hours=limit)
    for rate in c.error_scales:
        label=f'误差标准差率{100*rate:g}%'
        if label in result['cases']:continue
        cc=replace(c,error_sd_rate=rate);ratio=rate/c.error_sd_rate
        # 用原同一高斯随机数缩放实际使用的基线日及调用日；未使用日期不进入计算。
        p=dict(public);p['baseline']=c.load_mw+(data['call_baselines'][0]-c.load_mw)*ratio
        p['bounds']=public['bounds']*ratio
        true_before=untreated_load(c,public,0);true_after=c.load_mw+(true_before-c.load_mw)*ratio
        prior_after=c.load_mw+(data['original_meters'][0]-c.load_mw)*ratio
        if min(true_after.min(),prior_after.min())<0:raise RuntimeError('实际使用负荷存在负值')
        # 和实验三相同：真实调节不得超过当时正常用电；不会给ISO提供true_after。
        fraction=np.clip((np.minimum(public['left']+.25,np.minimum(public['stops'],public['starts']+c.true_duration_hours))-
            np.maximum(public['left'],public['starts']))/.25,0,1)  # 原15分钟计量时段内的实际调用比例。
        potential=np.divide(true_before-data['call_meters'][0],fraction,out=np.zeros_like(true_before),where=fraction>0)
        delivered=np.minimum(potential,true_after)*fraction
        m=true_after-delivered;H=duration(cc,p,m,bound,maximum)
        if rate==c.error_sd_rate:value=current['all_history_capacity']
        else:
            engine=SparsePortfolioAnnual(c,choice,power[None,:,:],full['total_upper_MW'])
            value=engine.capacity(engine.duration_segments([H]),False,upper=full['total_upper_MW'])
        save(label,H,value,error_SD_MW=rate*c.load_mw,error_SD_rate=rate,
            minimum_used_normal_load_MW=float(min(true_after.min(),prior_after.min())),
            physically_limited_points=int(np.sum(potential>true_after)),
            reduced_historical_delivery_MWh=float(.25*np.sum(np.maximum(potential-true_after,0)*fraction)),
            comparison='相同高斯随机数下的负荷波动情景；真实响应保留正常用电上限。')
    # 单独比较纯核验范围效应：真实客户和全部原始记录固定，只扩大ISO误差界。
    for rate in c.error_scales:
        if rate<c.error_sd_rate:continue
        label=f'固定记录_核验标准差率{100*rate:g}%'
        if label in result['cases']:continue
        restricted=dict(public,baseline=data['call_baselines'][0],bounds=public['bounds']*(rate/c.error_sd_rate))
        H=duration(c,restricted,data['call_meters'][0],bound,maximum)
        if rate==c.error_sd_rate:value=current['all_history_capacity']
        else:
            engine=SparsePortfolioAnnual(c,choice,power[None,:,:],full['total_upper_MW'])
            value=engine.capacity(engine.duration_segments([H]),False,upper=full['total_upper_MW'])
        save(label,H,value,error_SD_MW=rate*c.load_mw,error_SD_rate=rate,
            physically_limited_points=0,reduced_historical_delivery_MWh=0.,
            comparison='同一客户、同一真实调节和全部15分钟记录；只扩大核验误差界。')
    if not (DEST/'pras_results.json').exists():native_inputs(c,choice,power,old)
    for fraction in c.payback_fractions:
        if fraction==0:continue  # 零额外用电即已归档的完整信息主结果。
        label=f'调用后增加用电{100*fraction:g}%'
        if label in result['cases']:continue
        engine=PaybackAnnual(c,choice,power,fraction,c.payback_hours,full['total_upper_MW'])
        value=engine.capacity(engine.duration_segments([c.true_duration_hours]),True,upper=full['total_upper_MW'])
        save(label,c.true_duration_hours,value,payback_fraction=fraction,payback_hours=c.payback_hours)
    result['full_physics']=old['groups']['正态1户']['physical_state_check'];result['annual_atoms']=40300
    result['hours']=8760;result['exact_annual_paths']=1788;write_json(path,result)


def surplus_comparison():
    c,public,bound,maximum,data,model,calibration,old=inputs()
    c=replace(c,capacity_tolerance_mw=c.sensitivity_tolerance_mw)
    file=DEST/'margin_sensitivity.json'
    result=json.loads(file.read_text(encoding='utf-8')) if file.exists() else dict(cases={})
    if result.get('parameters') is not None and result['parameters']!=json.loads(json.dumps(c.values())):
        raise RuntimeError('供需余量结果的共同参数与当前情景不同')
    result['parameters']=c.values()
    for factor in c.surplus_factors:
        scale=c.system_surplus_mw*factor;label=f'{scale:g}'
        if label in result['cases']:continue
        if factor==1:
            value=old['groups']['正态1户']['full_information'];reference=calibration['no_resource_reference_MW'];plans=calibration['invitation_comparison']['plans']
        else:
            cc=replace(c,system_surplus_mw=scale);choice=ScheduleChoice(cc);choice.add_plan(choice.engine.reference)
            power=future_power(cc,model,'正态1')
            engine=SparsePortfolioAnnual(cc,choice,power[None,:,:],choice.engine.reference+float(power.max()))
            seg=engine.duration_segments([cc.true_duration_hours]);value=engine.capacity(seg,True)
            for _ in range(cc.schedule_iterations):
                if not choice.add_plan(value['total_upper_MW']):break
                engine.clear_geometry();updated=engine.capacity(seg,True,lower=value['total_lower_MW'])
                improvement=updated['net_lower_MW']-value['net_lower_MW'];value=updated
                if improvement<=cc.capacity_tolerance_mw:break
            reference=choice.engine.reference
            plans=[dict(trigger_reference_MW=p['threshold_MW']) for p in choice.plans]
            # 正态积分功率的全时段上界小于原状态供给上界，任何调用安排均满足所有中间状态界。
            if power.max()>cc.state_power_ratio*cc.nomination_mw:raise RuntimeError('新的余量情景需要显式状态求解')
        result['cases'][label]=dict(surplus_mw=scale,no_resource_reference_MW=reference,**value,plans=plans)
        write_json(file,result)
        pd.DataFrame([dict(非缺电时段余量尺度_兆瓦=v['surplus_mw'],原系统容量基准_兆瓦=v['no_resource_reference_MW'],
            资源净容量下界_兆瓦=v['net_lower_MW'],资源净容量上界_兆瓦=v['net_upper_MW']) for v in result['cases'].values()]).to_csv(
            DEST/'margin_sensitivity.csv',index=False,encoding='utf-8-sig')
        say(f'供需余量尺度{scale:g}兆瓦：净容量{value["net_lower_MW"]:.2f}—{value["net_upper_MW"]:.2f}兆瓦')
    result['source']='1000MW为既定补全尺度，500/1500MW为作者上下50%情景；官方缺电时段原值、全部场景概率及全年时序保持不变。'
    result['invitation']='每档均沿用同一缺口触发和服务格式生成规则重新形成邀约；未将原邀约时间表当成新供需情景的最优安排。'
    write_json(file,result)


if __name__=='__main__':main();surplus_comparison()
