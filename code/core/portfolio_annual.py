"""逐户持续能力下的全年组合调度；共用实验二冻结的邀约与年度场景。"""
from __future__ import annotations
import numpy as np
from common import cvar
from annual_call_schedule import ScheduleChoice
from fixed_call_profile_physics import background_prefix,annual_losses


def frozen_choice(c,saved):
    choice=ScheduleChoice(c)
    for plan in saved['plans']:choice.add_plan(plan['trigger_reference_MW'])
    return choice


class PortfolioAnnual:
    def __init__(self,c,choice,powers):
        self.c=c;self.choice=choice;self.e=choice.engine;self.unit=np.asarray(powers)
        self.M,self.R,self.T=self.unit.shape;self.full=(1<<self.M)-1
        self.profiles={0:np.zeros((self.R,self.T))}
        for subset in range(1,1<<self.M):
            self.profiles[subset]=sum(self.unit[i] for i in range(self.M) if subset&(1<<i))
        self.e.set_power(self.profiles[self.full]);self.q=None;self.prefix={}
        self.zero=np.zeros_like(self.e.s0);self.probability=self.e.pi[None,:]/self.R

    def prepare(self,q):
        if self.q!=q:
            self.q=q;self.prefix={}
            self.total=np.repeat(self.e.baseline(q)[1],self.e.variants)

    def saved(self,q,key,duration,plan):
        self.prepare(q)
        if duration<=0 or key==0:return np.zeros((self.R,len(self.e.pi)))
        power=self.profiles[key]
        if key not in self.prefix:self.prefix[key]=background_prefix(q,self.e.g,power,self.e.surplus)
        # 原核函数令无资源缺供参照为零，返回物理缺供S1；由总缺供减S1取得节省量。
        physical=annual_losses(q,duration,self.e.g,power,self.prefix[key],self.e.profiles,
            self.e.ptr,self.e.hours,self.e.values,self.zero,self.e.baseline(q)[1],
            plan['ptr'],plan['calls'],self.e.variants,self.e.surplus)
        return self.total[None,:]-physical

    def duration_segments(self,H):
        H=np.asarray(H,float);knots=np.unique(np.r_[0.,H]);segments=[]
        for start,stop in zip(knots[:-1],knots[1:]):
            subset=sum(1<<i for i in range(self.M) if H[i]>start)
            if subset and stop>start:segments.append((float(start),float(stop),subset))
        return segments

    def envelope_segments(self,thresholds):
        knots=np.unique(np.r_[0.,list(thresholds.values())]);segments=[]
        for start,stop in zip(knots[:-1],knots[1:]):
            allowed=[s for s,t in thresholds.items() if t<=start+self.c.time_tolerance_hours]
            minimal=tuple(sorted(s for s in allowed if not any(other!=s and (other&s)==other for other in allowed)))
            if 0 in minimal:break
            if len(minimal)==1:key=minimal[0]
            else:
                key=('minimum',minimal)
                if key not in self.profiles:
                    self.profiles[key]=np.minimum.reduce([self.profiles[s] for s in minimal])
            segments.append((float(start),float(stop),key))
        return segments

    def losses(self,q,segments):
        self.prepare(q);best=None
        for plan in self.choice.plans:
            saved=np.zeros((self.R,len(self.e.pi)))
            for start,stop,key in segments:
                saved+=self.saved(q,key,stop,plan)
                if start>0:saved-=self.saved(q,key,start,plan)
            loss=np.maximum(self.total[None,:]-saved-np.repeat(self.e.s0,self.e.variants)[None,:],0.)
            if best is None:best=loss
            else:np.minimum(best,loss,out=best)
        return best

    def risk(self,q,segments,known_joint):
        loss=self.losses(q,segments)
        if known_joint:
            value=cvar(loss,self.probability,self.c.tail_level)
            return dict(lower_MWh=value,upper_MWh=value)
        return self.e.transport.risk(loss)

    def capacity(self,segments,known_joint,lower=None,upper=None):
        reference=self.e.reference
        if not segments:
            return dict(net_lower_MW=0.,net_upper_MW=0.,total_lower_MW=self.e.reference_bracket['lower_MW'],total_upper_MW=self.e.reference_bracket['upper_MW'])
        lo=self.e.reference_bracket['lower_MW'] if lower is None else lower
        hi=self.e.reference_bracket['upper_MW']+float(self.profiles[self.full].max()) if upper is None else upper
        low=self.risk(lo,segments,known_joint);high=self.risk(hi,segments,known_joint);calls=2
        step=max(hi-lo,self.c.capacity_tolerance_mw)  # 用已有结果给初始括区间；不能夹根时逐步扩大数值搜索。
        while low['upper_MWh']>self.c.risk_budget_mwh:
            lo=max(self.e.reference_bracket['lower_MW'],lo-step);step*=2
            low=self.risk(lo,segments,known_joint);calls+=1
            if lo==self.e.reference_bracket['lower_MW'] and low['upper_MWh']>self.c.risk_budget_mwh:
                raise RuntimeError('组合容量下端未满足原风险要求')
        step=max(hi-lo,self.c.capacity_tolerance_mw)
        while high['lower_MWh']<=self.c.risk_budget_mwh:
            if high['upper_MWh']<=self.c.risk_budget_mwh:lo,low=hi,high
            hi+=step;step*=2
            high=self.risk(hi,segments,known_joint);calls+=1
        while hi-lo>self.c.capacity_tolerance_mw:
            # 在已夹住预算的两点之间插值，每步至少缩小区间10%；只加速求根，不放松风险要求。
            fraction=np.clip((self.c.risk_budget_mwh-low['upper_MWh'])/(high['lower_MWh']-low['upper_MWh']),.1,.9)
            mid=lo+fraction*(hi-lo);value=self.risk(mid,segments,known_joint);calls+=1
            if value['upper_MWh']<=self.c.risk_budget_mwh:lo,low=mid,value
            elif value['lower_MWh']>self.c.risk_budget_mwh:hi,high=mid,value
            else:
                # 插值点恰在风险根附近时，用容量精度四分之一的左右点重新夹住根；不将不确定风险当作通过。
                step=min((hi-lo)/4,self.c.capacity_tolerance_mw/4)
                left,right=mid-step,mid+step
                a,b=self.risk(left,segments,known_joint),self.risk(right,segments,known_joint);calls+=2
                if a['upper_MWh']<=self.c.risk_budget_mwh< b['lower_MWh']:lo,low,hi,high=left,a,right,b
                else:raise RuntimeError(('风险根邻域仍未夹住预算',left,right,a['lower_MWh'],a['upper_MWh'],b['lower_MWh'],b['upper_MWh']))
            if calls%4==0:print(f'  年度容量求根：{lo-reference:.3f}—{hi-reference:.3f}兆瓦',flush=True)
        return dict(total_lower_MW=lo,total_upper_MW=hi,net_lower_MW=lo-reference,net_upper_MW=hi-reference,
            risk_lower_MWh=low['upper_MWh'],risk_upper_MWh=high['lower_MWh'],root_evaluations=calls)
