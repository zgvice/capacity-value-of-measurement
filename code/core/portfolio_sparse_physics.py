"""完整年度的等价稀疏积分：共用背景缺口，仅逐路径计算受功率限制的部分。"""
from __future__ import annotations
import numpy as np
from numba import njit,prange
from portfolio_annual import PortfolioAnnual


@njit(cache=True,parallel=True)
def base_prefix(q,g,surplus):
    result=np.empty((len(g),g.shape[1]+1))
    for f in prange(len(g)):
        result[f,0]=0.;total=0.;correction=0.
        for h in range(g.shape[1]):
            y=max(q-surplus*g[f,h],0.)-correction
            update=total+y;correction=(update-total)-y;total=update;result[f,h+1]=total
    return result


@njit(cache=True)
def base_value(t,q,f,g,prefix,surplus):
    h=min(int(t),g.shape[1])
    if h==g.shape[1]:return prefix[f,h]
    return prefix[f,h]+(t-h)*max(q-surplus*g[f,h],0.)


@njit(cache=True,parallel=True)
def background_savings(q,end,g,prefix,profiles,call_ptr,calls,variants,surplus):
    result=np.zeros(len(call_ptr)-1)
    for k in prange(len(result)):
        f=profiles[k//variants];total=0.
        for c in range(call_ptr[k],call_ptr[k+1]):
            start=calls[c,0];stop=min(calls[c,2],start+end)
            if stop>start:total+=base_value(stop,q,f,g,prefix,surplus)-base_value(start,q,f,g,prefix,surplus)
        result[k]=total
    return result


@njit(cache=True,parallel=True)
def sparse_correction_prefix(q,g,power,ptr,hours,surplus):
    R=len(power);F=len(g);result=np.zeros((R,len(hours)+F))
    for rf in prange(R*F):
        r,f=rf//F,rf%F;offset=ptr[f]+f;total=0.;error=0.
        for i in range(ptr[f],ptr[f+1]):
            h=hours[i];raw=max(q-surplus*g[f,h],0.);value=0.
            for quarter in range(4):value+=.25*max(raw-power[r,4*h+quarter],0.)  # 每小时四个15分钟点。
            y=value-error;update=total+y;error=(update-total)-y;total=update
            result[r,offset+i-ptr[f]+1]=total
    return result


@njit(cache=True)
def correction_value(t,q,r,f,g,power,ptr,hours,prefix,surplus):
    h=min(int(t),g.shape[1]);lo,hi=ptr[f],ptr[f+1]
    at=np.searchsorted(hours[lo:hi],h);result=prefix[r,lo+f+at]
    if h==g.shape[1] or at==hi-lo or hours[lo+at]!=h:return result
    raw=max(q-surplus*g[f,h],0.)
    for quarter in range(4):
        length=max(0.,min(t,h+(quarter+1)/4.)-(h+quarter/4.))
        result+=length*max(raw-power[r,4*h+quarter],0.)
    return result


@njit(cache=True,parallel=True)
def limited_intervals(profiles,call_ptr,calls,variants,a,b,ptr,hours):
    counts=np.zeros(len(call_ptr)-1,np.int64)
    for k in prange(len(counts)):
        f=profiles[k//variants];hh=hours[ptr[f]:ptr[f+1]]
        for c in range(call_ptr[k],call_ptr[k+1]):
            left=calls[c,0]+a;right=min(calls[c,2],calls[c,0]+b)
            if right>left and np.searchsorted(hh,np.floor(left))<np.searchsorted(hh,np.ceil(right)):counts[k]+=1
    starts=np.empty(len(counts)+1,np.int64);starts[0]=0;starts[1:]=np.cumsum(counts);windows=np.empty((starts[-1],2))
    for k in prange(len(counts)):
        f=profiles[k//variants];hh=hours[ptr[f]:ptr[f+1]];at=starts[k]
        for c in range(call_ptr[k],call_ptr[k+1]):
            left=calls[c,0]+a;right=min(calls[c,2],calls[c,0]+b)
            if right>left and np.searchsorted(hh,np.floor(left))<np.searchsorted(hh,np.ceil(right)):
                windows[at,0]=left;windows[at,1]=right;at+=1
    return starts,windows


@njit(cache=True,parallel=True)
def background_correction(q,g,power,ptr,hours,prefix,profiles,window_ptr,windows,variants,surplus):
    R=len(power);K=len(window_ptr)-1;result=np.zeros((R,K))
    for rk in prange(R*K):
        r,k=rk//K,rk%K;f=profiles[k//variants];total=0.
        for i in range(window_ptr[k],window_ptr[k+1]):
            total+=correction_value(windows[i,1],q,r,f,g,power,ptr,hours,prefix,surplus)
            total-=correction_value(windows[i,0],q,r,f,g,power,ptr,hours,prefix,surplus)
        result[r,k]=total
    return result


@njit(cache=True)
def event_rows(k,profiles,ptr,hours,call_ptr,calls,variants,seg):
    j=k//variants;first,last=call_ptr[k],call_ptr[k+1];current=first;count=0
    for ix in range(ptr[j],ptr[j+1]):
        h=hours[ix]
        while current<last and calls[current,2]<=h:current+=1
        c=current
        while c<last and calls[c,0]<h+1:
            for si in range(len(seg)):
                left=max(float(h),calls[c,0]+seg[si,0]);right=min(float(h+1),calls[c,2],calls[c,0]+seg[si,1])
                if right>left:
                    for quarter in range(4):
                        length=max(0.,min(right,h+(quarter+1)/4.)-max(left,h+quarter/4.))
                        if length>0:count+=1
            c+=1
    return count


@njit(cache=True,parallel=True)
def event_geometry(g,profiles,ptr,hours,values,call_ptr,calls,variants,seg):
    K=len(call_ptr)-1;counts=np.empty(K,np.int64)
    for k in prange(K):counts[k]=event_rows(k,profiles,ptr,hours,call_ptr,calls,variants,seg)
    starts=np.empty(K+1,np.int64);starts[0]=0;starts[1:]=np.cumsum(counts);N=starts[-1]
    quarter_ids=np.empty(N,np.int64);segment_ids=np.empty(N,np.int64);lengths=np.empty(N);actual=np.empty(N);background=np.empty(N)
    for k in prange(K):
        j=k//variants;f=profiles[j];first,last=call_ptr[k],call_ptr[k+1];current=first;at=starts[k]
        for ix in range(ptr[j],ptr[j+1]):
            h=hours[ix]
            while current<last and calls[current,2]<=h:current+=1
            c=current
            while c<last and calls[c,0]<h+1:
                for si in range(len(seg)):
                    left=max(float(h),calls[c,0]+seg[si,0]);right=min(float(h+1),calls[c,2],calls[c,0]+seg[si,1])
                    if right>left:
                        for quarter in range(4):
                            length=max(0.,min(right,h+(quarter+1)/4.)-max(left,h+quarter/4.))
                            if length>0:
                                quarter_ids[at]=4*h+quarter;segment_ids[at]=si;lengths[at]=length
                                actual[at]=values[ix];background[at]=g[f,h];at+=1
                c+=1
    return starts,quarter_ids,segment_ids,lengths,actual,background


@njit(cache=True,parallel=True)
def event_adjustment(q,power,ptr,quarters,segments,lengths,actual,background,surplus):
    R=power.shape[1];K=len(ptr)-1;result=np.zeros((R,K))
    for rk in prange(R*K):
        r,k=rk//K,rk%K;total=0.
        for at in range(ptr[k],ptr[k+1]):
            p=power[segments[at],r,quarters[at]];bg=max(q-surplus*background[at],0.);raw=max(q+actual[at],0.)
            total+=lengths[at]*(min(raw,p)-min(bg,p))
        result[r,k]=total
    return result


class SparsePortfolioAnnual(PortfolioAnnual):
    def __init__(self,c,choice,powers,search_upper):
        super().__init__(c,choice,powers);self.search_upper=search_upper;self.sparse={};self.geometry={};self.base_q=None

    def clear_geometry(self):self.geometry={}

    def power_structure(self,key):
        if key not in self.sparse:
            p=self.profiles[key];floor=p.reshape(self.R,8760,4).min(axis=(0,2))  # 原年度8760小时及15分钟网格。
            bad=self.search_upper-self.e.surplus*self.e.g>floor[None,:]
            counts=bad.sum(axis=1);ptr=np.r_[0,np.cumsum(counts)].astype(np.int64);hours=np.nonzero(bad)[1].astype(np.int64)
            self.sparse[key]=dict(ptr=ptr,hours=hours,q=None,prefix=None)
        return self.sparse[key]

    def geometry_for(self,segments):
        token=tuple(segments)
        if token not in self.geometry:
            seg=np.array([[a,b] for a,b,key in segments]);geometry=[]
            for plan in self.choice.plans:
                events=event_geometry(self.e.g,self.e.profiles,self.e.ptr,self.e.hours,self.e.values,
                    plan['ptr'],plan['calls'],self.e.variants,seg)
                windows=[]
                for a,b,key in segments:
                    sparse=self.power_structure(key)
                    if len(sparse['hours']):
                        windows.append(limited_intervals(self.e.profiles,plan['ptr'],plan['calls'],self.e.variants,a,b,sparse['ptr'],sparse['hours']))
                    else:windows.append(None)
                geometry.append(dict(events=events,windows=windows))
            self.geometry[token]=geometry
        return self.geometry[token]

    def losses(self,q,segments):
        self.prepare(q)
        if not segments:return np.maximum(self.total[None,:]-np.repeat(self.e.s0,self.e.variants)[None,:],0.)*np.ones((self.R,1))
        if q>self.search_upper:
            self.search_upper=q;self.sparse={};self.geometry={}  # 搜索扩大时重建稀疏索引；不施加额外物理容量上限。
        if self.base_q!=q:
            self.base=base_prefix(q,self.e.g,self.e.surplus);self.base_q=q
        geometry=self.geometry_for(segments);powers=np.array([self.profiles[key] for a,b,key in segments]);best=None
        for plan,geo in zip(self.choice.plans,geometry):
            shared=background_savings(q,segments[-1][1],self.e.g,self.base,self.e.profiles,plan['ptr'],plan['calls'],self.e.variants,self.e.surplus)
            saved=np.broadcast_to(shared,(self.R,len(shared))).copy()
            for (a,b,key),windows in zip(segments,geo['windows']):
                if windows is None:continue
                sparse=self.power_structure(key)
                if sparse['q']!=q:
                    sparse['prefix']=sparse_correction_prefix(q,self.e.g,self.profiles[key],sparse['ptr'],sparse['hours'],self.e.surplus);sparse['q']=q
                saved-=background_correction(q,self.e.g,self.profiles[key],sparse['ptr'],sparse['hours'],sparse['prefix'],self.e.profiles,*windows,self.e.variants,self.e.surplus)
            saved+=event_adjustment(q,powers,*geo['events'],self.e.surplus)
            loss=np.maximum(self.total[None,:]-saved-np.repeat(self.e.s0,self.e.variants)[None,:],0.)
            if best is None:best=loss
            else:np.minimum(best,loss,out=best)
        return best
