"""组合历史核验：逐户持续能力与同一组基线误差约束同时求解。"""
from __future__ import annotations
import json,math,warnings
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.linalg import cholesky
from scipy.stats import truncnorm
from threadpoolctl import threadpool_limits

from common import DATA,CORES
from load_covariance import load_covariance,block_covariance


CACHE=DATA/'frozen/historical_error_max.npz'


def calibration_maxima(c,public):
    settings={key:getattr(c,key) for key in ('baseline_days','load_rho15','load_daily_variance_share','calibration_draws','calibration_seed','calibration_batch')}
    settings.update(points=public['n'],statistics=public['W'].shape[0])
    if CACHE.exists():
        with np.load(CACHE) as z:
            if json.loads(str(z['settings']))==settings:return dict(global_=z['global_'].copy(),other=z['other'].copy())
    covariance=block_covariance(public['error_operator'],load_covariance(c))
    factor=cholesky(covariance,lower=True);operator=np.asarray(public['W']@factor)
    sd=np.sqrt(np.einsum('ij,ij->i',operator,operator));operator/=sd[:,None]
    masks=dict(global_=public['error_families']=='global',other=public['error_families']!='global')
    maxima={name:np.empty(c.calibration_draws) for name in masks}
    rng=np.random.default_rng(c.calibration_seed)
    with threadpool_limits(CORES,user_api='blas'):
        for start in range(0,c.calibration_draws,c.calibration_batch):
            size=min(c.calibration_batch,c.calibration_draws-start)
            sample=abs(rng.standard_normal((size,public['n']))@operator.T)
            for name,mask in masks.items():maxima[name][start:start+size]=sample[:,mask].max(axis=1)
            if start==0 or (start+c.calibration_batch)%(16*c.calibration_batch)==0:
                print(f'历史联合边界：{start+size}/{c.calibration_draws}',flush=True)  # 沿用每16批输出一次进度。
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    for name in maxima:maxima[name].sort()
    np.savez_compressed(CACHE,settings=np.asarray(json.dumps(settings)),**maxima)
    return maxima


def multipliers(c,public,M,maxima):
    # M户边界及聚合边界同时覆盖；单户的聚合边界与其自身相同，计一次。
    views=1 if M==1 else M+1
    failure=(1-c.verification_coverage)*c.error_budget_share/views
    critical=np.empty(public['W'].shape[0]);details={}
    for name,share in [('global_',c.global_error_budget_share),('other',1-c.global_error_budget_share)]:
        delta=failure*share;rank=math.ceil((c.calibration_draws+1)*(1-delta))
        value=float(maxima[name][rank-1]);mask=public['error_families']=='global'
        if name=='other':mask=~mask
        critical[mask]=value;details[name]=dict(multiplier=value,failure_budget=delta,rank=rank)
    return critical,details


def history_power_bounds(c,profiles,members,weights,n):
    failure=(1-c.verification_coverage)*(1-c.error_budget_share)
    result=[]
    for name,w in zip(members,weights):
        profile=profiles[name]
        if profile['kind']=='normal':
            value=c.response_mean_ratio if c.response_sd_ratio==0 else truncnorm.ppf(1-failure/(len(members)*n),
                -c.response_mean_ratio/c.response_sd_ratio,np.inf,loc=c.response_mean_ratio,scale=c.response_sd_ratio)
        else:value=float(profile['values'].max())  # 已声明的实测经验分布，其支持上界直接来自该户全部原记录。
        result.append(c.nomination_mw*w*value)
    return np.asarray(result)


class PortfolioHistory:
    def __init__(self,c,public,baseline,meter,weights,power_bounds,count,critical,gamma,threads=CORES):
        selected=public['call_ids']>=public['J']-count;rows=public['required']<=count
        W=public['W'][rows][:,selected].tocsr();M=len(weights);n=int(selected.sum());s=W.shape[0]
        self.c=c;self.M=M;self.n=n;self.P=np.asarray(power_bounds);self.weights=np.asarray(weights);self.threads=threads
        self.left,self.starts,self.stops=(public[k][selected] for k in ('left','starts','stops'))
        self.maximum=max(float(np.max(self.stops-self.starts)),c.verified_min_duration_hours)
        centers=np.array([W@(b[selected]-y[selected]-c.error_mean_rate*c.load_mw*w) for b,y,w in zip(baseline,meter,weights)])
        unit=sparse.hstack([sparse.block_diag([W]*M,format='csr'),sparse.kron(sparse.eye(M),sparse.csr_matrix(np.asarray(W.sum(axis=1))),format='csr')],format='csr')
        stochastic=public['error_stat_SD_MW'][rows]*critical[rows]
        bound=(self.weights[:,None]*stochastic).ravel();center=centers.ravel()
        A=[unit,-unit];rhs=[center+bound,-center+bound]
        if M>1:
            aggregate=sparse.hstack([sparse.hstack([W]*M),sparse.csr_matrix(np.asarray(W.sum(axis=1)).reshape(-1,1)*np.ones((1,M)))],format='csr')
            bound=stochastic*np.sqrt(gamma+(1-gamma)*float(self.weights@self.weights));center=centers.sum(axis=0)
            A.extend([aggregate,-aggregate]);rhs.extend([center+bound,-center+bound])
        self.A=sparse.vstack(A,format='csr');self.b=np.concatenate(rhs)
        self.bias=c.load_mw*c.bias_bound_rate*self.weights
        self.lower=np.r_[np.zeros(M*n),-self.bias];self.upper=np.r_[np.repeat(self.P,n),self.bias]
        self.check_count=0;self.checked={}

    def cap(self,H):
        H=np.asarray(H)
        available=np.clip((np.minimum(self.left[None,:]+.25,np.minimum(self.stops[None,:],self.starts[None,:]+H[:,None]))
            -np.maximum(self.left[None,:],self.starts[None,:]))/.25,0,1)  # 沿用15分钟交叠比例。
        return (self.P[:,None]*available).ravel()

    def lp(self,objective,A,b,lower,upper):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options')
            r=linprog(objective,A_ub=A,b_ub=b,bounds=np.c_[lower,upper],method='highs',
                options={'threads':self.threads,'primal_feasibility_tolerance':self.c.solver_tolerance,'dual_feasibility_tolerance':self.c.solver_tolerance})
        if not r.success and r.status!=2:raise RuntimeError(r.message)
        return r

    def solve(self,H):
        if np.any(np.asarray(H)<self.c.verified_min_duration_hours):return SimpleNamespace(success=False)
        key=tuple(np.asarray(H,float))
        if key in self.checked:return SimpleNamespace(success=self.checked[key])
        self.check_count+=1;upper=self.upper.copy();upper[:self.M*self.n]=self.cap(H)
        result=self.lp(np.zeros(len(upper)),self.A,self.b,self.lower,upper)
        self.checked[key]=result.success
        return result

    def minimum_shared(self,fixed,variable,minimum=None,maximum=None):
        variable=np.asarray(variable,int);fixed=np.asarray(fixed,float).copy()
        minimum=self.c.verified_min_duration_hours if minimum is None else max(minimum,self.c.verified_min_duration_hours)  # 原参数中的已核定工程下限。
        maximum=self.maximum if maximum is None else maximum
        def durations(t):
            H=fixed.copy();H[variable]=t;return H
        if self.solve(durations(minimum)).success:return dict(lower=minimum,upper=minimum,H=durations(minimum).tolist())
        if not self.solve(durations(maximum)).success:return None
        knots=np.unique(np.clip(np.r_[minimum,self.left-self.starts,self.left+.25-self.starts,maximum],minimum,maximum))
        lo=0;hi=len(knots)-1
        while hi-lo>1:
            mid=(lo+hi)//2
            if self.solve(durations(float(knots[mid]))).success:hi=mid
            else:lo=mid
        a,b=float(knots[lo]),float(knots[hi]);ca=self.cap(durations(a));cb=self.cap(durations(b));slope=(cb-ca)/(b-a)
        A=sparse.vstack([sparse.hstack([self.A,sparse.csr_matrix((self.A.shape[0],1))]),
            sparse.hstack([sparse.eye(self.M*self.n),sparse.csr_matrix((self.M*self.n,self.M)),sparse.csr_matrix(-slope[:,None])])],format='csr')
        rhs=np.r_[self.b,ca-slope*a];lower=np.r_[self.lower,a];upper=np.r_[self.upper,b];objective=np.r_[np.zeros(len(self.lower)),1.]
        r=self.lp(objective,A,rhs,lower,upper)
        if not r.success:raise RuntimeError(r.message)
        dual=np.minimum(np.asarray(r.ineqlin.marginals),0.);reduced=objective-A.T@dual
        bound=float(dual@rhs+np.maximum(reduced,0)@lower+np.minimum(reduced,0)@upper)
        t=min(maximum,float(r.x[-1])+self.c.duration_check_tolerance_hours)
        H=durations(t);witness=self.solve(H)
        if not witness.success:raise RuntimeError('持续能力边界未满足完整历史约束')
        return dict(lower=max(minimum,bound-self.c.duration_check_tolerance_hours),upper=t,H=H.tolist())

    def envelope(self):
        full=np.full(self.M,self.maximum)
        if not self.solve(full).success:raise RuntimeError('已知历史窗口内没有相容物理解释')
        thresholds={};witnesses=[]
        for subset in range(1<<self.M):
            inactive=[i for i in range(self.M) if not subset&(1<<i)]
            value=self.minimum_shared(full,inactive) if inactive else dict(lower=0.,upper=0.,H=full.tolist())
            if value is None:continue
            thresholds[subset]=value['lower'];witnesses.append(value['H'])
        # 在已有可行状态上逐户缩短，得到联合可行的边界候选；每次都用完整记录核验。
        candidates=[]
        for H in np.unique(np.asarray(witnesses),axis=0):
            for order in [list(range(self.M)),list(range(self.M-1,-1,-1))]:
                current=np.asarray(H,float)
                for i in order:
                    value=self.minimum_shared(current,[i],maximum=current[i])
                    if value is not None:current=np.asarray(value['H'])
                candidates.append(current)
        candidates=np.unique(np.round(candidates,9),axis=0)
        return thresholds,candidates
