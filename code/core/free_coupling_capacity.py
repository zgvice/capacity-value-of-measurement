"""Worst annual tail over every coupling of fixed physical marginals.

Each row is one resource path with mass 1/R. Each column retains the original
official system probability. The normalized tail mass sums to one and has the
two marginal caps implied by 1-alpha. Remaining probabilities are completed by
their product. No baseline row or resource/system independence is introduced.
"""
from __future__ import annotations
from common import *
from common_settings import TAIL_LEVEL
from scipy.optimize._highspy import _core as hs
import math
from numba import njit,prange


@njit(cache=True,parallel=True)
def tail_argmax(loss,penalty):
    """逐系统场景取原目标的最大值，免去反复生成R×J中间矩阵。"""
    R,J=loss.shape;chosen=np.empty(J,np.int64);envelope=np.empty(J)
    for j in prange(J):
        index=0;value=loss[0,j]-penalty[0]
        for r in range(1,R):
            candidate=loss[r,j]-penalty[r]
            if candidate>value:index=r;value=candidate
        chosen[j]=index;envelope[j]=value
    return chosen,envelope


@njit(cache=True,parallel=True)
def minimum_reduced_cost(cost,u,v,eta,arithmetic_bound):
    """原对偶约化成本及其五次浮点运算误差界，按资源行并行计算。"""
    R,J=cost.shape;minimum=np.empty(R)
    for r in prange(R):
        value=np.inf
        for j in range(J):
            rc=cost[r,j]-(u[j]+v[r]+eta)
            error=arithmetic_bound*(abs(cost[r,j])+abs(u[j])+abs(v[r])+abs(eta))
            value=min(value,rc-error)
        minimum[r]=value
    return minimum.min()


def _highs(A,cost,lower,upper,threads,col_upper=None):
    A=sparse.csc_matrix(A);n=A.shape[1]
    lp=hs.HighsLp();lp.num_col_=n;lp.num_row_=A.shape[0]
    lp.a_matrix_.num_col_=n;lp.a_matrix_.num_row_=A.shape[0]
    lp.a_matrix_.format_=hs.MatrixFormat.kColwise
    lp.a_matrix_.start_=A.indptr.astype(np.int32)
    lp.a_matrix_.index_=A.indices.astype(np.int32);lp.a_matrix_.value_=A.data
    lp.col_cost_=np.asarray(cost,float);lp.col_lower_=np.zeros(n)
    lp.col_upper_=np.full(n,np.inf) if col_upper is None else np.asarray(col_upper,float)
    lp.row_lower_=np.asarray(lower,float);lp.row_upper_=np.asarray(upper,float)
    h=hs._Highs();opt=hs.HighsOptions();opt.output_flag=False
    opt.threads=int(threads);opt.parallel='on' if threads>1 else 'off'
    # 1e-9沿用原生稀疏线性规划接口的数值可行性与对偶容差。
    opt.solver='simplex';opt.primal_feasibility_tolerance=1e-9;opt.dual_feasibility_tolerance=1e-9
    h.passOptions(opt);h.passModel(lp)
    return h


def _optimal(h,label):
    h.run();status=h.getModelStatus()
    if status!=hs.HighsModelStatus.kOptimal:
        raise RuntimeError(label+': '+h.modelStatusToString(status))
    return h.getSolution()



def _gamma(n):
    a=n*np.finfo(float).eps
    if a>=1:raise ValueError('Floating-point accumulation bound unavailable')
    return a/(1-a)


class TailTransport:
    def __init__(self,pi,R,alpha=TAIL_LEVEL,threads=CORES):
        self.pi=np.array(pi,dtype=float,copy=True)
        self.R=int(R);self.alpha=float(alpha)
        if self.R<1 or self.R!=R or not 0<self.alpha<1:
            raise ValueError('R must be positive and alpha in(0,1)')
        if self.pi.ndim!=1 or not self.pi.size or not np.isfinite(self.pi).all() or np.any(self.pi<0):
            raise ValueError('System probabilities must be a nonnegative finite vector')
        if abs(float(self.pi.sum())-1)>1e-12:
            raise ValueError('System probabilities must sum to one')
        self.J=len(self.pi);self.c=1-self.alpha
        self.resource_weights=np.full(self.R,1/self.R)
        self.system_cap=self.pi/self.c;self.resource_cap=self.resource_weights/self.c
        self.indices=np.arange(self.R*self.J,dtype=np.int32)
        system=sparse.kron(np.ones((1,self.R)),sparse.eye(self.J),format='csr')
        resource=sparse.kron(sparse.eye(self.R),np.ones((1,self.J)),format='csr')
        mass=sparse.csr_matrix(np.ones((1,len(self.indices))))
        A=sparse.vstack([system,resource,mass],format='csc')
        lo=np.r_[np.full(self.J+self.R,-np.inf),1.]
        up=np.r_[self.system_cap,self.resource_cap,1.]
        self.h=_highs(A,np.zeros(len(self.indices)),lo,up,int(threads))

    def _repair_tail(self,tail):
        z=np.maximum(np.asarray(tail,float),0.).reshape(self.R,self.J)
        col=z.sum(axis=0)
        z*=np.minimum(1.,self.system_cap/np.maximum(col,1e-300))[None,:]
        row=z.sum(axis=1)
        z*=np.minimum(1.,self.resource_cap/np.maximum(row,1e-300))[:,None]
        mass=float(z.sum())
        if mass>1:
            z/=mass
        elif mass<1:
            remain_r=np.maximum(self.resource_cap-z.sum(axis=1),0.)
            remain_j=np.maximum(self.system_cap-z.sum(axis=0),0.)
            z+=(1-mass)*np.outer(remain_r/remain_r.sum(),remain_j/remain_j.sum())
        return z

    def _margin_residual(self,z):
        row=z.sum(axis=1);col=z.sum(axis=0)
        tail_r=self.c*row;tail_j=self.c*col
        remain_r=self.resource_weights-tail_r;remain_j=self.pi-tail_j
        rr=np.maximum(remain_r,0.);cc=np.maximum(remain_j,0.)
        lifted_r=tail_r+rr*cc.sum()/self.alpha
        lifted_j=tail_j+cc*rr.sum()/self.alpha
        return max(float(np.max(row-self.resource_cap)),float(np.max(col-self.system_cap)),
            abs(float(z.sum())-1),float(-z.min()),float(-remain_r.min()),float(-remain_j.min()),
            float(np.max(abs(lifted_r-self.resource_weights))),float(np.max(abs(lifted_j-self.pi))),0.)

    def _dual_upper(self,cost,dual,scale):
        u=np.minimum(dual[:self.J],0.)
        v=np.minimum(dual[self.J:self.J+self.R],0.)
        eta=float(dual[-1])
        # With sum(z)=1, the smallest reduced cost corrects the dual bound.
        correction=min(float(minimum_reduced_cost(cost,u,v,eta,_gamma(5))),0.)
        terms=np.r_[self.system_cap*u,self.resource_cap*v,eta,correction]
        roundoff=_gamma(5)*float(np.sum(abs(terms)))
        return -(math.fsum(terms.tolist())-roundoff)*scale

    def _primal_lower(self,loss,z):
        terms=loss*z
        value=float(np.sum(terms,dtype=np.float64))
        row=z.sum(axis=1);col=z.sum(axis=0);mass=float(z.sum())
        # Any remaining arithmetic violations can be removed by scaling down
        # rows/columns and then filling missing mass with nonnegative weights.
        # At worst each removed unit costs max(loss); thus this is a lower bound.
        removed=float(np.maximum(row*(1+_gamma(self.J))-self.resource_cap,0.).sum())
        removed+=float(np.maximum(col*(1+_gamma(self.R))-self.system_cap,0.).sum())
        removed+=max(mass*(1+_gamma(z.size))-1,0.)
        # 非负加权损失：按全部项数的经典gamma_n界覆盖数组求和误差，保留原乘法误差留量。
        reduction=_gamma(terms.size)
        roundoff=(_gamma(5)+reduction)*value/(1-reduction)
        return max(0.,value-float(loss.max())*removed-roundoff)

    def risk(self,loss):
        """Return certified numerical bounds and a normalized tail witness."""
        loss=np.asarray(loss,float)
        if loss.shape!=(self.R,self.J) or not np.isfinite(loss).all() or np.any(loss<0):
            raise ValueError('Loss must be nonnegative finite R×J; omit the zero-reference row')
        scale=max(1.,float(loss.max())/100.)
        cost=-loss/scale
        self.h.changeColsCost(len(self.indices),self.indices,cost.ravel())
        sol=_optimal(self.h,'Free-coupling annual tail transport')
        z=self._repair_tail(sol.col_value)
        residual=self._margin_residual(z)
        lower=self._primal_lower(loss,z)
        upper=min(float(loss.max()),self._dual_upper(cost,np.asarray(sol.row_dual),scale))
        if residual>2e-9 or upper<lower-max(1e-8,abs(upper)*1e-12):
            raise AssertionError(dict(lower_MWh=lower,upper_MWh=upper,margin_residual=residual))
        return dict(lower_MWh=lower,upper_MWh=max(0.,upper,lower),tail=z,margin_residual=residual)


def self_check():
    from scipy.optimize import linprog
    import warnings
    rng=np.random.default_rng(2026110503)
    pi=np.array([.10,.23,.08,.31,.28]);alpha=.55
    x=np.array([1.,7.,3.,16.,9.]);checks=[]
    for R in [1,4]:
        loss=np.tile(x,(R,1));solver=TailTransport(pi,R,alpha)
        result=solver.risk(loss);reference=cvar(x,pi,alpha)
        checks.append(dict(case='identical rows',R=R,reference=reference,
            lower=result['lower_MWh'],upper=result['upper_MWh'],residual=result['margin_residual']))
    R=3;J=7;pi=rng.dirichlet(np.ones(J));alpha=.55
    solver=TailTransport(pi,R,alpha)
    A=np.r_[np.tile(np.eye(J),(1,R)),np.kron(np.eye(R),np.ones((1,J)))]
    b=np.r_[pi,np.full(R,1/R)]/(1-alpha)
    active_resource_cap=False
    for k in range(4):
        loss=rng.uniform(0,200,(R,J))
        if k==0:loss[0]+=300.  # make the resource marginal cap genuinely bind
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected')
            reference=linprog(-loss.ravel(),A_ub=A,b_ub=b,A_eq=np.ones((1,R*J)),b_eq=[1.],
                              bounds=(0,None),method='highs',options={'threads':22})
        if not reference.success:raise RuntimeError(reference.message)
        result=solver.risk(loss)
        active_resource_cap|=bool(np.any(abs(result['tail'].sum(axis=1)-solver.resource_cap)<1e-9))
        checks.append(dict(case='independent monolithic LP',R=R,reference=float(-reference.fun),
            lower=result['lower_MWh'],upper=result['upper_MWh'],residual=result['margin_residual']))
    for row in checks:
        if not row['lower']-1e-9<=row['reference']<=row['upper']+1e-9:
            raise AssertionError(row)
        if row['upper']-row['lower']>1e-7:raise AssertionError(row)
    if not active_resource_cap:raise AssertionError('Small LP did not exercise resource marginal cap')
    return dict(cases=len(checks),resource_marginal_cap_exercised=True,max_bound_width_MWh=max(x['upper']-x['lower'] for x in checks),
                max_margin_residual=max(x['residual'] for x in checks),checks=checks)


if __name__=='__main__':
    print(json.dumps(self_check(),ensure_ascii=False,indent=2))
