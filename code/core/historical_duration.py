"""完整15分钟历史记录下的最短可行持续时长；沿用原线性约束。"""
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from common import CORES

QUARTER = .25  # 原始电表每15分钟计量；换算为小时。

class HistoricalDuration:
    def __init__(self,public,meter,count,power_bound,*,bias_bound,solver_tolerance,duration_check_tolerance):
        selected = public['call_ids']>=public['J']-count
        rows = public['required']<=count
        W = public['W'][rows][:,selected].tocsr()
        self.starts,self.stops,self.left = (public[k][selected] for k in ('starts','stops','left'))
        power_bound = np.asarray(power_bound,float)
        self.P = float(power_bound) if power_bound.ndim==0 else power_bound[selected]
        self.bias_bound = float(bias_bound)
        self.duration_check_tolerance = duration_check_tolerance
        self.solver_tolerance = solver_tolerance
        self.n = int(selected.sum())
        center = W@(public['baseline'][selected]-meter[selected]-public.get('mean_error_MW',0.))
        # 已知误差中心单独扣除；均值偏差范围与随机误差幅度不混用。
        bias = np.asarray(W.sum(axis=1)).ravel()
        self.A = sparse.vstack([sparse.hstack([-W,-bias[:,None]]),
                                sparse.hstack([W,bias[:,None]])],format='csr')
        self.b = np.r_[public['bounds'][rows]-center,public['bounds'][rows]+center]

    def cap(self,duration):
        # stops为ISO邀约结束，starts+duration为客户能力允许的结束；实际响应同时受两者限制。
        return self.P*np.clip((np.minimum(self.left+QUARTER,np.minimum(self.stops,self.starts+duration))
                            -np.maximum(self.left,self.starts))/QUARTER,0,1)

    def solve(self,duration):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options')
            r = linprog(np.zeros(self.n+1),A_ub=self.A,b_ub=self.b,
                bounds=list(zip(np.zeros(self.n),self.cap(duration)))+[(-self.bias_bound,self.bias_bound)],
                method='highs',options={'threads':CORES,'primal_feasibility_tolerance':self.solver_tolerance,
                                       'dual_feasibility_tolerance':self.solver_tolerance})
        if not r.success and r.status!=2:raise RuntimeError(r.message)
        return r

    def minimum(self,maximum,minimum=0.):
        # minimum来自已核定工程下限；原资料对照未给此项信息，默认下限为零。
        if self.solve(minimum).success:return float(minimum)
        if not self.solve(maximum).success:raise RuntimeError('True duration family excluded')
        knots=np.unique(np.clip(np.r_[minimum,self.left-self.starts,self.left+QUARTER-self.starts,maximum],minimum,maximum))
        lo,hi=0,len(knots)-1
        while hi-lo>1:
            mid=(lo+hi)//2
            if self.solve(float(knots[mid])).success:hi=mid
            else:lo=mid
        a,b=float(knots[lo]),float(knots[hi])
        ca,cb=self.cap(a),self.cap(b)
        slope=(cb-ca)/(b-a)
        old=sparse.hstack([self.A,sparse.csr_matrix((self.A.shape[0],1))])
        capacity=sparse.hstack([sparse.eye(self.n),sparse.csr_matrix((self.n,1)),
                               sparse.csr_matrix(-slope[:,None])])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options')
            r=linprog(np.r_[np.zeros(self.n+1),1.],A_ub=sparse.vstack([old,capacity]),
                b_ub=np.r_[self.b,ca-slope*a],bounds=list(zip(np.zeros(self.n),np.broadcast_to(self.P,(self.n,))))+
                    [(-self.bias_bound,self.bias_bound),(a,b)],
                method='highs',options={'threads':CORES,'primal_feasibility_tolerance':self.solver_tolerance,
                                       'dual_feasibility_tolerance':self.solver_tolerance})
        if not r.success:raise RuntimeError(r.message)
        duration=float(r.x[-1])
        assert self.solve(duration+self.duration_check_tolerance).success  # 边界核对容差从集中参数入口传入。
        return duration

