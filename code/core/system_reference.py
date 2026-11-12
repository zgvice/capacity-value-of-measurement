"""原系统全年缺供积分；保持原排序和补偿求和算法。"""
import numpy as np
from numba import njit,prange

@njit(parallel=True,cache=True)
def _sort_zero_background(g,surplus):
    F,T=g.shape;ordered=np.empty((F,T));prefix=np.empty_like(ordered)
    for f in prange(F):
        values=np.sort(surplus*g[f]);ordered[f]=values
        total=0.;compensation=0.
        for t in range(T):
            y=values[t]-compensation;updated=total+y
            compensation=(updated-total)-y;total=updated;prefix[f,t]=total
    return ordered,prefix


@njit(parallel=True,cache=True)
def _zero_loss(q,ordered,prefix,g,profiles,ptr,hours,values,s0,surplus):
    F=g.shape[0];base=np.empty(F)
    for f in prange(F):
        k=np.searchsorted(ordered[f],q)
        base[f]=k*q-(prefix[f,k-1] if k>0 else 0.)
    J=len(profiles);total=np.empty(J);loss=np.empty(J)
    for j in prange(J):
        f=profiles[j];v=base[f]
        for ix in range(ptr[j],ptr[j+1]):
            t=hours[ix]
            v+=max(values[ix]+q,0.)-max(q-surplus*g[f,t],0.)
        total[j]=max(v,0.);loss[j]=max(v-s0[j],0.)
    return loss,total

