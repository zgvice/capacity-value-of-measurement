"""Shared paths and exact risk helpers for the delivered experiment tree."""
from __future__ import annotations
from common_settings import TAIL_LEVEL
import os
CORES = int(os.environ.get('NUMBA_NUM_THREADS', os.cpu_count() or 1))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')  # process workers, not nested BLAS pools
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'inputs'
OUT = ROOT/'results'
AUDIT = ROOT/'checks'
for p in (OUT,AUDIT): p.mkdir(exist_ok=True)

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')

def read_json(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()

def cvar(loss, weights, alpha=TAIL_LEVEL):
    x=np.asarray(loss).ravel(); w=np.broadcast_to(weights,np.shape(loss)).ravel()
    idx=np.argsort(-x,kind='stable'); u=w[idx]; mass=1-alpha
    take=np.minimum(u,np.maximum(mass-np.r_[0,np.cumsum(u)[:-1]],0))
    return float(take@x[idx]/mass)

def monotone_root(fun, target, lo=0., hi=2500., tol=.05):
    if fun(lo)>target+1e-7: raise ValueError('Risk at zero exceeds budget')
    while fun(hi)<=target:
        hi*=2
        if hi>1e6: raise ValueError('Unbounded bracket')
    calls=0
    while hi-lo>tol:
        mid=(lo+hi)/2; val=fun(mid); calls+=1
        if val<=target:lo=mid
        else:hi=mid
    return (lo if target==0 else (lo+hi)/2), {'lower_MW':lo,'upper_MW':hi,'calls':calls}

def season(month):
    return np.select([np.isin(month,[12,1,2]),np.isin(month,[3,4,5]),np.isin(month,[6,7,8])],[0,1,2],default=3).astype(np.int8)

def hour_window(hour, center, duration):
    # Fraction of a source one-hour interval inside a fixed daily window.
    total=np.zeros(np.broadcast_shapes(np.shape(hour),np.shape(center)),float)
    for shift in (-24,0,24):
        left=np.asarray(center)-duration/2+shift; right=left+duration
        total+=np.maximum(0,np.minimum(np.asarray(hour)+1,right)-np.maximum(hour,left))
    return np.minimum(total,1.)

def complete_calendar_windows(hourly_dates,centers,duration):
    """Join actual dated windows, including season/daytype changes at midnight."""
    T=len(hourly_dates);start=hourly_dates[0].normalize();days=(T+23)//24
    daily=pd.date_range(start-pd.Timedelta(days=1),periods=days+2,freq='D')
    ss=season(daily.month);kind=(daily.dayofweek>=5).astype(int);mid=centers[ss,kind]
    avail=np.zeros(T);anchor=np.full(T,-1,np.int32);dest=np.full(days+2,-1,np.int32)
    for k,c in enumerate(mid):
        absolute=(k-1)*24+c;left=absolute-duration/2;right=absolute+duration/2
        dest[k]=int(np.ceil(right))
        for h in range(max(0,int(np.floor(left))),min(T,int(np.ceil(right)))):
            part=max(0,min(h+1,right)-max(h,left))
            if part>0:
                if avail[h]>0:raise AssertionError('Overlapping separate production windows')
                avail[h]=part;anchor[h]=k
    return avail,anchor,dest

def log(msg): print(time.strftime('%Y-%m-%d %H:%M:%S'),msg,flush=True)
