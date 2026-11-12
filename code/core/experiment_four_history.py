"""十五次及二十次资料共用的核验统计与联合校准。"""
from __future__ import annotations
import json
import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz,cholesky
from scipy.stats import norm
from common import write_json
from history_verification import joint_bounds


def verification_history(c,public,destination):
    held=set(c.winter_dates+c.summer_dates)
    dates=np.array([str(d['task']['date']) for d in public['designs']])
    retained=np.flatnonzero(~np.isin(dates,list(held)))
    ids=public['call_ids'];hour=np.floor(np.maximum(public['left']-public['starts'],0)+c.hour_bin_tolerance).astype(int)
    W=public['W'].tocsr();extra=[]
    keys={(tuple(W[i].indices),tuple(W[i].data)) for i in range(W.shape[0])}
    # 两边均使用逐点、逐次、按调用后小时合并、最近若干次合并这四类原统计。
    # 先固定两套资料所需统计的全集，一次校准；取得更多资料后保留已有约束。
    for count in range(1,len(retained)+1):
        mask=np.isin(ids,retained[-count:])
        masks=[mask,*[mask&(hour==h) for h in np.unique(hour[mask])]]
        for keep in masks:
            row=sparse.csr_matrix(keep.astype(float)/int(keep.sum()))
            key=(tuple(row.indices),tuple(row.data))
            if key not in keys:extra.append(row);keys.add(key)
    combined=sparse.vstack([W,*extra],format='csr')
    family=np.r_[public['error_families'],np.full(len(extra),'global')]
    F=public['error_operator'];n=public['n'];daily=c.load_daily_variance_share+(1-c.load_daily_variance_share)*toeplitz(c.load_rho15**np.arange(96)) # 每日96个15分钟。
    covariance=np.zeros((n,n))
    for day in np.unique(F.indices//96):
        part=F[:,day*96:(day+1)*96].tocsr();active=np.unique(part.nonzero()[0]);block=part[active].toarray()
        covariance[np.ix_(active,active)]+=block@daily@block.T
    op=np.asarray(combined@cholesky(covariance,lower=True))
    sd=np.sqrt(np.einsum('ij,ij->i',op,op))*public['error_scale_MW']
    critical=np.empty(len(family));failure=(1-c.verification_coverage)*c.error_budget_share
    for mask,share in ((family=='global',c.global_error_budget_share),(family!='global',1-c.global_error_budget_share)):
        critical[mask]=norm.ppf(1-failure*share/(2*int(mask.sum()))) # 原双侧失败预算分配。
    common=dict(public,W=combined,error_families=family,error_stat_SD_MW=sd,
        error_critical=critical,bounds=sd*critical,required=np.zeros(len(family),int))
    settings=dict(training_dates=dates[retained].tolist(),all_dates=dates.tolist(),statistics=combined.shape[0],
        calibration_draws=c.calibration_draws,calibration_seed=c.calibration_seed,
        coverage=c.verification_coverage,error_budget_share=c.error_budget_share,
        global_share=c.global_error_budget_share,rho15=c.load_rho15,daily_share=c.load_daily_variance_share,
        baseline_days=c.baseline_days,error_calibration_days=c.error_calibration_days,
        call_design=[dict(date=str(d['task']['date']),start_hour=float(d['task']['start_hour']),
            end_hour=float(d['task']['end_hour']),baseline_input_days=d['prior'].tolist()) for d in public['designs']])
    path=destination/'verification_rule.json'
    saved=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if saved and saved['settings']==settings:
        calibration=saved['calibration']
        multiplier=np.where(family=='global',calibration['calibration']['global']['joint_multiplier'],calibration['calibration']['other']['joint_multiplier'])
        common['bounds']=sd*multiplier
    else:
        common['bounds'],calibration=joint_bounds(c,common,None)
    selected=np.isin(ids,retained);rows=np.asarray(combined[:,~selected].getnnz(axis=1)).ravel()==0
    pooled=np.flatnonzero(rows&(family=='global'));used=set()
    for at in pooled:used.update(ids[combined[at].indices].tolist())
    if used!=set(retained):raise RuntimeError('跨调用统计没有使用全部保留记录')
    rule=dict(settings=settings,calibration=calibration,additional_statistics=len(extra),
        training_calls=len(retained),training_points=int(selected.sum()),
        training_statistics=int(rows.sum()),training_pooled_statistics=len(pooled),
        training_pooled_dates=dates[sorted(used)].tolist(),
        confidence_family='两套资料的固定统计全集共同校准；二十次资料保留十五次资料的全部约束。')
    write_json(path,rule)
    return common,rule
