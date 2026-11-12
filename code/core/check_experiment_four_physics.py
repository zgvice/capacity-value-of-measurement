"""以完整年度线性规划独立核对额外用电积分。"""
import json
import numpy as np
from scipy import sparse
import highspy
from common import write_json,CORES
from run_experiment_four import inputs,future_power,DEST,say
from portfolio_annual import frozen_choice
from portfolio_sparse_physics import base_prefix
from experiment_four_physics import returns_geometry,deficit_prefix,annual_payback


def main():
    c,public,bound,maximum,data,model,calibration,old=inputs();choice=frozen_choice(c,calibration['invitation_comparison']);e=choice.engine
    power=future_power(c,model,'正态1')[:1]
    j=int(np.argmax(e.s0));k=j*e.variants;plan=choice.plans[-1]
    calls=plan['calls'][plan['ptr'][k]:plan['ptr'][k+1]].copy();cp=np.array([0,len(calls)])
    ret=returns_geometry(cp,calls,c.payback_hours)
    start,stop=e.ptr[j:j+2];hours=e.hours[start:stop];values=e.values[start:stop];ptr=np.array([0,len(hours)])
    f=e.profiles[j];g=e.g;profile=np.array([f]);q=old['groups']['正态1户']['full_information']['total_lower_MW']
    base=base_prefix(q,g,e.surplus);prefix=deficit_prefix(q,g,power,profile,ptr,hours,values,e.surplus)
    gap=-e.surplus*g[f].copy();gap[hours]=values
    steps=8760*12;dt=1/12  # 完整年，5分钟对齐原邀约边界；原15分钟功率每三个点相同。
    ids=[];which=[]
    for at,(left,_,right) in enumerate(calls):
        if ret[at,-1]<0:continue
        right=min(right,left+c.true_duration_hours)
        indexes=np.arange(int(round(left*12)),int(round(right*12)))
        ids.extend(indexes);which.extend([at]*len(indexes))
    ids=np.asarray(ids,int);which=np.asarray(which,int);nd=len(ids);nc=len(calls)
    # d为每个有效5分钟的功率，E为每次调节电量，u为全部年度5分钟缺供。
    Aeq=sparse.coo_matrix((np.r_[-np.full(nd,dt),np.ones(nc)],
        (np.r_[which,np.arange(nc)],np.r_[np.arange(nd),nd+np.arange(nc)])),shape=(nc,nd+nc+steps)).tocsr()
    records=[]
    for fraction in c.payback_fractions:
        if fraction==0:continue
        rr=[np.arange(steps),ids];cc=[nd+nc+np.arange(steps),np.arange(nd)];vv=[-np.ones(steps),-np.ones(nd)]
        for at in range(nc):
            if ret[at,-1]<0:continue
            target=np.concatenate([np.arange(h*12,(h+1)*12) for h in ret[at]])
            rr.append(target);cc.append(np.full(len(target),nd+at));vv.append(np.full(len(target),fraction/c.payback_hours))
        A=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(steps,nd+nc+steps)).tocsr()
        obj=np.r_[np.zeros(nd+nc),np.full(steps,dt)]
        bounds=[(0,float(power[0,t//3])) for t in ids]+[(0,None)]*(nc+steps)
        # 所有逐期状态界由初态与全年最大消耗的严格界核定不绑定；此处使用其等价消元。
        state_check=old['groups']['正态1户']['physical_state_check']
        assert min(x['minimum_state_lower_MWh'] for x in state_check)>0
        say('求解完整年线性规划')
        # 直接读取原生求解器解向量；不计算此项核对不使用的逐列对偶数组。
        matrix=sparse.vstack([A,Aeq],format='csc');lp=highspy.HighsLp()
        lp.num_col_=len(obj);lp.num_row_=matrix.shape[0];lp.col_cost_=obj
        lp.col_lower_=np.array([b[0] for b in bounds]);lp.col_upper_=np.array([highspy.kHighsInf if b[1] is None else b[1] for b in bounds])
        lp.row_lower_=np.r_[np.full(steps,-highspy.kHighsInf),np.zeros(nc)]
        lp.row_upper_=np.r_[-(q+np.repeat(gap,12)),np.zeros(nc)]
        lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_=matrix.indptr;lp.a_matrix_.index_=matrix.indices;lp.a_matrix_.value_=matrix.data
        solver=highspy.Highs();solver.setOptionValue('threads',CORES);solver.setOptionValue('solver','ipm');solver.setOptionValue('output_flag',False)
        solver.passModel(lp);solver.run()
        if solver.getModelStatus()!=highspy.HighsModelStatus.kOptimal:raise RuntimeError(str(solver.getModelStatus()))
        optimum=solver.getObjectiveValue();x=np.asarray(solver.getSolution().col_value)
        say('线性规划完成，比较年度积分')
        out=annual_payback(q,c.true_duration_hours,fraction,g,power,profile,ptr,hours,values,
            np.array([e.s0[j]]),np.array([e.baseline(q)[1][j]]),base,prefix,cp,calls,ret,1,e.surplus)
        expected=max(optimum-e.s0[j],0.);difference=float(abs(out[0,0]-expected))
        if difference>c.solver_tolerance*steps:raise RuntimeError(('完整年LP与积分不一致',difference))
        records.append(dict(payback_fraction=fraction,hours=8760,time_steps=steps,annual_group=j,
            LP_incremental_loss_MWh=expected,integrated_incremental_loss_MWh=float(out[0,0]),difference_MWh=difference,
            maximum_energy_equality_residual_MWh=float(np.max(abs(Aeq@x))),state_bounds=state_check))
        say(f'完整年度LP核对：额外用电{fraction:.0%}，缺供差{difference:.3g}兆瓦时')
    write_json(DEST/'payback_check.json',dict(cases=records))


if __name__=='__main__':main()
