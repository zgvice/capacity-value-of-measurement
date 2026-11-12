"""复用已锁定外部测试系统，比较完整年度的独立与共同停机。"""
import json,time
import h5py
import numpy as np
import pandas as pd
from common import ROOT,OUT,write_json

DEST=OUT/'exp4'


def main():
    source=ROOT/'code/adequacy'
    cfg=json.loads((source/'experiment_config.json').read_text(encoding='utf-8'))
    with h5py.File(source/cfg['rts_file'],'r') as f:
        # 读取源系统完整年（本源文件为8784小时），保留153台机组的容量及逐时故障/修复概率。
        load=f['regions/load'][:].sum(axis=1)*cfg['load_scale']
        cap=f['generators/capacity'][:];lam=f['generators/failureprobability'][:];mu=f['generators/repairprobability'][:]
    N=cfg['N'];T=len(load);G=cap.shape[1];P=cfg['power_mw'];E=cfg['energy_mwh']
    # 路径数、负荷尺度、资源功率/初态和随机种子均沿用既有外部核对配置，不按结果调整。
    rng=np.random.default_rng(cfg['seed']);available={r:np.empty((N,T)) for r in ('independent','common_innovation')}
    states={}
    for t in range(T):
        u=rng.random((N,G))
        for r in available:
            z=u if r=='independent' else u[:,:1]
            states[r]=z<mu[t]/(lam[t]+mu[t]) if t==0 else np.where(states[r],z>=lam[t],z<mu[t])
            available[r][:,t]=(states[r]*cap[t]).sum(axis=1)
    rows=[]
    def ue(av,q,mode):
        deficit=np.maximum(load[None,:]+q-av-(P if mode=='firm' else 0),0.)
        if mode=='buffer':return deficit.sum(axis=1)-np.minimum(E,np.minimum(deficit,P).sum(axis=1))
        return deficit.sum(axis=1)
    for regime,av in available.items():
        baseline=ue(av,0.,'noDR').mean()
        for mode in ('noDR','firm','buffer'):
            lo,hi=0.,float(P)
            if mode=='noDR':hi=0.
            else:
                while hi-lo>.1:  # 沿用主实验0.1MW数值精度；不改变资源或风险要求。
                    mid=(lo+hi)/2
                    if ue(av,mid,mode).mean()<=baseline:lo=mid
                    else:hi=mid
            rows.append(dict(停机情景=regime,资源=mode,完整年度小时数=T,完整年度路径数=N,
                原系统期望缺供_兆瓦时=float(baseline),加入资源后期望缺供_兆瓦时=float(ue(av,0.,mode).mean()),
                物理可靠容量下界_兆瓦=lo,物理可靠容量上界_兆瓦=hi))
        print(time.strftime('%H:%M:%S'),f'外部系统全年停机比较：{regime}，期望缺供{baseline:.1f}兆瓦时',flush=True)
    DEST.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DEST/'outage_dependence.csv',index=False,encoding='utf-8-sig')
    write_json(DEST/'outage_params.json',dict(source=(source/'experiment_config.json').relative_to(ROOT).as_posix(),
        N=N,T=T,generators=G,power_MW=P,state_MWh=E,load_scale=cfg['load_scale'],seed=cfg['seed'],
        per_generator_marginal_laws_unchanged=True,initial_distribution='repair/(failure+repair)',
        regimes='逐机独立随机数；同一时刻共用随机数；均沿用既有外部对照定义。',
        physical_state='有状态资源初态为E，逐期x[t+1]=x[t]-d[t]，0<=x<=E、0<=d<=P；保持整年状态。',
        source_identity='原生工具附带测试系统；与PJM主实验的40300个年度场景分别保留。'))


if __name__=='__main__':main()
