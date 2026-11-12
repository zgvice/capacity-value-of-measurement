"""用逐15分钟、逐户直接积分核对组合全年核函数的不同退出时刻。"""
from types import SimpleNamespace
import json
import numpy as np
from common import AUDIT,write_json
from portfolio_inputs import parameters,make_inputs,definitions
from portfolio_annual import frozen_choice,PortfolioAnnual


def main():
    c=parameters();_,_,data,_=make_inputs(c)
    source=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))['invitation_comparison']
    choice=frozen_choice(c,source);e=choice.engine;group=definitions(c)[-1]
    power=np.array([data['power'][name][:2]*w for name,w in zip(group['members'],group['weights'])])  # 前两条已生成的功率路径，仅用于核函数核对。
    history=json.loads((AUDIT/'portfolio_history.json').read_text(encoding='utf-8'))['groups'][group['name']]
    H=np.asarray(history['cases'][-1]['candidates'][0])
    q=source['capacity_cases'][0]['total_lower_MW']  # 沿用已求出的实验二容量边界，不另设物理参数。
    selected=np.unique([int(np.argmax(e.s0)),int(np.argmax(e.weights)),len(e.weights)-1])  # 最大缺供、最大权重及末组各取一条完整年度路径。
    ptr=[0];hours=[];values=[];plans=[]
    for j in selected:
        sl=slice(e.ptr[j],e.ptr[j+1]);hours.extend(e.hours[sl]);values.extend(e.values[sl]);ptr.append(len(hours))
    for original in choice.plans:
        cp=[0];blocks=[]
        for k,j in enumerate(selected):
            variant=k*(e.variants-1)//max(1,len(selected)-1)  # 从已有20种调用格式取首、中、末三种。
            col=j*e.variants+variant;block=original['calls'][original['ptr'][col]:original['ptr'][col+1]]
            blocks.append(block);cp.append(cp[-1]+len(block))
        plans.append(dict(ptr=np.array(cp,dtype=np.int64),calls=np.concatenate(blocks)))
    total=e.baseline(q)[1][selected];background=e.g[e.profiles[selected]]
    small=SimpleNamespace(g=background,profiles=np.arange(len(selected)),ptr=np.array(ptr),hours=np.array(hours),
        values=np.array(values),s0=e.s0[selected],pi=np.full(len(selected),1/len(selected)),variants=1,
        surplus=e.surplus,set_power=lambda power:None,baseline=lambda amount:(None,total))
    engine=PortfolioAnnual(c,SimpleNamespace(engine=small,plans=plans),power)
    fast=engine.losses(q,engine.duration_segments(H));direct=np.full_like(fast,np.inf)
    for plan in plans:
        for k,j in enumerate(selected):
            gap=np.maximum(q-e.surplus*background[k],0.)
            sl=slice(e.ptr[j],e.ptr[j+1]);gap[e.hours[sl]]=np.maximum(q+e.values[sl],0.)
            saved=np.zeros(len(power[0]))
            for start,original_end,stop in plan['calls'][plan['ptr'][k]:plan['ptr'][k+1]]:
                stop=min(stop,start+float(H.max()))
                if stop<=start:continue
                changes=start+H;changes=changes[(changes>start)&(changes<stop)]
                knots=np.unique(np.r_[start,stop,np.arange(np.ceil(start*4),np.ceil(stop*4))/4,changes])  # 每小时四个15分钟点，并加入各户实际退出时刻。
                middle=(knots[1:]+knots[:-1])/2;quarter=np.floor(middle*4).astype(int)
                active=middle[None,:]<start+H[:,None]
                for r in range(len(saved)):
                    delivered=(power[:,r,quarter]*active).sum(axis=0)
                    saved[r]+=np.diff(knots)@np.minimum(gap[quarter//4],delivered)
            direct[:,k]=np.minimum(direct[:,k],np.maximum(total[k]-saved-e.s0[j],0.))
    difference=float(np.max(abs(fast-direct)))
    if difference>1e-5:raise RuntimeError(('组合逐户积分与全年核函数不一致',difference))  # 1e-5兆瓦时小于原0.002兆瓦时风险求解间隙。
    path=AUDIT/'portfolio_kernel_check.json';audit=json.loads(path.read_text(encoding='utf-8'))
    audit.update(heterogeneous_duration_direct_integration_max_difference_MWh=difference,checked_annual_groups=selected.tolist(),
        checked_resource_paths=len(power[0]),checked_individual_duration_hours=H.tolist())
    write_json(path,audit);print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
