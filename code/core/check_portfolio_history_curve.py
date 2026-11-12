"""核对逐次曲线覆盖、原端点、每次增量及原风险标准。"""
import json
import numpy as np
import pandas as pd
from portfolio_inputs import parameters,definitions,PortfolioParameters
from run_portfolio_history_curve import RESULT,TABLE
from common import AUDIT,write_json


def main():
    data=pd.read_csv(TABLE);result=json.loads(RESULT.read_text(encoding='utf-8'));c=PortfolioParameters(**result['parameters'])
    anchors=json.loads((AUDIT/'portfolio_capacity.json').read_text(encoding='utf-8'))
    if result['parameters']!=anchors['parameters']:raise RuntimeError('逐次曲线改变了原客户或年度参数')
    if set(data['组合'])!={g['name'] for g in definitions(c)}:raise RuntimeError('缺少约定客户组合')
    endpoint_error=0.
    for group in definitions(c):
        name=group['name'];part=data[data['组合']==name].sort_values('每户历史次数')
        if list(part['每户历史次数'])!=list(range(21)):raise RuntimeError((name,'次数未覆盖0—20'))
        lo=part['认证下界_兆瓦'].to_numpy();hi=part['认证上界_兆瓦'].to_numpy()
        if np.any(lo>hi+c.solver_tolerance) or np.any(np.diff(lo)<-c.solver_tolerance) or np.any(np.diff(hi)<-c.solver_tolerance):
            raise RuntimeError((name,'容量界与历史嵌套关系不符'))
        mid=part['认证约值_兆瓦'].to_numpy();delta=part['本次新增约值_兆瓦'].to_numpy()
        if not np.allclose(delta,np.r_[0.,np.diff(mid)],atol=c.solver_tolerance):raise RuntimeError((name,'逐次增量与累计容量不符'))
        for old in anchors['groups'][name]['history_cases']:
            n=old['count'];endpoint_error=max(endpoint_error,old['net_lower_MW']-lo[n],hi[n]-old['net_upper_MW'])
    if endpoint_error>c.solver_tolerance:raise RuntimeError(('逐次曲线超出原1次和20次端点的已证区间',endpoint_error))
    kernel=json.loads((AUDIT/'portfolio_kernel_equivalence.json').read_text(encoding='utf-8'))
    physics_error=max(r['maximum_loss_difference_MWh'] for r in kernel['checks'])
    if physics_error>1e-5:raise RuntimeError('稀疏积分与原全年物理计算不符')  # 原逐15分钟独立核对的1e-5兆瓦时容差。
    result['validation']=dict(groups=len(definitions(c)),points=len(data),historical_counts_per_group=21,
        endpoint_enclosure_error_MW=endpoint_error,annual_kernel_max_difference_MWh=physics_error,
        maximum_capacity_interval_MW=float((data['认证上界_兆瓦']-data['认证下界_兆瓦']).max()),
        risk_level=c.tail_level,risk_budget_MWh=c.risk_budget_mwh)
    write_json(RESULT,result);print(json.dumps(result['validation'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
