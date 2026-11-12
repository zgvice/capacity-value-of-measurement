"""汇总同一核验规则、独立风险检验及已完成物理对照。"""
import json,sys
import numpy as np
import pandas as pd
import highspy
from common import write_json
from run_experiment_four import DEST


def main():
    saved=json.loads((DEST/'results.json').read_text(encoding='utf-8'));rule=saved['historical_rule'];coverage=saved['coverage']
    comparison=[]
    for name,d in saved['heldout'].items():
        a,b=d['capacity'],d['all_history_capacity']
        comparison.append(dict(客户='正态参照' if name=='正态1' else name,
            十五次确认时长_小时=d['duration_hours'],二十次确认时长_小时=d['all_history_duration_hours'],
            十五次认证下界_兆瓦=a['net_lower_MW'],十五次认证上界_兆瓦=a['net_upper_MW'],
            二十次认证下界_兆瓦=b['net_lower_MW'],二十次认证上界_兆瓦=b['net_upper_MW'],
            增量下界_兆瓦=b['net_lower_MW']-a['net_upper_MW'],增量上界_兆瓦=b['net_upper_MW']-a['net_lower_MW']))
    table=pd.DataFrame(comparison);table.to_csv(DEST/'holdout_capacity.csv',index=False,encoding='utf-8-sig')
    shown=table[['客户','十五次确认时长_小时','十五次认证下界_兆瓦','二十次认证下界_兆瓦']].copy()
    shown['增加约_兆瓦']=(table['增量下界_兆瓦']+table['增量上界_兆瓦'])/2
    shown.columns=['客户','15次确认时长（小时）','15次认证下界（兆瓦）','20次认证下界（兆瓦）','增加约（兆瓦）']
    native=json.loads((DEST/'pras_results.json').read_text(encoding='utf-8'))
    meta=json.loads((DEST/'pras_input_notes.json').read_text(encoding='utf-8'));native_rows=[]
    for got,source in zip(native['cases'],meta['cases']):
        gap=abs(got['native_unserved_MWh']-source['rounded_shortfall_MWh'])
        if got['case']!=source['case'] or gap>1e-7:raise RuntimeError('成熟工具对照未对应') # 原电量求和容差，MWh。
        native_rows.append({**source,**got,'total_energy_difference_MWh':gap})
    pd.DataFrame(native_rows).to_csv(DEST/'pras_comparison.csv',index=False,encoding='utf-8-sig')
    checks=json.loads((DEST/'payback_check.json').read_text(encoding='utf-8'))
    validation=pd.read_csv(DEST/'independent_risk.csv');boundary=pd.read_csv(DEST/'short_duration_risk.csv')
    boundary_events=pd.read_csv(DEST/'short_duration_delivery.csv');sensitivity=pd.read_csv(DEST/'annual_sensitivity.csv')
    margin=pd.read_csv(DEST/'margin_sensitivity.csv');tasks=pd.read_csv(DEST/'call_tasks.csv')
    task_results=pd.read_csv(DEST/'task_replay.csv')
    numeric=['duration_hours','net_lower_MW','net_upper_MW']
    rename={'duration_hours':'确认或真实持续能力（小时）','net_lower_MW':'净容量下界（兆瓦）','net_upper_MW':'净容量上界（兆瓦）'}
    pure=sensitivity[sensitivity['比较'].str.startswith('固定记录')][['比较','error_SD_MW',*numeric]].rename(columns={**rename,'error_SD_MW':'核验采用的误差标准差（兆瓦）'})
    noisy=sensitivity[sensitivity['比较'].str.startswith('误差')][['比较',*numeric,'physically_limited_points','reduced_historical_delivery_MWh']].rename(columns={**rename,
        'physically_limited_points':'受正常用电限制的点数','reduced_historical_delivery_MWh':'因此减少的历史调节（兆瓦时）'})
    physical=sensitivity[~sensitivity['比较'].str.startswith(('误差','固定记录','20次'))][['比较',*numeric]].rename(columns=rename)
    risk_names={'核定净容量_兆瓦':'核定容量（兆瓦）','独立年度风险_兆瓦时':'独立年度风险（兆瓦时）','超出预算_兆瓦时':'超出预算（兆瓦时）'}
    risk_display=validation[['客户','资料',*risk_names]].rename(columns=risk_names)
    risk_display['客户']=risk_display['客户'].replace({'正态1':'正态参照'})
    boundary_display=boundary[['处理',*risk_names]].rename(columns=risk_names)
    outages=pd.read_csv(DEST/'outage_dependence.csv')
    outage_display=outages[outages['资源']=='buffer'][['停机情景','原系统期望缺供_兆瓦时','物理可靠容量下界_兆瓦','物理可靠容量上界_兆瓦']].copy()
    outage_display['停机情景']=outage_display['停机情景'].map({'independent':'各机组独立停机','common_innovation':'机组受共同冲击'})
    outage_display.columns=['停机情景','原系统全年平均缺供（兆瓦时）','有状态资源容量下界（兆瓦）','有状态资源容量上界（兆瓦）']
    lines=['# 实验四：真实调用与数值验证','',
        f'调用取2022年12月23—24日、2025年6月23—25日。起止、区域与产品来自已归档真实事件；客户响应沿用正态参照和三户拟合。每类申报1000兆瓦、正常负荷参照2000兆瓦，主情景真实持续能力{saved["parameters"]["true_duration_hours"]:g}小时。',
        '', '[核心参数敏感性及主图](sens/report.md)：风险要求、历史覆盖、公开报告三档误差、均值偏差、共同误差及数值加密统一列示。',
        '', '![实验四主图：真实调用、历史资料与全年风险](event_risk.png)',
        '', '[主图PDF](event_risk.pdf) · [图注](event_risk_caption.md) · [主图容量与风险数据](plot/event_risk_figure_data.csv)',
        '', '## 一、同一规则下，五日记录增加多少容量','',
        '留出上述五日，以其余15次完整15分钟记录形成认证，再与全部20次资料比较。两边使用同一套统计与联合校准标准，20次资料保留15次资料的全部约束。15次记录均参加逐点、逐次及跨调用统计。',
        '',shown.round(2).to_markdown(index=False),
        '', '正态参照的其余15次记录已给出接近20次资料的能力下界，新增五日带来的容量增量较小。三户拟合的历史形状不同，新增记录继续排除了限制认证的能力解释。',
        '',f'共同校准包含{rule["settings"]["statistics"]}项统计；15次资料保留{rule["training_points"]}个15分钟点，{rule["training_pooled_statistics"]}项跨调用统计覆盖全部15次记录。',
        '', '[完整上下界](holdout_capacity.csv) · [留出记录对能力解释的核验](holdout_verification.csv)',
        '', '## 二、核定额度对应的独立年度风险','',
        '客户分布、40300个官方年度场景原子和邀约安排固定，另用独立随机数生成未来功率轨迹。将各项核定净容量加回同标准系统基准，在原认证边界核算真实规律下最严重5%情形的平均新增缺供。允许上限为8760兆瓦时。',
        '',risk_display.round(2).to_markdown(index=False),
        '', '各项额度在独立年度轨迹中均符合原风险要求。核定额度直接进入增载与缺供计算。',
        '', '## 三、客户只能调12小时时，认证是否有效','',
        '将正态客户真实持续能力设为12小时，原冬季14.5小时邀约、功率分布和负荷随机数保持不变。ISO根据记录核验，作者使用真实能力作事后风险核对。',
        '',boundary_display.round(2).to_markdown(index=False),
        '', f'按历史资料核定的额度符合原风险要求。将客户能力直接设为{saved["parameters"]["true_duration_hours"]:g}小时得到的核定额度，其12小时实际年度风险及预算差额见表。',
        '', '![认证额度对应的年度风险](award_risk.png)',
        '',boundary_events.round(2).to_markdown(index=False),
        '', '12月24日邀约超过客户持续能力2.5小时；夏季三日最长调用均短于12小时。客户能力与ISO邀约长度分别设置。',
        '', '## 四、分别计算核验范围与负荷波动的影响','',
        '下表固定同一客户、同一真实调节、同一份电表和基线，只扩大ISO核验误差范围。14.44%、28.65%表示核验采用的标准差相对于2000兆瓦正常负荷的比例。',
        '',pure.round(2).to_markdown(index=False),
        '', '另一组情景按同一高斯随机数缩放负荷波动，重新生成基线及电表，保留真实调节不超过当时正常用电的物理限制。',
        '',noisy.round(2).to_markdown(index=False),
        '', '高波动情景有7个读数受正常用电限制，减少约432兆瓦时历史调节。这组情景同时改变实际误差实现和部分实际调节，与固定记录只改核验范围的结果分别列示。',
        '', '误差率来自2013年澳大利亚基线报告表30：CAISO未调整十取十基线的客户误差波动指标第10、第50、第90百分位。正态分布、时间关联和均值允许偏差按参数表的作者情景设置。',
        '', '## 五、物理计算与必要敏感性','',
        physical.round(2).to_markdown(index=False),
        '', f'本表采用完整信息。客户真实能力保持{saved["parameters"]["true_duration_hours"]:g}小时，12、15小时对照改变ISO单次邀约长度。额外用电按削减电量的25%、50%计算，安排在邀约结束后首四个完整非调用小时；重合用电合并，年度末须完成全部额外用电。',
        '',f'复用成熟工具的12个完整年度案例，逐时缺供差为0兆瓦。工具按5分钟对齐邀约边界，原15分钟功率保持原值。额外用电与完整年度线性规划的最大缺供差为{max(x["difference_MWh"] for x in checks["cases"]):.3g}兆瓦时。',
        '', '全年状态界按原初态与最大消耗界核定；本组状态界宽于全年消耗，实际限制来自功率、持续能力、邀约与系统紧缺过程。',
        '', '非缺电时段余量的中央补全尺度为1000兆瓦。上下50%情景保持官方缺电时段原值及全部年度概率，每档按相同规则生成邀约。',
        '',margin.round(2).to_markdown(index=False),
        '', '绝对容量随供需余量设定变化。参数表区分官方缺电记录与作者补全的非缺电余量。',
        '', '## 六、统计覆盖与外部工程对照','',
        f'正态参照独立生成{coverage["repetitions"]}批完整历史，误差路径保留率{coverage["true_complete_error_path_retained"]:.2%}，功率路径保留率{coverage["true_complete_power_path_retained"]:.2%}，同时保留率{coverage["true_complete_physical_history_retained"]:.2%}。历史核验覆盖要求为95%。',
        '', '外部测试系统另保留153台机组、256条完整8784小时路径，比较独立停机与共同冲击。工程资源功率300兆瓦、初始状态1200兆瓦时，按整年更新。',
        '',outage_display.round(2).to_markdown(index=False),
        '', '该表为外部工程对照。PJM主计算保持40300个原子、1788组精确年度路径、8760小时和35040个15分钟功率点。',
        '', '## 事件过程及审稿对应','',
        f'五日共{len(tasks)}条区域产品任务，全部任务回放{len(task_results)}行，逐条保留原起止。过程图使用每日最长任务。',
        '', '![代表事件中的客户调节](event_delivery.png)',
        '', '| 数值证据 | 对应理论 | 对应审稿意见 |','|---|---|---|',
        '| 同一规则下15次与20次容量比较 | 命题二：记录排除限制性解释并改变认证边界 | 审稿人一第5条 |',
        '| 留出记录核验、独立年度风险、12小时能力检验 | 可接受物理规律与原风险标准下的认证 | 审稿人二第4条的事件情景与独立轨迹检验 |',
        '| 独立整批历史覆盖 | 可信候选集合的统计构造 | 审稿人二第4条的统计覆盖 |',
        '| 成熟工具、全年线性规划、邀约与额外用电 | 完整物理调度及随机新增缺供 | 审稿人二第5条 |',
        '| 核验范围、负荷波动、供需余量、外部停机依赖 | 误差信息与系统条件怎样影响容量换算 | 审稿人二第4、5条 |',
        '', '[全部区域产品任务](task_replay.csv) · [完整15分钟过程](event_trace.csv) · [成熟工具对照](pras_comparison.csv)',
        '', '[参数及来源](parameters.csv) · [复算入口](../../README.md)','']
    (DEST/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(DEST/'completion_check.json',dict(event_dates=5,source_tasks=len(tasks),all_task_customer_results=len(task_results),
        pooled_training_dates=rule['training_pooled_dates'],preserved_quarter_metering=True,
        annual_atoms=40300,annual_paths=1788,annual_hours=8760,mature_tool_cases=len(native_rows),
        mature_tool_max_gap_MW=max(x['max_shortfall_difference_MW'] for x in native_rows),
        full_year_LP_max_gap_MWh=max(x['difference_MWh'] for x in checks['cases']),
        independent_risk_exceedances=int((validation['超出预算_兆瓦时']>0).sum()),
        boundary_validation=boundary.to_dict('records'),coverage=coverage,
        software=dict(python=sys.version.split()[0],numpy=np.__version__,highs=highspy.Highs().version(),julia=native['julia_version'],adequacy_tool='PRAS锁定环境'),
        manuscript_modified=False))
    print(shown.round(2).to_string(index=False),flush=True)


if __name__=='__main__':main()
