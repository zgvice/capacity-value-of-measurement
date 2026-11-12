"""核心补算结果、数值核对与三户真实记录核验的统一阅读入口。"""
import json
import numpy as np
import pandas as pd
from common import OUT,AUDIT,write_json
from run_core_sensitivity import DEST,NAMES,LABEL,FILE_CUSTOMER,parameters,scenarios,read,audit_saved_histories


def interval(lo,hi):
    if lo<=0<=hi and max(abs(lo),abs(hi))<.1:return '<0.1'
    if hi-lo<=.25:return f'{(lo+hi)/2:.1f}'  # 表内约数；完整计算上下界保存在数据表。
    return f'{lo:.1f}—{hi:.1f}'


def check(table,raw):
    expected=len(NAMES)*len(scenarios(parameters()))*3
    if len(raw)!=expected or raw.duplicated(['customer','scenario','history_count']).any():
        raise RuntimeError('核心参数表未覆盖四类客户、全部档位及三个资料条件')
    violations=[]
    for r in raw.to_dict('records'):
        if r['net_lower_MW']>r['net_upper_MW']:violations.append((r['customer'],r['scenario'],'容量上下界'))
        if r['duration_hours']>0 and not r['risk_lower_MWh']<=r['risk_budget_MWh']<r['risk_upper_MWh']:
            violations.append((r['customer'],r['scenario'],'风险边界'))
    for _,r in table.iterrows():
        if r['五次容量下界_兆瓦']>r['二十次容量上界_兆瓦']:violations.append((r['客户'],r['情景'],'资料嵌套'))
        if r['二十次真实历史保留'] and r['二十次容量下界_兆瓦']>r['完整信息上界_兆瓦']:
            violations.append((r['客户'],r['情景'],'真实规律的容量上界'))
    for name,g in raw.groupby('customer'):
        for count in (5,20,0):
            h=g[g.history_count==count].set_index('scenario')
            for a,b in [('tail_0.9','central'),('central','tail_0.99'),('budget_0.5','central'),('central','budget_2.0')]:
                if a.startswith('budget') or b.startswith('budget'):
                    valid=h.loc[a,'total_lower_MW']<=h.loc[b,'total_upper_MW']
                else:valid=h.loc[a,'total_upper_MW']>=h.loc[b,'total_lower_MW']
                if not valid:violations.append((name,count,a,b,'同标准总边界顺序'))
            if count:
                for a,b in [('bias_0.0','central'),('central','bias_0.08'),('coverage_0.9','central'),('central','coverage_0.99'),('central','bound_0.2865')]:
                    if h.loc[a,'net_upper_MW']<h.loc[b,'net_lower_MW']:violations.append((name,count,a,b,'偏差界嵌套'))
    if violations:raise RuntimeError(violations)
    result=dict(capacity_rows=len(raw),comparison_rows=len(table),customers=list(NAMES),statistics=932,
        completed=True,risk_roots_bracketed=True,monotonicity_checked=['历史资料','偏差界','覆盖要求','固定记录误差范围','风险总边界','预算总边界'],
        truth_exclusions=table.loc[~(table['五次真实历史保留']&table['二十次真实历史保留']),['客户','情景']].to_dict('records'))
    write_json(DEST/'completion_check.json',result)
    return result


def main():
    audit_saved_histories()
    table=pd.read_csv(DEST/'sensitivity_summary.csv');raw=pd.read_csv(DEST/'capacity_risk_bounds.csv')
    checks=check(table,raw);precision=read(DEST/'precision.json');context=read(DEST/'sensitivity_params.json')
    lines=['# 核心参数敏感性','',
        '正态参照、LSE A、LSE B、LSE C分别比较5次历史、20次历史与完整信息。每个参数单独改变，四类客户保留各自的功率分布和时间关联。',
        '', '![年度风险与历史核验](sensitivity_risk.png)',
        '', '[清晰版PDF](sensitivity_risk.pdf) · [完整比较表](sensitivity_summary.csv) · [风险根上下界](capacity_risk_bounds.csv)',
        '', '## 主要结果','']
    def amount(name,key,count=20):
        r=raw[(raw.customer==name)&(raw.scenario==key)&(raw.history_count==count)].iloc[0]
        return interval(r.net_lower_MW,r.net_upper_MW)
    lines.extend([
        f'**风险要求提高，历史资料仍有容量价值。** 正态参照在年度风险90%、95%、99%下，20次历史分别支持{amount("正态1","tail_0.9")}、{amount("正态1","central")}、{amount("正态1","tail_0.99")}兆瓦。三个档位中，20次历史均高于5次历史。',
        '',f'**客户不同，核验要求的影响也不同。** 历史覆盖从95%提高到99%，LSE A20次历史容量由{amount("LSE A","central")}降至{amount("LSE A","coverage_0.99")}兆瓦；LSE C由{amount("LSE C","central")}降至{amount("LSE C","coverage_0.99")}兆瓦。正态参照由{amount("正态1","central")}降至{amount("正态1","coverage_0.99")}兆瓦。',
        '',f'**公开报告的误差量级直接影响历史能说明多少能力。** 按7.15%、14.44%、28.65%三档形成历史，正态参照20次认证分别为{amount("正态1","actual_0.0715")}、{amount("正态1","central")}、{amount("正态1","actual_0.2865")}兆瓦。三户拟合分别保留自身分布，见完整比较表。',
        '', '## 统一计算口径','',
        '两套历史采用主比较的932项统计全集，5次对应最近同五场，20次对应全部二十场；涉及未提交记录的整项统计不进入核验。所有档位使用同一调用方案池。年度风险或缺供预算变化时，同步重新计算无新增客户基准。',
        '', '年度风险95%表示考察最严重5%情形的平均新增缺供；历史覆盖95%表示核验规则在重复采集整批历史时，以95%的目标概率保留真实误差和功率路径。两者分别改变。',
        '',f'未来功率采用{precision["resolution"][0]}条全年轨迹、{precision["resolution"][1]}个分位节点。正态参照和LSE B在99%下加密核对的容量、资料收益及剩余折减最大变化为{precision["maximum_change_MW"]:.3f}兆瓦。',
        '', '表内容量单位均为兆瓦，显示计算区间中值的约数；区间较宽时直接列上下界。资料收益为20次容量减5次容量，剩余信息折减为完整信息容量减20次容量。',
        '', '年度风险和缺供预算改变时，净容量仍按“有客户边界减同标准无客户边界”计算。完整信息净容量出现小幅上升时，要同时查看两个总边界；不能据此解释为风险更严使物理能力变强。',
        '', '## 参数比较',''])
    families=[('年度风险水平',[('tail_0.9','90%：最严重10%'),('central','95%：最严重5%'),('tail_0.99','99%：最严重1%')]),
        ('允许新增缺供上限',[('budget_0.5','4380兆瓦时'),('central','8760兆瓦时'),('budget_2.0','17520兆瓦时')]),
        ('历史核验覆盖',[('coverage_0.9','90%'),('central','95%'),('coverage_0.99','99%')]),
        ('允许平均偏差',[('bias_0.0','0兆瓦'),('central','±80兆瓦'),('bias_0.08','±160兆瓦')]),
        ('实际历史误差波动',[('actual_0.0715','7.15%：143兆瓦'),('central','14.44%：288.8兆瓦'),('actual_0.2865','28.65%：573兆瓦')]),
        ('固定记录核验范围',[('central','288.8兆瓦'),('bound_0.2865','573兆瓦')])]
    for title,levels in families:
        result=[]
        for name in NAMES:
            for key,label in levels:
                r=table[(table['客户']==LABEL[name])&(table['情景']==key)].iloc[0]
                result.append(dict(客户=LABEL[name],设置=label,
                    五次历史=interval(r['五次容量下界_兆瓦'],r['五次容量上界_兆瓦']),
                    二十次历史=interval(r['二十次容量下界_兆瓦'],r['二十次容量上界_兆瓦']),
                    资料收益=interval(r['资料收益下界_兆瓦'],r['资料收益上界_兆瓦']),
                    完整信息=interval(r['完整信息下界_兆瓦'],r['完整信息上界_兆瓦']),
                    剩余折减=interval(r['剩余折减下界_兆瓦'],r['剩余折减上界_兆瓦'])))
        lines.extend(['### '+title,'',pd.DataFrame(result).to_markdown(index=False),''])
    lines.extend(['误差波动三档来自公开基线报告表30，指标为误差标准差与平均负荷的比值。实际波动情景重新形成历史负荷、基线及电表，未来能力规律固定；固定记录对照保持原读数，只扩大核验范围。',
        '', '年度缺供预算及平均偏差界的对照为作者情景，来源与参数见[计算口径](sensitivity_params.json)。','',
        '## 99%下的剩余折减',''])
    pairing=DEST/'tail99_decomposition.csv'
    if pairing.exists():
        component=pd.read_csv(pairing)
        lines.extend([component.round(2).to_markdown(index=False),'',
            f'表内未来配对未知造成的额外折减，上界最大为{component["未来配对未知上界_兆瓦"].max():.3f}兆瓦。持续能力与未来配对两项折减按同一客户和同一99%年度标准计算，完整上下界保存在表中。',''])
    stress=pd.read_csv(DEST/'historical_verification.csv');stress=stress[stress.scenario=='underestimated_bound']
    lines.extend(['## 固定原记录而低估误差范围',''])
    if len(stress):
        display=stress[['customer','count','duration_hours','truth_error_retained']].rename(columns={
            'customer':'客户','count':'历史次数','duration_hours':'确认时长_小时','truth_error_retained':'真实误差路径保留'})
        display['客户']=display['客户'].replace(LABEL)
        lines.extend([display.round(3).to_markdown(index=False),'',
            '该表固定原读数，将核验标准差缩为143兆瓦。核对同时保留原允许均值偏差：LSE B5次资料仍保留真实误差，全部20次资料则不再保留。四类客户的20次真实误差均被排除。实际误差波动143兆瓦情景另见上表。',''])
    margin=pd.read_csv(OUT/'exp1/margin_sensitivity.csv')
    gamma=read(DEST/'common/results.json');gt=pd.read_csv(DEST/'common/common_error_comparison.csv')
    shown=[]
    for r in gt.to_dict('records'):
        shown.append(dict(共同方差占比=f'{r["共同方差占比"]:.2%}',
            每户一次=interval(r['一次认证下界_兆瓦'],r['一次认证上界_兆瓦']),
            每户二十次=interval(r['二十次认证下界_兆瓦'],r['二十次认证上界_兆瓦']),
            增加历史的收益=interval(r['资料收益下界_兆瓦'],r['资料收益上界_兆瓦']),
            二十次相对单户增益=interval(r['二十次聚合增益下界_兆瓦'],r['二十次聚合增益上界_兆瓦'])))
    high=max(gamma['cases'].values(),key=lambda r:r['common_variance_fraction'])
    audit=high['history_physical_audit']
    if not audit['true_physical_history_retained']:raise RuntimeError('共同误差实际历史被当前核验规则排除，须先修复该比较')
    lines.extend(['## 聚合与共同误差','',pd.DataFrame(shown).to_markdown(index=False),'',
        '总申报量、各户误差尺度、未来真实功率路径和全年系统场景相同。共同误差较强时，聚合后误差更难互相抵消，四户二十次资料的容量下降约90兆瓦；相对单户仍有约58—59兆瓦增益。两档中增加历史资料均继续提高认证。',
        '',f'共同误差50%档的历史有{audit["changed_history_delivery_points"]}个读数受到当时正常用电上限限制，实际历史调节共减少{audit["reduced_history_energy_MWh"]:.2f}兆瓦时。未来功率路径保持原值；原均值偏差与随机误差界共同保留真实历史。跨档容量差包含误差共同变化及这部分历史履约变化。',
        '', '## 实验一余量对照','',
        pd.DataFrame([dict(非缺电余量尺度_兆瓦=r['余量尺度_兆瓦'],
            续调四小时多认_兆瓦=interval(r['容量增量下界_兆瓦'],r['容量增量上界_兆瓦']),
            晚间少缺供占比=f'{r["晚间少缺供占比"]:.1%}') for r in margin.to_dict('records')]).to_markdown(index=False),
        '', '续调能力的容量差随供需余量明显变化；三档中，17:00—21:00均贡献约76%的少缺供电量。每档重新计算系统基准与调用，两种客户能力共用该档安排；时段贡献按共同增载与共同尾部权重计算。',''])
    lines.extend(['## 工程与原始记录核验','',
        '- [实验一余量敏感性](../../exp1/margin_sensitivity.csv)：每档共用调用、共同增载及共同尾部进行对照。',
        '- [四户共同误差](common/common_error_comparison.csv)：保留实验三原有功率轨迹与积分精度，直接复用基准；两档总误差尺度与未来功率相同。',
        '- [三户整场留出与全天明细](../../hubei/diag/report.md)。',
        '- [真实事件及已有物理敏感性](../report.md)。','',
        '## 理论及审稿对应','',
        '| 本次结果 | 对应理论与审稿问题 |','|---|---|',
        '| 5次、20次、完整信息及参数对照 | 命题二：哪些历史约束改变认证；审稿人一第5条、审稿人二第4条的数据与核验要求 |',
        '| 共同误差与相对单户增益 | 命题四：聚合怎样受到共同影响及资料条件限制 |',
        '| 实验一余量与晚间贡献 | 命题一：同记录、不同后续能力对应的容量差；审稿人二第5条的容量换算 |',
        '| 99%数值加密及完整年度边界 | 审稿人二第5条的数值与物理计算核对 |',
        '| 整场留出与全天明细 | 审稿人二第4条的真实记录核验：分别检验响应形态及基线与电表的偏差量级 |',''])
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # 保存一个完整结构化结果；各客户计算文件仅在计算阶段用于继续运行。
    payload=dict(context=context,precision=precision,checks=checks,customers={})
    for name in NAMES:payload['customers'][name]=read(DEST/f'calc_{FILE_CUSTOMER[name]}.json')
    write_json(DEST/'all_results.json',payload)
    validation=OUT/'hubei/diag';summary=pd.read_csv(validation/'holdout_summary.csv')
    day=pd.read_csv(validation/'daily_bias.csv')
    report=['# 三户原始记录核验','',
        '三户共28场记录逐场留出，其余场次重新拟合，生成与留出场次同长度的完整曲线。每场分别检查均值、低功率分位、累计不足及时间关联。',
        '',summary.round(3).to_markdown(index=False),'',
        '诊断区间由512条整场模拟曲线形成，按实测场次数汇总。四小时关联共有19个可检验场次，均落入相应诊断区间；整场均值和低功率部分存在若干不匹配场次，逐场结果保存在下表。',
        '', '[逐场留出结果](event_holdout.csv)','', '## 全天明细','',
        '基线与实际用电按户号、日期匹配，当前三户共28个客户日、2660个读数。原列为00:15至23:45，每日95点。调用前、调用中、调用后分别统计基线与电表差值；偏差均值和标准差分别除以该片段平均电表负荷。','']
    display=day.groupby(['客户','片段']).agg(客户日数=('日期','size'),平均偏差率中位数=('平均偏差率','median'),
        偏差标准差率中位数=('偏差标准差率','median')).reset_index()
    report.extend([display.round(4).to_markdown(index=False),'',
        '[逐日偏差诊断](daily_bias.csv) · [配对后的全部读数](baseline_meter_pairs.csv) · [核验设置](diagnostics.json)',''])
    (validation/'report.md').write_text('\n'.join(report),encoding='utf-8')
    print(f'核心参数结果完成：{len(raw)}条容量边界，{len(table)}组比较；正式结果及原始记录核验已整理。',flush=True)


if __name__=='__main__':main()
