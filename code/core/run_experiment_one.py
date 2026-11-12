"""实验一正式入口：相同15分钟记录、固定调用安排下的不同客户续调能力。"""
from functools import lru_cache
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import set_num_threads
from common import DATA,OUT,AUDIT,CORES,cvar,monotone_root,write_json
from system_reference import _sort_zero_background,_zero_loss
from call_schedule_physics import onset_hours,call_formats,tail_weights
from experiment_one_physics import losses,clock_account
from experiment_one_parameters import ExperimentOneParameters

DEST = OUT/'exp1'
METADATA = json.loads((DATA/'annual/official_reconciliation.json').read_text(encoding='utf-8'))
WEATHER_YEARS = METADATA['weather_years']  # 官方年度场景中的天气年顺序。
ROTATIONS = METADATA['rotations']  # 官方场景每个天气年的旋转档数，原数据为13。
HOURS = None  # 从年度背景数组读取完整时序长度。


def verify_common_history(c):
    frozen = pd.read_csv(DATA/'exp1_shared_input.csv',float_precision='round_trip')
    untreated = c.load_mw*frozen['正常用电相对参照'].to_numpy()
    baseline = c.load_mw*frozen['基线相对参照'].to_numpy()
    meter = untreated-c.power_mw
    if np.any(meter<0):
        raise ValueError('设定交付量超过当前模拟正常用电')
    # 所有续调情景共用同一份记录；差别从原调用结束后开始。
    pd.DataFrame(dict(读数时段开始=frozen['读数时段开始'],基线_兆瓦=baseline,
        电表_兆瓦=meter,计量响应_兆瓦=baseline-meter)).to_csv(
            DEST/'shared_records.csv',index=False,encoding='utf-8-sig')
    return dict(points=len(meter),interval_minutes=15,delivered_power_MW=c.power_mw,
        baseline_source='inputs/exp1_shared_input.csv:基线相对参照',
        simulated_untreated_source='inputs/exp1_shared_input.csv:正常用电相对参照',
        observed_fields=['基线','电表','调用起止'],same_records_for_all_extensions=True)


def hourly_table(clock):
    parts = []
    for year_id, values in enumerate(clock):
        if np.any(values):
            labels = pd.date_range(f'{WEATHER_YEARS[year_id]}-06-01', periods=HOURS, freq='h').strftime('%m-%d %H:%M')
            parts.append(pd.DataFrame(values, index=labels,
                columns=['调用外新增缺供_兆瓦时', '调用内净新增缺供_兆瓦时', '续调少缺供_兆瓦时']))
    result = pd.concat(parts).groupby(level=0).sum()
    result.index.name = '模型月日时刻'
    return result


def calculate(c,check_only=False):
    global HOURS
    DEST.mkdir(exist_ok=True)
    common_record = verify_common_history(c)
    lengths = np.array([row[2] for row in call_formats()])
    with np.load(DATA/'annual/exact_path_groups.npz') as src:
        profiles, weights, ptr, hours, values, atoms = (
            src[key] for key in ['profile_ids','weights','ptr','hours','values','atom_to_group'])
    with np.load(DATA/'annual/background_arrays.npz') as src:
        g = src['g_committed_median_allnumeric']
    HOURS = g.shape[1]
    assert np.isclose(weights.sum(),1.)
    gap = -c.system_surplus_mw*g[profiles]
    original = np.zeros(len(weights))
    for j in range(len(weights)):
        sl = slice(ptr[j],ptr[j+1])
        gap[j,hours[sl]] = values[sl]
        original[j] = values[sl].sum()
    ordered,prefix = _sort_zero_background(g,c.system_surplus_mw)
    probability = np.repeat(weights/len(lengths),len(lengths))

    @lru_cache(maxsize=128)
    def without_resource(q):
        return _zero_loss(q,ordered,prefix,g,profiles,ptr,hours,values,original,c.system_surplus_mw)

    reference, reference_bracket = monotone_root(
        lambda q:cvar(without_resource(q)[0],weights,alpha=c.risk_level), c.risk_budget_mwh, tol=c.reference_tolerance_mw)
    onsets, counts = onset_hours(gap, reference)
    print(f'调用安排固定于无客户增载边界 {reference:.6f} 兆瓦',flush=True)

    @lru_cache(maxsize=128)
    def annual(q):
        result = losses(q,reference,gap,onsets,counts,lengths,without_resource(q)[1],original,
            np.asarray(c.extension_hours),c.power_mw,c.invitation_extension_hours)
        if np.any(np.diff(result,axis=0)>c.monotonicity_tolerance_mwh):
            raise AssertionError('A longer extension increased annual shortage')
        return result

    if check_only:
        saved = pd.read_csv(DEST/'capacity.csv')
        for i,row in saved.iterrows():
            low = cvar(annual(row['下界_兆瓦'])[i],probability,alpha=c.risk_level)
            high = cvar(annual(row['上界_兆瓦'])[i],probability,alpha=c.risk_level)
            if not low <= c.risk_budget_mwh < high:
                raise RuntimeError('实验一容量风险上下界与定稿不一致')
        result = json.loads((AUDIT/'exp1_check.json').read_text(encoding='utf-8'))
        result['parameters'],result['common_history'] = c.values(),common_record
        result['delivery_check'] = dict(annual_risk_roots_checked=len(saved),cores=CORES,
            complete_quarter_hour_records_preserved=True)
        write_json(AUDIT/'exp1_check.json',result)
        write_report(result)
        print('实验一核对通过：四项年度风险根与共同15分钟记录一致。',flush=True)
        return

    upper = reference_bracket['upper_MW']+c.power_mw
    rows, diagnostics = [], []
    for i,extension in enumerate(c.extension_hours):
        risk = lambda q:cvar(annual(q)[i],probability,alpha=c.risk_level)
        root,bracket = monotone_root(risk,c.risk_budget_mwh,hi=upper,tol=c.capacity_tolerance_mw)
        risk_lo,risk_hi = risk(bracket['lower_MW']),risk(bracket['upper_MW'])
        assert risk_lo<=c.risk_budget_mwh<risk_hi
        rows.append(dict(最多续调_小时=extension,缺供预算_兆瓦时=c.risk_budget_mwh,
                         无资源增载_兆瓦=reference,容量边界_兆瓦=root,
                         净增加_兆瓦=root-reference,下界_兆瓦=bracket['lower_MW'],
                         上界_兆瓦=bracket['upper_MW'],下界风险_兆瓦时=risk_lo,上界风险_兆瓦时=risk_hi))
        tail = tail_weights(annual(root)[i],probability,alpha=c.risk_level)
        table = hourly_table(clock_account(root,reference,gap,onsets,counts,lengths,tail,profiles//ROTATIONS,extension,c.power_mw,c.invitation_extension_hours))
        np.testing.assert_allclose(table.iloc[:,:2].to_numpy().sum(),risk(root),atol=c.accounting_tolerance_mwh,rtol=0)
        table['核对口径'],table['最多续调_小时'],table['增载_兆瓦'] = '各自容量边界',extension,root
        diagnostics.append(table.reset_index())
        print(f'{extension}小时：增载{root:.6f}，净容量{root-reference:.6f}',flush=True)

    roots = [row['容量边界_兆瓦'] for row in rows]
    for row in rows:
        row['比不续调多认_兆瓦'] = row['容量边界_兆瓦']-roots[0]
    # At a common load, use the same worst 5% selected under no extension.
    # This identifies where the electrical savings occur without changing years.
    q = roots[-1]
    shared_tail = tail_weights(annual(q)[0],probability,alpha=c.risk_level)
    table = hourly_table(clock_account(q,reference,gap,onsets,counts,lengths,shared_tail,profiles//ROTATIONS,c.extension_hours[-1],c.power_mw,c.invitation_extension_hours))
    common_saving = table['续调少缺供_兆瓦时'].sum()
    np.testing.assert_allclose(common_saving,shared_tail@(annual(q)[0]-annual(q)[-1]),atol=c.accounting_tolerance_mwh,rtol=0)
    table['核对口径'],table['最多续调_小时'],table['增载_兆瓦'] = '共同增载和共同尾部',c.extension_hours[-1],q
    diagnostics.append(table.reset_index())
    hour = table.index.str[6:8].astype(int)
    evening_saving = table.loc[(hour>=c.diagnostic_start_hour)&(hour<c.diagnostic_end_hour),'续调少缺供_兆瓦时'].sum()

    # Direct capacity diagnostic. In both arms the original calls are kept.
    # Only the hours in which the extension can deliver are changed.
    evening = (np.arange(24)>=c.diagnostic_start_hour)&(np.arange(24)<c.diagnostic_end_hour)
    masked = {}
    for label,mask in [('晚间区间内',evening),('晚间区间外',~evening)]:
        def risk(q):
            outcome = losses(q,reference,gap,onsets,counts,lengths,without_resource(q)[1],original,
                np.asarray(c.extension_hours),c.power_mw,c.invitation_extension_hours,mask)
            return cvar(outcome[-1],probability,alpha=c.risk_level)
        root,_ = monotone_root(risk,c.risk_budget_mwh,hi=upper,tol=c.capacity_tolerance_mw)
        masked[label] = root-roots[0]
        print(label,masked[label],flush=True)

    pd.DataFrame(rows).to_csv(DEST/'capacity.csv',index=False,encoding='utf-8-sig')
    pd.concat(diagnostics).to_csv(DEST/'hourly_shortfall.csv',index=False,encoding='utf-8-sig')
    leading = table.nlargest(c.leading_hours_count,'续调少缺供_兆瓦时')['续调少缺供_兆瓦时']
    result = dict(parameters=c.values(),no_resource_reference_MW=reference,
        no_resource_reference_bracket=reference_bracket,annual_atoms=len(atoms),annual_hours=HOURS,
        call_templates=len(lengths),common_history=common_record,capacity_rows=rows,
        common_load_MW=q,common_tail_saving_MWh=float(common_saving),
        diagnostic_interval_share=float(evening_saving/common_saving) if common_saving else 0.,masked_capacity_gain_MW=masked,
        leading_hours=leading.to_dict(),cores=CORES)
    write_json(AUDIT/'exp1_check.json',result)
    write_report(result)


def write_report(result):
    c = ExperimentOneParameters(**result['parameters'])
    source = Path(__file__).with_name('experiment_one_parameters.py').read_text(encoding='utf-8')
    notes = {line.strip().split(':',1)[0]:line.split(' # ',1)[1]
             for line in source.splitlines() if ' = ' in line and ' # ' in line}
    pd.DataFrame([dict(参数=k,取值=v,来源=notes[k]) for k,v in c.values().items()]).to_csv(
        DEST/'parameters.csv',index=False,encoding='utf-8-sig')
    body = ['# 实验一：同一份记录对应的不同容量','',
        '前置的[用户拟合](../../data/README.md)整理客户经验分布与正态参照。本实验固定调用期间的交付量，比较记录未覆盖的后续能力。','',
        '## 参数与共同记录','',
        '| 项目 | 设定或来源 |','|---|---|',
        f'| 共同历史的正常用电参照 | {c.load_mw:g}兆瓦 |',
        f'| 原调用期间交付 | 每15分钟均交付{c.power_mw:g}兆瓦 |',
        f'| 客户最多续调 | {"、".join(f"{x:g}" for x in c.extension_hours)}小时 |',
        f'| ISO邀约 | 原结束后最多延长{c.invitation_extension_hours:g}小时，按预定缺口结束 |',
        f'| 年度场景 | 2027—2028年度{result["annual_atoms"]:,}个原始场景，完整{result["annual_hours"]}小时 |',
        f'| 调用时长 | {result["call_templates"]}种真实调用格式 |',
        f'| 风险标准 | 最严重{100*(1-c.risk_level):g}%情形的平均新增缺供不超过{c.risk_budget_mwh:g}兆瓦时 |','',
        '[参数表](parameters.csv)列出全部设定及来源。[共同15分钟记录](shared_records.csv)在全部情景中完全相同。',
        f'先计算没有新增客户时电网能多带的负荷，得到{result["no_resource_reference_MW"]:.2f}兆瓦；据此定好调用起点和邀约结束，再固定时间表计算客户容量。','',
        '## 同一份记录下的容量差异','',
        '| 客户最多续调 | 净容量（兆瓦） | 比到点就停增加（兆瓦） |','|---|---:|---:|']
    for row in result['capacity_rows']:
        body.append(f'| {row["最多续调_小时"]:g}小时 | {row["净增加_兆瓦"]:.2f} | {row["比不续调多认_兆瓦"]:.2f} |')
    body += ['', '表中容量按各续调能力分别计算。原调用结束前的记录相同，原结束后能否继续交付存在差别；ISO只看这份记录时，仍无法确定客户属于哪种能力情景。全年缺电计算给出这种能力差别对应的容量差。','',
        '## 哪些时段贡献最大','',
        f'在同一增载量{result["common_load_MW"]:.2f}兆瓦、同一组最差年度下，续调少缺供{result["common_tail_saving_MWh"]:,.0f}兆瓦时。'
        f'{c.diagnostic_start_hour:02d}:00—{c.diagnostic_end_hour:02d}:00贡献{result["diagnostic_interval_share"]:.1%}。',
        '贡献靠前的小时为'+'、'.join(result['leading_hours'])+'。','',
        f'只允许晚间区间内续调，容量增加{result["masked_capacity_gain_MW"]["晚间区间内"]:.2f}兆瓦；'
        f'只允许区间外续调，增加{result["masked_capacity_gain_MW"]["晚间区间外"]:.2f}兆瓦。两项分别求容量边界。','',
        '[容量结果](capacity.csv) · [逐时缺电](hourly_shortfall.csv)','',
        '下一项[实验二](../exp2/report.md)固定同一模拟客户，计算既有历史资料能确认多少能力、支持多少容量。','']
    (DEST/'report.md').write_text('\n'.join(body),encoding='utf-8')


if __name__ == '__main__':
    set_num_threads(CORES)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='覆盖experiment_one_parameters.py中指定参数的JSON文件')
    parser.add_argument('--check',action='store_true',help='核对定稿四项容量风险边界')
    args = parser.parse_args()
    parameters = (ExperimentOneParameters(**json.loads((AUDIT/'exp1_check.json').read_text(encoding='utf-8'))['parameters'])
        if args.check else ExperimentOneParameters.read(args.config))
    calculate(parameters,check_only=args.check)

