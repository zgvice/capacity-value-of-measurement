"""实验二：固定客户、调用和年度场景，计算既有历史资料的容量价值。

参数入口：normal_case_parameters.py；--config可覆盖其中任意参数。
默认完整复算；--check核对已交付结果；--report仅从已交付数值生成说明和表格。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numba import set_num_threads

from common import AUDIT, CORES, OUT, write_json
from normal_case_parameters import NormalCaseParameters
from normal_customer import build_history, normal_power_paths
from historical_duration import HistoricalDuration
from history_verification import joint_bounds
from annual_call_schedule import ScheduleChoice

DEST = OUT/'exp2'
SAVED = AUDIT/'exp2_check.json'


def duration_model(c, public, meter, count, power_bound):
    return HistoricalDuration(public, meter, count, power_bound,
        bias_bound=c.load_mw*c.bias_bound_rate, solver_tolerance=c.solver_tolerance,
        duration_check_tolerance=c.duration_check_tolerance_hours)


def verified_history(c, calibration=None):
    public, meter, power_bound, maximum, truth, records, source = build_history(c)
    if calibration is None:
        bounds, calibration = joint_bounds(c, public, meter)
    else:
        family = public['error_families']
        multipliers = calibration['calibration']
        critical = np.where(family == 'global', multipliers['global']['joint_multiplier'],
                            multipliers['other']['joint_multiplier'])
        bounds = public['error_stat_SD_MW']*critical
    public = dict(public, bounds=bounds)
    durations = []
    for count in range(1, public['J']+1):
        model = duration_model(c, public, meter, count, power_bound)
        duration = model.minimum(maximum, minimum=c.verified_min_duration_hours)
        durations.append(duration)
    return public, meter, maximum, truth, records, source, calibration, durations


def make_schedule(c):
    """先形成共用调用方案，之后所有资料数量采用同一组方案。"""
    choice = ScheduleChoice(c)
    choice.add_plan(choice.engine.reference)
    current = choice.capacity(c.true_duration_hours, True)
    for _ in range(c.schedule_iterations):
        if not choice.add_plan(current['total_upper_MW']):
            break
        updated = choice.capacity(c.true_duration_hours, True,
                                  lower=current['total_lower_MW'])
        improvement = updated['net_lower_MW']-current['net_lower_MW']
        current = updated
        if improvement <= c.capacity_tolerance_mw:
            break
    return choice, current


def compare_history_groups(c,public,meter,power_bound,maximum,choice):
    """复算前半、后半及合用的历史；所有15分钟约束保留。"""
    ids = np.arange(public['J'])
    middle = len(ids)//2  # 按日期等分现有调用，不是客户能力参数。
    groups = [('较早一半记录',ids[:middle]),('较近一半记录',ids[middle:]),('全部已有记录',ids)]
    rows,separate = [],np.zeros(len(public['bounds']),bool)
    for label,members in groups:
        selected = np.isin(public['call_ids'],members)
        retain = np.asarray(public['W'][:,~selected].getnnz(axis=1)).ravel()==0
        subset = {key:public[key][selected] for key in ('baseline','left','starts','stops','call_ids')}
        subset.update(n=int(selected.sum()),J=public['J'],W=public['W'][retain][:,selected],
            bounds=public['bounds'][retain],required=np.zeros(int(retain.sum()),int),
            mean_error_MW=public['mean_error_MW'])
        model = duration_model(c,subset,meter[selected],public['J'],power_bound)
        duration = model.minimum(maximum,minimum=c.verified_min_duration_hours)
        rows.append(dict(资料=label,调用次数=len(members),全部15分钟点数=model.n,
            确认持续时长_小时=duration,**choice.capacity(duration,False)))
        if len(members)<len(ids):
            separate |= retain
    subset = dict(public,W=public['W'][separate],bounds=public['bounds'][separate],
                  required=np.zeros(int(separate.sum()),int))
    duration = duration_model(c,subset,meter,public['J'],power_bound).minimum(
        maximum,minimum=c.verified_min_duration_hours)
    return dict(rows=rows,separate_constraints_duration_hours=duration)


def parameter_table(c):
    """把集中入口中的参数和就地来源注释一并输出。"""
    source = Path(__file__).with_name('normal_case_parameters.py').read_text(encoding='utf-8')
    notes = {}
    for line in source.splitlines():
        if ' = ' in line and ' # ' in line:
            key = line.strip().split(':', 1)[0]
            notes[key] = line.split(' # ', 1)[1]
    return pd.DataFrame([dict(参数=key,取值=value,来源=notes[key])
                         for key,value in c.values().items()])


def write_results(result):
    """完整生成定稿结果，不追加旧口径。"""
    DEST.mkdir(exist_ok=True)
    c = NormalCaseParameters(**result['parameters'])
    parameter_table(c).to_csv(DEST/'parameters.csv',index=False,encoding='utf-8-sig')
    rows = result['capacity_cases']
    table = pd.DataFrame([dict(ISO掌握的资料=r['label'],确认持续时长_小时=r['duration_hours'],
        净容量下界_兆瓦=r['net_lower_MW'],净容量上界_兆瓦=r['net_upper_MW'],
        占申报量下界_百分比=100*r['net_lower_MW']/c.nomination_mw,
        占申报量上界_百分比=100*r['net_upper_MW']/c.nomination_mw) for r in rows])
    table.to_csv(DEST/'capacity.csv',index=False,encoding='utf-8-sig')
    if 'history_curve' in result:
        pd.DataFrame(result['history_curve']).to_csv(
            DEST/'record_count_capacity.csv',index=False,encoding='utf-8-sig')
    invitation = result.get('invitation_comparison')
    if invitation:
        pd.DataFrame([dict(资料=r['label'],原邀约净容量下界_兆瓦=r['net_lower_MW'],
            原邀约净容量上界_兆瓦=r['net_upper_MW'],延长邀约净容量下界_兆瓦=a['net_lower_MW'],
            延长邀约净容量上界_兆瓦=a['net_upper_MW'])
            for r,a in zip(rows[1:],invitation['capacity_cases'])]).to_csv(
                DEST/'call_rule_comparison.csv',index=False,encoding='utf-8-sig')
    invitation_text = ('沿用实际调用格式，延续至缺口解除' if c.maximum_extension_hours is None
        else f'沿用实际调用格式，原结束后最多延长{c.maximum_extension_hours:g}小时')
    body = ['# 实验二：既有历史资料的容量价值','',
        '承接[实验一](../exp1/report.md)关于同一记录与不同后续能力的比较，本实验固定客户真实能力，计算ISO逐次取得既有历史资料后能支持多少容量。当前定稿采用前置[用户拟合](../../data/README.md)中的正态参照。','',
        '## 参数与资料','',
        '| 项目 | 本轮设定 |','|---|---|',
        f'| 正常用电负荷、申报量 | {c.load_mw:g}兆瓦、{c.nomination_mw:g}兆瓦 |',
        f'| 可调功率 | 非负正态，分布位置{c.nomination_mw*c.response_mean_ratio:g}兆瓦、尺度{c.nomination_mw*c.response_sd_ratio:g}兆瓦 |',
        f'| 基线误差 | 均值{c.load_mw*c.error_mean_rate:g}兆瓦、标准差{c.load_mw*c.error_sd_rate:g}兆瓦；另允许均值偏差正负{c.load_mw*c.bias_bound_rate:g}兆瓦 |',
        f'| 历史记录 | {result["historical_calls"]}次调用，共{result["historical_points"]}个15分钟点 |',
        f'| ISO邀约 | {invitation_text}，属于模拟服务安排 |',
        f'| 客户真实持续能力 | 独立预设{c.true_duration_hours:g}小时 |',
        f'| 历史核验覆盖水平 | {100*c.verification_coverage:g}% |',
        f'| 年度风险标准 | 最严重{100*(1-c.tail_level):g}%情形的平均新增缺供不超过{c.risk_budget_mwh:g}兆瓦时 |','',
        '完整参数、来源和数值精度见[参数表](parameters.csv)。原始调用与年度系统场景固定。',
        'ISO取得每次调用的完整15分钟电表记录、对应基线和调用起止时刻；模拟中的正常用电与真实调节不作为核验输入。','',
        '## 历史资料带来多少容量','',
        '| ISO掌握的资料 | 确认持续时长（小时） | 净认证容量（兆瓦） | 占申报量 |',
        '|---|---:|---:|---:|']
    if invitation and c.maximum_extension_hours is not None:
        body[0]=f'# 实验二：原结束后最多延长{c.maximum_extension_hours:g}小时的服务安排对照'
        body[2:2]=['实验二、三共用主图采用按预定缺口维持调用的安排，主结果及过程见[复现索引](../../replication_index.csv)。本页保存服务安排的完整对照。','']
    for r in rows:
        body.append(f'| {r["label"]} | {r["duration_hours"]:.3f} | '
            f'{r["net_lower_MW"]:.2f}—{r["net_upper_MW"]:.2f} | '
            f'{100*r["net_lower_MW"]/c.nomination_mw:.2f}% |')
    body += ['',f'全部容量扣除同一个无新增资源基准{result["no_resource_reference_MW"]:.3f}兆瓦。',
        '更多历史记录共同限制基线误差能够解释的用电下降，确认的持续能力因此增加，再由全年缺电计算换算成容量。','',
        '## 确认更长持续能力后的容量','',
        '| 确认持续时长（小时） | 净认证容量（兆瓦） |','|---:|---:|']
    for r in result['duration_comparison']:
        body.append(f'| {r["duration_hours"]:.3f} | {r["net_lower_MW"]:.2f}—{r["net_upper_MW"]:.2f} |')
    body += ['', '后续容量增量受客户能力、ISO邀约范围及缺电情况共同影响。这组结果仍有小幅增加。','']
    groups = result.get('history_group_comparison')
    if groups:
        body += ['## 两组历史记录合用的结果','',
            '| 资料 | 确认持续时长（小时） | 净认证容量（兆瓦） |','|---|---:|---:|']
        for r in groups['rows']:
            body.append(f'| {r["资料"]} | {r["确认持续时长_小时"]:.3f} | '
                f'{r["net_lower_MW"]:.2f}—{r["net_upper_MW"]:.2f} |')
        body += ['',f'把两组各自的约束合在一起，仍允许持续能力只有{groups["separate_constraints_duration_hours"]:.3f}小时；'
            f'利用跨记录误差统计后，确认时长提高至{rows[1]["duration_hours"]:.3f}小时。这里比较的是同类历史记录的共同使用。','']
    if invitation:
        body += ['## ISO延长邀约的对照','',
            f'对照将邀约延续至缺口解除，客户真实持续能力仍为{c.true_duration_hours:g}小时。',
            '每列内部比较资料价值，跨列差额来自服务安排变化。','',
            '| 资料 | 原邀约容量（兆瓦） | 延长邀约容量（兆瓦） |','|---|---:|---:|']
        for r,a in zip(rows[1:],invitation['capacity_cases']):
            body.append(f'| {r["label"]} | {r["net_lower_MW"]:.2f} | {a["net_lower_MW"]:.2f}—{a["net_upper_MW"]:.2f} |')
    body += ['', '实际响应在ISO邀约结束与客户持续能力允许结束的两个时刻中先到的时刻停止；相接邀约合并计时。','',
        '[容量结果](capacity.csv)；完整输入和风险根见checks/exp2_check.json。','',
        '下一项[实验三](../exp3/report.md)比较扩大聚合规模与改善资料的作用。','']
    (DEST/'report.md').write_text('\n'.join(body),encoding='utf-8')


def check_saved():
    result = json.loads(SAVED.read_text(encoding='utf-8'))
    c = NormalCaseParameters(**result['parameters'])
    public,meter,maximum,truth,records,source,calibration,durations = verified_history(c,result['history_calibration'])
    expected = result['capacity_cases'][1]['duration_hours']
    if abs(durations[-1]-expected) > c.solver_tolerance:
        raise RuntimeError('历史确认时长与定稿不一致')
    choice = ScheduleChoice(c)
    for plan in result['plans']:
        choice.add_plan(plan['trigger_reference_MW'])
    for i,row in enumerate(result['capacity_cases'][1:],1):
        low = choice.risk(row['total_lower_MW'],row['duration_hours'],i==3)
        high = choice.risk(row['total_upper_MW'],row['duration_hours'],i==3)
        if not low['upper_MWh'] <= c.risk_budget_mwh < high['lower_MWh']:
            raise RuntimeError('定稿容量的风险上下界未夹住预算')
    # 用另一申报量检验参数传递；1%只用于核对，不进入正式情景或容量。
    factor = 1.01
    changed = normal_power_paths(replace(c,nomination_mw=c.nomination_mw*factor))
    difference = float(np.max(np.abs(changed-choice.engine.power*factor)))
    if difference > c.solver_tolerance:
        raise RuntimeError('申报量参数未传入全年功率路径')
    # 定稿次数曲线只使用本轮参数与本轮共用调用；相同时长复用同一年度根。
    capacities = {r['duration_hours']:r for r in result['capacity_cases']}
    capacities.update({r['duration_hours']:r for r in result['history_group_comparison']['rows']})
    for row in result.get('history_curve',[]):
        capacities[row['确认持续时长_小时']] = dict(net_lower_MW=row['净容量下界_兆瓦'],
            net_upper_MW=row['净容量上界_兆瓦'])
    curve = []
    for count,duration in enumerate(durations,1):
        if duration not in capacities:
            capacities[duration] = choice.capacity(duration,False)
        value = capacities[duration]
        curve.append(dict(历史调用次数=count,确认持续时长_小时=duration,
            净容量下界_兆瓦=value['net_lower_MW'],净容量上界_兆瓦=value['net_upper_MW'],
            占申报量下界_百分比=100*value['net_lower_MW']/c.nomination_mw))
    result['history_curve'] = curve
    records.to_csv(DEST/'normal_records.csv',index=False,encoding='utf-8-sig')
    result['delivery_check'] = dict(history_duration_hours=durations[-1],risk_roots_checked=3,
        nomination_scaling_max_difference_MW=difference,cores=CORES,
        parameters_from='code/core/normal_case_parameters.py')
    result['source'] = source
    write_json(SAVED,result)
    write_results(result)
    print('实验二核对通过：完整历史、三项年度风险根和申报量参数传递一致。',flush=True)


def run(c):
    started = time.perf_counter()
    DEST.mkdir(parents=True,exist_ok=True)
    public,meter,maximum,truth,records,source,calibration,durations = verified_history(c)
    choice,oracle = make_schedule(c)
    # 调用集合在全部资料比较之前固定；各次数之间不调整客户或调用安排。
    capacities = {}
    for count,duration in enumerate(durations,1):
        if duration not in capacities:
            capacities[duration] = choice.capacity(duration,False)
        print(f'{count}次历史：确认{duration:.6f}小时，净容量{capacities[duration]["net_lower_MW"]:.3f}兆瓦',flush=True)
    actual = capacities[durations[-1]]
    known = choice.capacity(truth,False)
    middle = maximum+c.duration_comparison_fraction*(truth-maximum)
    longer = [choice.capacity(d,False) for d in sorted(set([maximum,middle]))]
    result = dict(parameters=c.values(),historical_points=public['n'],historical_calls=public['J'],
        history_search_upper_hours=maximum,history_calibration=calibration,
        no_resource_reference_MW=choice.engine.reference,
        plans=[dict(trigger_reference_MW=p['threshold_MW'],call_blocks=len(p['calls'])) for p in choice.plans],
        capacity_cases=[dict(label='最近1次调用',**capacities[durations[0]]),
            dict(label=f'全部{public["J"]}次调用',**actual),
            dict(label='真实持续能力已知，未来配对未知',**known),dict(label='完整信息',**oracle)],
        duration_comparison=[actual,*longer,known],source=source,
        elapsed_seconds=time.perf_counter()-started)
    result['history_group_comparison'] = compare_history_groups(c,public,meter,
        source['power_prediction_bound_MW'],maximum,choice)
    result['history_curve'] = [dict(历史调用次数=count,确认持续时长_小时=d,
        净容量下界_兆瓦=capacities[d]['net_lower_MW'],净容量上界_兆瓦=capacities[d]['net_upper_MW'],
        占申报量下界_百分比=100*capacities[d]['net_lower_MW']/c.nomination_mw)
        for count,d in enumerate(durations,1)]
    # 邀约对照独立计算，历史及客户输入不变。
    if c.maximum_extension_hours is not None:
        alternative = replace(c,maximum_extension_hours=None)
        other,other_oracle = make_schedule(alternative)
        result['invitation_comparison'] = dict(parameters=alternative.values(),
            plans=[dict(trigger_reference_MW=p['threshold_MW'],call_blocks=len(p['calls'])) for p in other.plans],
            capacity_cases=[other.capacity(durations[-1],False),other.capacity(truth,False),other_oracle])
    records.to_csv(DEST/'normal_records.csv',index=False,encoding='utf-8-sig')
    write_json(SAVED,result)
    write_results(result)
    print('实验二完整复算完成。',flush=True)


if __name__ == '__main__':
    set_num_threads(CORES)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='覆盖集中参数表中指定字段的JSON文件')
    parser.add_argument('--check',action='store_true',help='核对定稿历史与容量风险根')
    parser.add_argument('--report',action='store_true',help='从定稿数值生成说明及参数表')
    args = parser.parse_args()
    if args.check:
        check_saved()
    elif args.report:
        write_results(json.loads(SAVED.read_text(encoding='utf-8')))
    else:
        run(NormalCaseParameters.read(args.config))
