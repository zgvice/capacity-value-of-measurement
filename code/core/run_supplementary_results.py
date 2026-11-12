"""从既有四实验补齐核心参数、数据核验及核心配图。"""
import json
import pandas as pd
from pipeline import call
from run_core_sensitivity import DEST,PLAN_FILE,parameters,scenarios,read,NAMES
from common import OUT


def main():
    if not PLAN_FILE.exists():call('run_core_sensitivity.py','--stage','prepare')
    else:
        context=read(DEST/'sensitivity_params.json')
        # 比较本轮参数，避免把不同基准的容量继续拼接；不另建文件指纹。
        if context['parameters']!=json.loads(json.dumps(parameters().values())):raise RuntimeError('核心参数补算的中央参数已改变，须先整理该轮结果再重算')
    if not (DEST/'precision.json').exists():call('run_core_sensitivity.py','--stage','convergence')
    path=DEST/'capacity_risk_bounds.csv';expected=len(NAMES)*len(scenarios(parameters()))*3
    if not path.exists() or len(pd.read_csv(path))!=expected:call('run_core_sensitivity.py','--stage','core')
    if not (DEST/'tail99_decomposition.json').exists():call('run_core_sensitivity.py','--stage','pairing')
    if not (OUT/'exp1/margin_sensitivity.csv').exists():call('supplement_experiment_one.py')
    if not (OUT/'hubei/diag/diagnostics.json').exists():call('validate_customer_records.py')
    call('common_error_sensitivity.py')
    # 已有端点直接复用，核对原偏差并集及汇总相对单户的容量增益。
    call('report_core_sensitivity.py');call('plot_core_sensitivity.py')
    call('run_duration_sensitivity.py');call('plot_duration_sensitivity.py')
    from prepare_presentation_data import update_index
    update_index()


if __name__=='__main__':main()
