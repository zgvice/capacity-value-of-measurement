"""数据处理和四个实验的统一执行顺序。"""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from common import ROOT,OUT,AUDIT,CORES

HERE=Path(__file__).resolve().parent


def call(name,*args):
    print('运行：'+name,flush=True)
    subprocess.run([sys.executable,'-B','-X','utf8',str(HERE/name),*args],check=True,cwd=ROOT)


def check_stage(stage):
    if stage=='data':call('data_processing.py','--check')
    elif stage=='1':call('run_experiment_one.py','--check')
    elif stage=='2':
        from normal_case_parameters import NormalCaseParameters
        from run_experiment_two import verified_history
        saved=json.loads((AUDIT/'exp2_check.json').read_text(encoding='utf-8'))
        c=NormalCaseParameters(**saved['parameters'])
        public,meter,maximum,truth,records,source,calibration,durations=verified_history(c,saved['history_calibration'])
        if abs(durations[-1]-saved['capacity_cases'][1]['duration_hours'])>c.duration_check_tolerance_hours:
            raise RuntimeError('实验二历史确认时长与定稿不一致')
        for row in saved['capacity_cases'][1:]:
            if not row['risk_lower_boundary_MWh']<=c.risk_budget_mwh<row['risk_upper_boundary_MWh']:
                raise RuntimeError('实验二风险边界未夹住原预算')
        print('实验二：定稿参数、完整历史与已保存风险边界一致。',flush=True)
    elif stage=='3':
        call('run_portfolio_comparison.py','--check')
        call('check_portfolio_history_curve.py')
    else:
        saved=json.loads((OUT/'exp4/completion_check.json').read_text(encoding='utf-8'))
        if saved['annual_atoms']!=40300 or saved['annual_hours']!=8760 or saved['all_task_customer_results']!=172:
            raise RuntimeError('实验四输入覆盖与定稿不一致')
        if saved['mature_tool_max_gap_MW']!=0:raise RuntimeError('成熟工具对照未通过')
        if saved['independent_risk_exceedances']!=0:raise RuntimeError('独立年度风险超过原预算')
        coverage=saved['coverage']
        if coverage['true_complete_physical_history_retained']<coverage['required_coverage']:
            raise RuntimeError('已保存的整批历史覆盖未达核验要求')
        print('实验四：完整年度、全部事件、独立风险结果及物理对照已保存。',flush=True)


def run_stage(stage,config=None,check=False):
    if config and stage=='4':os.environ['EXPERIMENT_FOUR_CONFIG']=str(Path(config).resolve())
    if check:return check_stage(stage)
    extra=['--config',str(Path(config).resolve())] if config else []
    if stage=='data':call('data_processing.py')
    elif stage=='1':call('run_experiment_one.py',*extra)
    elif stage=='2':call('run_experiment_two.py',*extra)
    elif stage=='3':
        call('run_portfolio_comparison.py',*extra)
        for name in ('run_portfolio_history_curve.py','refine_portfolio_history_curve.py'):call(name)
        call('refine_portfolio_history_curve.py','--merge')
        call('check_portfolio_history_curve.py');call('plot_portfolio_history_curve.py')
    else:
        for name in ('run_experiment_four.py','experiment_four_validation.py','experiment_four_physics.py',
                     'experiment_four_outages.py','check_experiment_four_physics.py'):call(name)
        external=ROOT/'code/adequacy';env=os.environ.copy()
        env.setdefault('JULIA_NUM_THREADS',str(CORES));env['JULIA_PKG_PRECOMPILE_AUTO']='0'
        subprocess.run([os.environ.get('JULIA', 'julia'),'--startup-file=no','--history-file=no',
            '--project='+str(external/'env'),str(HERE/'experiment_four_native.jl'),str(OUT/'exp4')],
            check=True,cwd=ROOT,env=env)
        call('report_experiment_four.py')
        call('prepare_presentation_data.py','--stage','4')
        call('plot_experiment_four.py');call('plot_experiment_four_main.py')


def cli(stage):
    parser=argparse.ArgumentParser(description={'data':'拟合脱敏客户资料并核对冻结输入','1':'实验一：共同记录与不同能力',
        '2':'实验二：历史资料价值','3':'实验三：聚合与资料','4':'实验四：真实调用与数值验证'}[stage])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true',help='核对已保存输入和结果')
    mode.add_argument('--prepare-display',action='store_true',help='复用当前结果，整理展示数据并补齐必要过程')
    if stage=='4':mode.add_argument('--supplement',action='store_true',help='补齐核心参数敏感性、数据核验及核心图')
    if stage!='data':parser.add_argument('--config',help='当前实验的参数覆盖文件')
    args=parser.parse_args()
    if getattr(args,'supplement',False):
        if getattr(args,'config',None):parser.error('补算参数统一在code/core/sensitivity_parameters.py定义，共同基准继承前三实验')
        call('run_supplementary_results.py')
    elif args.prepare_display:
        call('prepare_presentation_data.py','--stage',stage)
    else:run_stage(stage,getattr(args,'config',None),args.check)
