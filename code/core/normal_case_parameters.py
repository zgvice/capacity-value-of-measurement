"""正态基准实验的参数入口；敏感性分析逐项替换这些值。"""
from common_settings import TAIL_LEVEL,RISK_BUDGET_MWH,SYSTEM_SURPLUS_MW,VERIFICATION_COVERAGE
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class NormalCaseParameters:
    load_mw: float = 2000.  # 用户指定的正常用电负荷参照。
    nomination_mw: float = 1000.  # 用户指定的申报调节功率。
    response_mean_ratio: float = 1.  # 已定正态基准：可调期间平均功率/申报功率。
    response_sd_ratio: float = .1  # 已定正态基准：功率标准差/申报功率为10%。
    response_rho15: float = 0.  # 原正态基准的相邻15分钟功率关联。
    error_mean_rate: float = 0.  # 本轮沿用已计算的零中心情景；与下方均值允许偏差分别设置。
    error_sd_rate: float = .1444  # 2013年澳大利亚报告表30：CAISO未调整十取十基线的客户误差标准差/平均负荷中位数；作为模拟量级。
    bias_bound_rate: float = .04  # 沿用误差均值允许偏差/正常负荷的4%。
    verification_coverage: float = VERIFICATION_COVERAGE  # 原核验整体覆盖设置，与年度风险尾部水平分别设置。
    error_budget_share: float = .5  # 原设置：核验失败预算一半分给误差，一半分给功率。
    global_error_budget_share: float = .5  # 原设置：误差失败预算一半用于跨调用约束。
    baseline_days: int = 10  # 原基线的日历回看天数，筛选同日型完整非调用日。
    error_calibration_days: int = 10  # 误差率的原校准基线；改变回看天数时不同时改变客户负荷波动。
    load_daily_variance_share: float = .5  # 原负荷生成：日共同扰动占一半方差。
    load_rho15: float = .5**.25  # 原负荷日内扰动每小时相关性减半；不是物理恢复。
    verified_min_duration_hours: float = 0.  # 原对照未给ISO已独立核定的最短持续时长。
    true_duration_hours: float = 14.5  # 用户2026-11-06确定；取现有20个调用日最长14.5小时为模拟能力基准，从调用开始计时。
    maximum_extension_hours: float | None = 4.  # 主比较固定的ISO邀约最大延长时间；None为延长至缺口解除的另一个邀约情景。
    tail_level: float = TAIL_LEVEL  # 原年度风险计算取最严重5%情形的平均新增缺供。
    risk_budget_mwh: float = RISK_BUDGET_MWH  # 原新增缺供风险预算；属于数值设定。
    system_surplus_mw: float = SYSTEM_SURPLUS_MW  # 原非缺电时段余量补全尺度，与客户正常负荷分别设置。
    future_paths: int = 64  # 原未来功率轨迹数；应为数值积分节点数的整数倍。
    quadrature_nodes: int = 32  # 原等概率正态分位区间的积分节点数。
    load_seed: int = 2026110313  # 原客户负荷随机种子；对照采用共同随机数。
    future_seed: int = 2026110417  # 原未来功率随机种子；对照采用共同随机数。
    capacity_tolerance_mw: float = .1  # 原容量风险根区间精度。
    reference_tolerance_mw: float = .0001  # 原无资源容量基准的求根精度。
    solver_tolerance: float = 1e-8  # 原历史核验线性规划可行性容差。
    calibration_draws: int = 262144  # 定稿联合误差边界采用2^18条模拟路径，属于数值积分设置。
    calibration_seed: int = 2026110419  # 定稿误差边界的独立模拟种子。
    calibration_batch: int = 4096  # 定稿误差边界的矩阵运算批量，控制内存。
    schedule_iterations: int = 6  # 定稿可选调用方案搜索的最大迭代次数。
    duration_comparison_fraction: float = .5  # 最长历史邀约与真实能力之间取中点，作为持续时长对照。
    time_tolerance_hours: float = 1e-10  # 原15分钟读数与调用边界的浮点比较容差，单位小时。
    hour_bin_tolerance: float = 1e-9  # 原核验统计按调用后整小时分组的边界容差。
    duration_check_tolerance_hours: float = 1e-6  # 原历史时长线性规划解的可行性复核容差。
    covariance_relative_tolerance: float = 1e-11  # 原联合误差协方差复核的相对容差。
    transport_iterations: int = 500  # 原年度最坏配对风险分解的迭代上限。
    transport_solver_tolerance: float = 1e-10  # 原年度风险分解线性规划的可行性和对偶容差。
    transport_gap_mwh: float = .002  # 原年度风险上下界允许的数值间隙，单位兆瓦时。

    def values(self):
        return asdict(self)

    @classmethod
    def read(cls, path=None):
        # 配置文件只需写本次要改变的字段；其余使用上面注明来源的默认值。
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig'))) if path else cls()
