"""实验四参数；客户、年度风险和物理参数继承实验三的定稿配置。"""
from dataclasses import dataclass
from portfolio_inputs import PortfolioParameters


@dataclass(frozen=True)
class ExperimentFourParameters(PortfolioParameters):
    winter_dates: tuple = ('2022-12-23','2022-12-24')  # 用户指定的2022年冬季；call_tasks.csv/date。
    summer_dates: tuple = ('2025-06-23','2025-06-24','2025-06-25')  # 用户指定的2025年夏季代表过程；call_tasks.csv/date。
    coverage_draws: int = 10000  # 作者选择的独立覆盖核对次数，不进入认证规则。
    coverage_seed: int = 2026110511  # 作者固定的独立核对随机种子，与原历史及校准种子不同。
    coverage_batch: int = 1000  # 数值计算批量，控制内存。
    validation_future_seed: int = 2026110513  # 作者固定的独立未来功率轨迹种子；年度系统场景及客户分布不变。
    boundary_duration_hours: float = 12.  # 作者选择的持续能力检验：客户能调12小时，冬季14.5小时邀约不变。
    surplus_factors: tuple = (.5,1.,1.5)  # 作者选择的余量敏感性：既定1000MW补全尺度的上下50%；官方缺电时段数据保持原值。
    invitation_limits_hours: tuple = (12.,15.)  # 用户此前认可12/15小时服务要求对照；客户能力使用共同参数true_duration_hours。
    payback_fractions: tuple = (0.,.25,.5)  # 既定作者情景：调用后额外用电量/本次削减电量为0、25%、50%。
    payback_hours: int = 4  # 既定作者情景：调用结束后首四个完整非调用小时增加用电。
    error_scales: tuple = (.0715,.1444,.2865)  # 2013年澳大利亚报告表30，CAISO未调整十取十基线，客户误差波动指标的第10/50/90百分位。
    sensitivity_tolerance_mw: float = .2  # 敏感性容量区间精度；主实验已有精确区间直接复用。

