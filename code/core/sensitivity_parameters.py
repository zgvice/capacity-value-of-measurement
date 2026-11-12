"""核心参数补算：各档来源与数值核对设置。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SensitivityParameters:
    history_counts: tuple = (5, 20)  # 用户确认的两份历史资料；沿用主比较最近五次及全部二十次。
    annual_tail_levels: tuple = (.90, .95, .99)  # 用户确认的年度风险敏感性，分别考察最严重10%、5%、1%。
    risk_budget_factors: tuple = (.5, 1., 2.)  # 用户确认的作者情景：8760MWh的半倍、原值、两倍。
    history_coverages: tuple = (.90, .95, .99)  # 用户确认的历史整体覆盖对照，与年度尾部参数分别改变。
    bias_bound_rates: tuple = (0., .04, .08)  # 用户确认的作者情景：均值偏差界占正常负荷0%、4%、8%。
    error_sd_rates: tuple = (.0715, .1444, .2865)  # 澳大利亚2013基线报告表30，CAISO十取十指标的客户第10/50/90百分位。
    numerical_resolutions: tuple = ((64, 32), (128, 64), (256, 128))  # 用户确认的轨迹数/分位节点逐次加密；不改15分钟网格。
    refinement_trigger_mw: float = 1.  # 用户建议的作者数值标准；超过1MW或改变主要排序时继续加密。
    capacity_tolerance_mw: float = .1  # 沿用主计算的容量求根区间宽度，不作为分布离散精度。
    common_variance_fraction: float = .5  # 用户确认的较强共同误差作者情景；单户总误差率不变。
    surplus_scales_mw: tuple = (500., 1000., 1500.)  # 沿用实验四已完成的余量敏感性尺度。

