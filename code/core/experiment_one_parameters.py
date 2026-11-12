"""实验一集中参数；每项注明来源，JSON配置可覆盖指定字段。"""
from common_settings import TAIL_LEVEL,RISK_BUDGET_MWH,SYSTEM_SURPLUS_MW
from dataclasses import asdict,dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentOneParameters:
    load_mw: float = 5000.  # 原实验一共同历史的模拟正常用电参照；inputs/exp1_shared_input.csv按此归一化。
    power_mw: float = 500.  # 用户已定：四种情景在原调用结束前均交付500兆瓦。
    extension_hours: tuple = (0.,1.,2.,4.)  # 用户已定的四种客户续调能力情景。
    invitation_extension_hours: float = 4.  # 原定模拟ISO邀约允许延长的上限，独立于客户能力。
    risk_level: float = TAIL_LEVEL  # 定稿年度标准：最严重5%情形的平均新增缺供。
    risk_budget_mwh: float = RISK_BUDGET_MWH  # 定稿数值情景的年度新增缺供预算，单位兆瓦时。
    system_surplus_mw: float = SYSTEM_SURPLUS_MW  # 原非缺电时段余量补全尺度；形状来自inputs/annual/background_arrays.npz。
    reference_tolerance_mw: float = .0001  # 原无资源容量基准的数值求根精度。
    capacity_tolerance_mw: float = .0001  # 原四种续调情景的容量求根精度。
    accounting_tolerance_mwh: float = 1e-6  # 原逐时分账和年度缺供总量的核对精度。
    monotonicity_tolerance_mwh: float = 1e-7  # 原续调时长增加时缺供不增加的浮点比较精度。
    diagnostic_start_hour: int = 17  # 定稿逐时结果中的晚间缺电区间，作为贡献分解的比较时段。
    diagnostic_end_hour: int = 21  # 上述比较区间的结束时刻，不进入主容量设定。
    leading_hours_count: int = 5  # 结果说明列出贡献最大的五个小时，属于展示选择。

    def __post_init__(self):
        if not self.extension_hours or self.extension_hours[0]!=0 or list(self.extension_hours)!=sorted(self.extension_hours):
            raise ValueError('续调比较须保留0小时基准，并按时长递增排列')

    def values(self):
        return asdict(self)

    @classmethod
    def read(cls,path=None):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig'))) if path else cls()
