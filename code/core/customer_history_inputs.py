"""客户日历、实际调用与负荷生成；参数由当前实验显式传入。"""
import numpy as np
from scipy.signal import lfilter
from common import DATA
CALENDAR=DATA/"frozen/customer_calendar.npz"
TASK_FILE=DATA/"frozen/call_tasks.csv"

def _load_draw(n_days, sigma, *, reference, rho, daily_share, seed):
    # rho和daily_share沿用原日内关联与日共同方差比例；当前场景均可显式传入。
    rng = np.random.default_rng(seed)
    daily = rng.standard_normal((n_days, 1))
    innovation = rng.standard_normal((n_days, 96))
    innovation[:, 0] /= np.sqrt(1 - rho**2)
    quarter = lfilter([np.sqrt(1 - rho**2)], [1., -rho], innovation, axis=1)
    # reference显式传入当前场景正常负荷，沿用定稿负荷生成公式。
    load = reference + sigma * (np.sqrt(daily_share)*daily+np.sqrt(1-daily_share)*quarter)
    if np.any(load < 0):
        raise ValueError(f'Frozen full-calendar load has a negative value ({load.min():g} MW)')
    return load

