"""各实验共用的真实调用格式、缺口起点与邀约结束计算。"""
import re
import numpy as np
import pandas as pd
from numba import njit,prange
from common import DATA, OUT
from common_settings import TAIL_LEVEL

DEFAULT_INVITATION_EXTENSION_HOURS = 4.  # 既定模拟邀约情景；各实验可显式传入自己的参数。
TASK_FILE = DATA/'frozen/call_tasks.csv'  # 真实调用原始资料整理出的任务表。

def call_formats():
    tasks = pd.read_csv(TASK_FILE).sort_values(['duration_hours', 'task_id'], ascending=[False, True])
    tasks = tasks.drop_duplicates('date').sort_values('date')
    formats = [(t.date, int(re.search(r'\d+', t.lead_time).group())/60, float(t.end_hour-t.start_hour))
               for t in tasks.itertuples()]
    return formats


def tail_weights(loss, probability, alpha=TAIL_LEVEL):
    # alpha由各实验集中参数传入；默认值沿用既定年度标准。
    tail_mass = 1-alpha
    order = np.argsort(-loss, kind='stable')
    w = probability[order]
    result = np.zeros(len(w))
    result[order] = np.minimum(w, np.maximum(tail_mass-np.r_[0.,np.cumsum(w)[:-1]],0.))/tail_mass
    return result


@njit(parallel=True)
def onset_hours(gap, q):
    result = np.full((len(gap), (gap.shape[1]+1)//2), -1, np.int16)
    counts = np.zeros(len(gap), np.int32)
    for j in prange(len(gap)):
        active = False
        for h in range(gap.shape[1]):
            new = gap[j,h]+q > 0
            if new and not active:
                result[j,counts[j]] = h
                counts[j] += 1
            active = new
    return result,counts


@njit
def call_ends(gap, starts, count, lengths, variant, schedule_load, maximum_extension=DEFAULT_INVITATION_EXTENSION_HOURS):
    # Preserve the original call schedule. An extension stops when the
    # continuous shortfall clears or the next original call starts.
    HOURS = len(gap)  # 年度边界由输入数组给定。
    calls = np.empty((count, 3))
    end = 0.
    used = 0
    for event in range(count):
        start = starts[event]
        if start < end:
            continue
        end = start + lengths[(variant+event)%len(lengths)]
        # None为ISO延长邀约至缺口解除的对照情景；HOURS是年度边界，客户持续能力另行约束。
        latest = float(HOURS) if maximum_extension is None else min(HOURS, end+maximum_extension)
        stop = min(end, HOURS)
        while stop < latest and gap[int(stop)]+schedule_load > 0:
            stop = min(latest, int(stop)+1)
        calls[used,0], calls[used,1], calls[used,2] = start, min(end,HOURS), stop
        used += 1
    for i in range(used-1):
        calls[i,2] = min(calls[i,2], calls[i+1,0])
    return calls[:used]

