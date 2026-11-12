"""实验一的固定调用、续调能力与逐时缺供；数值参数由入口显式传入。"""
import numpy as np
from numba import njit,prange
from call_schedule_physics import call_ends

@njit
def interval_saving(gap, q, start, end, power_mw, hour_mask=None):
    saving = 0.
    for h in range(int(start), int(np.ceil(end))):
        if hour_mask is not None and not hour_mask[h % 24]:
            continue
        length = min(h+1,end)-max(h,start)
        raw = max(gap[h]+q,0.)
        saving += length*min(raw,power_mw)
    return saving


@njit(parallel=True)
def losses(q, schedule_load, gap, onsets, counts, lengths, total, original, extensions, power_mw, invitation_extension, extension_mask=None):
    n = len(lengths)
    out = np.empty((len(extensions),len(gap)*n))
    for k in prange(out.shape[1]):
        j = k//n
        calls = call_ends(gap[j],onsets[j],counts[j],lengths,k%n,schedule_load,invitation_extension)
        saving = np.zeros(len(extensions))
        for start,end,stop in calls:
            saving[0] += interval_saving(gap[j],q,start,end,power_mw)
            for i in range(1,len(extensions)):
                saving[i] += interval_saving(gap[j],q,end,min(stop,end+extensions[i]),power_mw,extension_mask)
        for i in range(len(extensions)):
            extra = saving[i] if i else 0.
            out[i,k] = max(total[j]-original[j]-saving[0]-extra,0.)
    return out


@njit
def clock_account(q, schedule_load, gap, onsets, counts, lengths, tail, year_ids, extension, power_mw, invitation_extension):
    # Columns: shortage outside the service period, net shortage inside it,
    # and how much service after the original stop prevented.
    clocks = np.zeros((int(year_ids.max())+1,gap.shape[1],3))
    n = len(lengths)
    for j in range(len(gap)):
        year = year_ids[j]
        weight = 0.
        for variant in range(n):
            w = tail[j*n+variant]
            if w == 0.:
                continue
            weight += w
            calls = call_ends(gap[j],onsets[j],counts[j],lengths,variant,schedule_load,invitation_extension)
            for start,end,stop in calls:
                finish = min(stop,end+extension)
                for h in range(int(start),int(np.ceil(finish))):
                    length = min(h+1,finish)-max(h,start)
                    raw = max(gap[j,h]+q,0.)
                    inc = raw-max(gap[j,h],0.)
                    saved = min(raw,power_mw)
                    clocks[year,h,0] -= w*length*inc
                    clocks[year,h,1] += w*length*(inc-saved)
                    if finish > end:
                        after = max(0.,min(h+1,finish)-max(h,end))
                        clocks[year,h,2] += w*after*saved
        if weight:
            for h in range(gap.shape[1]):
                clocks[year,h,0] += weight*(max(gap[j,h]+q,0.)-max(gap[j,h],0.))
    return clocks

