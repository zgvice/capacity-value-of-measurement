"""Fit the three anonymized customer profiles and check the frozen inputs."""
import argparse,json
import numpy as np
import pandas as pd
from common import DATA,OUT,AUDIT,write_json
from customer_history_inputs import CALENDAR,TASK_FILE
from customer_profiles import _empirical_profile


def check_inputs():
    raw=pd.read_csv(OUT/'hubei/meter_series.csv')
    if set(raw.user) != {'LSE A', 'LSE B', 'LSE C'}:
        raise RuntimeError('Expected the three anonymized customer profiles.')
    with np.load(CALENDAR) as z:
        days=len(z['dates']);assert len(z['daytype'])==days and len(z['event'])==days
    tasks=pd.read_csv(TASK_FILE)
    with np.load(DATA/'annual/exact_path_groups.npz') as z:
        assert len(z['atom_to_group'])==40300 and len(z['weights'])==1788 and np.isclose(z['weights'].sum(),1.)
    return dict(source_records=len(raw),source_users=raw.user.nunique(),source_customer_events=int(raw.groupby('user').event.nunique().sum()),
        actual_call_days=tasks.date.nunique(),actual_tasks=len(tasks),calendar_days=days,annual_atoms=40300,annual_groups=1788,hours=8760)


def report():
    rows=[];models={}
    for user in ('LSE A','LSE B','LSE C'):
        p,_=_empirical_profile(user);models[user]=p
        rows.append(dict(用户=user,记录数=p['records'],调用场次=p['events'],平均功率比例=p['mean_shape'],
            比例标准差=p['SD_shape'],相邻15分钟关联=p['gaussian_rank_AR1']))
    table=pd.DataFrame(rows);dest=OUT/'hubei'
    table.to_csv(dest/'fitted_profiles.csv',index=False,encoding='utf-8-sig')
    write_json(DATA/'frozen/power_ratio_profiles.json',models)
    info=check_inputs();write_json(AUDIT/'data_check.json',info)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Check the supplied observations and frozen inputs.')
    args=parser.parse_args()
    info=check_inputs() if args.check else report()
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
