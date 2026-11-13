"""三户整场留出和全天明细诊断；核验已有拟合，不更改仿真分布。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata
from numba import set_num_threads
from common import OUT,CORES,ROOT,write_json
from customer_profiles import _empirical_nodes,_ar_scores,_reorder_nodes

USERS=('LSE A','LSE B','LSE C')  # 三个匿名实测来源。
DEST=OUT/'hubei/diag'
DRAWS=512  # 作者选择的整场诊断模拟次数；不作为独立实测场次。
NODES=32  # 当前生成方法的等概率分位节点，诊断保留该方法。
SEED=2026110601  # 本轮固定的整场诊断随机种子。
NOMINATION=1000.  # 用户确认的虚拟客户申报量；实测比例映射沿用主实验。
PAIRED = DEST/'baseline_meter_pairs.csv'


def fitted(g):
    g=g.sort_values(['event','minute'])
    ratio=(g.actual_kW/g.target_kW).to_numpy()
    scores=norm.ppf((rankdata(ratio,method='average')-.5)/len(ratio))
    valid=(g.event.eq(g.event.shift())&g.minute.diff().eq(15)).to_numpy()[1:]
    rho=float(np.corrcoef(scores[:-1][valid],scores[1:][valid])[0,1])
    return rho,_empirical_nodes(np.sort(ratio),NODES)


def statistics(matrix,minute):
    x=np.atleast_2d(matrix);result={
        '整场平均比例':x.mean(axis=1),'低侧5%分位':np.quantile(x,.05,axis=1),  # 作者诊断指标，用于观察低功率部分。
        '整场标准差':x.std(axis=1),'累计不足_兆瓦时':.25*NOMINATION*np.maximum(1-x,0).sum(axis=1)}
    for lag,label in [(1,'15分钟'),(4,'1小时'),(16,'4小时')]:  # 15分钟计量的相应滞后点数。
        pairs=[i for i in range(len(minute)-lag) if np.all(np.diff(minute[i:i+lag+1])==15)]
        if len(pairs)<3:
            result[label+'关联']=np.full(len(x),np.nan);continue
        pairs=np.asarray(pairs);a=x[:,pairs];b=x[:,pairs+lag]
        a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
        denominator=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
        result[label+'关联']=np.divide((a*b).sum(axis=1),denominator,out=np.full(len(x),np.nan),where=denominator>0)
    return result


def leave_one_out(raw):
    rows=[]
    for ui,user in enumerate(USERS):
        g=raw[raw.user==user].sort_values(['event','minute'])
        for ei,(event,held) in enumerate(g.groupby('event',sort=True)):
            training=g[g.event!=event];rho,nodes=fitted(training)
            minute=held.minute.to_numpy(int);index=(minute-minute.min())//15
            innovations=np.random.default_rng(np.random.SeedSequence([SEED,ui,ei])).standard_normal((DRAWS,int(index.max())+1))
            simulated=_reorder_nodes(_ar_scores(innovations,rho),nodes)[:,index]
            observed=(held.actual_kW/held.target_kW).to_numpy()
            actual=statistics(observed,minute);samples=statistics(simulated,minute)
            for metric,values in samples.items():
                value=float(actual[metric][0]);valid=values[np.isfinite(values)]
                if not np.isfinite(value) or not len(valid):continue
                low,mid,high=np.quantile(valid,[.025,.5,.975])  # 作者选择的95%模拟诊断区间。
                rows.append(dict(客户=user,留出场次=event,训练场次=int(training.event.nunique()),训练点数=len(training),
                    留出点数=len(held),训练相邻关联=rho,指标=metric,实测值=value,
                    模拟区间下界=low,模拟中位数=mid,模拟区间上界=high,落入诊断区间=bool(low<=value<=high)))
    codes = raw.drop_duplicates('user').set_index('user')['customer_code']
    table = pd.DataFrame(rows)
    table['customer_code'] = table['客户'].map(codes)
    table.to_csv(DEST/'event_holdout.csv',index=False,encoding='utf-8-sig')
    summary=table.groupby(['客户','指标']).agg(可检验场次=('留出场次','size'),
        落入诊断区间场次=('落入诊断区间','sum'),实测平均值=('实测值','mean'),模拟中位数平均=('模拟中位数','mean')).reset_index()
    summary['customer_code'] = summary['客户'].map(codes)
    summary.to_csv(DEST/'holdout_summary.csv',index=False,encoding='utf-8-sig')
    return summary


def full_day(raw):
    """Repeat the segment diagnostics on the anonymized paired measurements."""
    table = pd.read_csv(PAIRED, dtype={'日期':str}, float_precision='round_trip')
    summary = []
    for (user,date), day in table.groupby(['客户','日期']):
        event = pd.Timestamp(date).strftime('%m-%d')
        call = raw[(raw.user == user) & (raw.event == event)]
        if call.empty:
            raise ValueError(f'Missing event interval for {user}, {date}')
        start, stop = call.minute.min(), call.minute.max()
        day = day.sort_values('分钟').copy()
        day['片段'] = np.select([day['分钟'] < start, day['分钟'] > stop],
                               ['调用前', '调用后'], default='调用中')
        for segment, g in day.groupby('片段'):
            load = float(g['电表_千瓦'].mean())
            difference = (g['基线_千瓦'] - g['电表_千瓦']).to_numpy()
            summary.append(dict(客户=user,日期=date,片段=segment,点数=len(g),平均用电_千瓦=load,
                平均偏差率=float(difference.mean()/load),偏差标准差率=float(difference.std()/load),
                相邻差值关联=float(np.corrcoef(difference[:-1],difference[1:])[0,1]) if len(g)>2 and difference.std()>0 else np.nan))
    result = pd.DataFrame(summary)
    codes = table.drop_duplicates('客户').set_index('客户')['customer_code']
    result['customer_code'] = result['客户'].map(codes)
    result.to_csv(DEST/'daily_bias.csv',index=False,encoding='utf-8-sig')
    return dict(matched_customer_days=table.groupby(['客户','日期']).ngroups,
        quarter_points=len(table), source_columns='T0015—T2345；按日期和匿名客户编号匹配',
        interpretation='调用前、调用中、调用后分别记录基线与电表差值；不改变主实验误差分布')


def main():
    DEST.mkdir(exist_ok=True);set_num_threads(CORES)
    raw=pd.read_csv(OUT/'hubei/meter_series.csv',dtype={'event':str})
    summary=leave_one_out(raw);day=full_day(raw)
    data = ROOT/'data'
    aliases = pd.read_csv(data/'Customer_Labels.csv').set_index('customer_code')['customer_alias']
    for saved, exported in [('event_holdout.csv', 'D33.csv'), ('holdout_summary.csv', 'D40.csv'),
                            ('daily_bias.csv', 'D35.csv')]:
        table = pd.read_csv(DEST/saved, float_precision='round_trip')
        table.assign(客户=table['customer_code'].map(aliases)).to_csv(
            data/exported, index=False, encoding='utf-8-sig'
        )
    write_json(DEST/'diagnostics.json',dict(users=list(USERS),draws_per_event=DRAWS,seed=SEED,nodes=NODES,
        empirical_event_counts={u:int(raw[raw.user==u].event.nunique()) for u in USERS},full_day=day,
        leave_one_out='每次留出一整场，其余场次重新拟合经验边际及相邻秩关联；生成同长度完整时序',
        interval='每场统计量的2.5%—97.5%模拟区间，按真实场次数汇总',model_changed=False))
    print(summary.to_string(index=False),flush=True)
    print('全天明细：',day,flush=True)


if __name__=='__main__':main()
