"""全年实验一：相同历史记录、不同后续能力的风险根与缺电时段。

沿用run_experiment_one的完整年度核函数；不替换为命题证明的单周期构造。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from numba import set_num_threads
from common import DATA,OUT,AUDIT,CORES,cvar,write_json
from experiment_one_parameters import ExperimentOneParameters
from system_reference import _sort_zero_background,_zero_loss
from call_schedule_physics import onset_hours,call_formats
from experiment_one_physics import losses

DEST=OUT/'exp1'
NAME='capacity_shortfall'
CURVE=DEST/'plot/annual_risk_curve.csv'
INK='#293744';GRAY='#75808A'
COLORS=['#7B858E','#79A9A3','#4E779D','#8A688D']  # 展示配色，依次为续调0、1、2、4小时。
SAMPLES=17  # 曲线基础展示网格；另加入全部定稿容量根，不重新求根。


def prepare():
    old=json.loads((AUDIT/'exp1_check.json').read_text(encoding='utf-8'))
    c=ExperimentOneParameters(**old['parameters']);cap=pd.read_csv(DEST/'capacity.csv')
    metadata=json.loads((AUDIT/'plot_metadata.json').read_text(encoding='utf-8'))
    settings=dict(parameters=old['parameters'],reference=old['no_resource_reference_MW'],roots=cap['容量边界_兆瓦'].tolist())
    if CURVE.exists() and metadata.get('实验一全年图',{}).get('settings')==settings:
        return old,c,cap,pd.read_csv(CURVE)
    with np.load(DATA/'annual/exact_path_groups.npz') as z:
        profiles,weights,ptr,hours,values,atoms=(z[k] for k in ['profile_ids','weights','ptr','hours','values','atom_to_group'])
    with np.load(DATA/'annual/background_arrays.npz') as z:g=z['g_committed_median_allnumeric']
    gap=-c.system_surplus_mw*g[profiles];original=np.zeros(len(weights))
    for j in range(len(weights)):
        sl=slice(ptr[j],ptr[j+1]);gap[j,hours[sl]]=values[sl];original[j]=values[sl].sum()
    ordered,prefix=_sort_zero_background(g,c.system_surplus_mw)
    reference=old['no_resource_reference_MW'];starts,counts=onset_hours(gap,reference)
    lengths=np.asarray([row[2] for row in call_formats()]);probability=np.repeat(weights/len(lengths),len(lengths))
    # 展示上界为定稿最大资源容量的1.13倍取整，不改变物理搜索边界或原容量。
    maximum=float(np.ceil(cap['净增加_兆瓦'].max()*1.13/10)*10)
    load_grid=np.unique(np.r_[np.linspace(0,maximum,SAMPLES),cap['净增加_兆瓦'].to_numpy()])
    rows=[]
    print(f'沿用全部{len(atoms)}个原始年度场景，补算{len(load_grid)}个展示点。',flush=True)
    for at,net in enumerate(load_grid):
        q=reference+net
        total=_zero_loss(q,ordered,prefix,g,profiles,ptr,hours,values,original,c.system_surplus_mw)[1]
        loss=losses(q,reference,gap,starts,counts,lengths,total,original,np.asarray(c.extension_hours),
                    c.power_mw,c.invitation_extension_hours)
        for i,h in enumerate(c.extension_hours):
            rows.append(dict(最多续调_小时=h,资源净容量_兆瓦=net,系统总增载_兆瓦=q,
                全年新增缺供风险_兆瓦时=cvar(loss[i],probability,c.risk_level),风险预算_兆瓦时=c.risk_budget_mwh))
        if at in (0,len(load_grid)//2,len(load_grid)-1):print(f'展示点 {at+1}/{len(load_grid)} 已完成。',flush=True)
    curve=pd.DataFrame(rows)
    for r in cap.to_dict('records'):
        part=curve[(curve['最多续调_小时']==r['最多续调_小时'])&np.isclose(curve['资源净容量_兆瓦'],r['净增加_兆瓦'],atol=c.reference_tolerance_mw,rtol=0)]
        risk=float(part.iloc[0]['全年新增缺供风险_兆瓦时'])
        if not r['下界风险_兆瓦时']-c.accounting_tolerance_mwh<=risk<=r['上界风险_兆瓦时']+c.accounting_tolerance_mwh:
            raise RuntimeError('风险曲线与原定稿容量区间不符')
    curve.to_csv(CURVE,index=False,encoding='utf-8-sig')
    metadata['实验一全年图']=dict(settings=settings,annual_atoms=len(atoms),annual_hours=g.shape[1],
        net_capacity_results_changed=False,risk_curve_points=len(load_grid),cores=CORES)
    write_json(AUDIT/'plot_metadata.json',metadata)
    return old,c,cap,curve


def draw():
    set_num_threads(CORES);old,c,cap,curve=prepare()
    hours=pd.read_csv(DEST/'plot/shortfall_timing.csv');hours['月']=hours['月日'].str[:2].astype(int)
    month_order=[6,7,8,9,10,11,12,1,2,3,4,5]  # 官方服务年度从六月开始，次年五月结束。
    matrix=hours.pivot_table(index='月',columns='小时',values='续调少缺供_兆瓦时',aggfunc='sum').reindex(index=month_order,columns=range(24)).fillna(0)
    if abs(matrix.to_numpy().sum()-old['common_tail_saving_MWh'])>c.accounting_tolerance_mwh:raise RuntimeError('月份时刻分账不符')
    matrix.rename_axis(index='月份',columns='小时').reset_index().to_csv(DEST/'plot/monthly_hourly_shortfall.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':7.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':'#586570','ytick.color':'#586570',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'figure.facecolor':'white','axes.facecolor':'white'})
    # 双栏183mm宽，四个面板依次给出共同观测、未用能力、年度容量和缺电时段。
    fig=plt.figure(figsize=(7.2,6.8))
    fig.text(.065,.965,'同一份历史记录，为什么仍对应不同容量',fontsize=12.0,weight='bold')
    fig.text(.065,.930,'四种预设能力共用20次历史、544个15分钟读数；未来年度场景与调用安排相同',fontsize=7.1,color=GRAY)
    axh=fig.add_axes([.105,.665,.375,.205])
    history=pd.read_csv(DEST/'shared_records.csv')
    stamp=pd.to_datetime(history['读数时段开始'])
    dates=stamp.dt.strftime('%Y-%m-%d');day=dates.value_counts().idxmax()  # 取原历史中计量点最多的长调用，选择规则与曲线差异无关。
    h=history[dates==day].copy();hs=pd.to_datetime(h['读数时段开始'])
    x=hs.dt.hour.to_numpy()+hs.dt.minute.to_numpy()/60
    stop=float(x[-1]+.25)  # 原15分钟时段结束时刻。
    for field,color,label in [('基线_兆瓦','#596873','基线'),('电表_兆瓦','#B68A52','电表读数')]:
        y=h[field].to_numpy();axh.step(np.r_[x,stop],np.r_[y,y[-1]],where='post',color=color,lw=1.1,label=label)
    axh.axvline(stop,color=GRAY,lw=.7,ls=(0,(3,3)))
    axh.set_title('a  历史中，四种能力留下相同记录',loc='left',fontsize=8.1,pad=11,weight='bold')
    axh.set(xlim=(x[0],stop),xlabel='调用日时刻',ylabel='用电功率（兆瓦）')
    ticks=np.linspace(x[0],stop,4);axh.set_xticks(ticks,[f'{int(t):02d}:{int(round((t%1)*60)):02d}' for t in ticks])
    axh.tick_params(labelsize=6.5,length=2);axh.grid(axis='y',color='#E7EBEF',lw=.5)
    axh.legend(loc='lower left',frameon=False,ncol=2,fontsize=6.4,handlelength=1.6)
    axh.text(.02,.98,f'{day}；真实调节均设为500兆瓦',transform=axh.transAxes,va='top',fontsize=6.0,color=GRAY)
    axh.text(1.,-.30,'基线含误差；四种能力共用图中的基线和电表记录。',
        transform=axh.transAxes,ha='right',fontsize=5.8,color=GRAY)
    axt=fig.add_axes([.615,.665,.33,.205])
    future=pd.read_csv(DEST/'plot/call_trace.csv')
    original=float(future['原邀约结束_小时'].iloc[0]-future['原邀约开始_小时'].iloc[0])
    allowed=float(future['延续邀约结束_小时'].iloc[0]-future['原邀约开始_小时'].iloc[0])
    for i,extension in enumerate(c.extension_hours):
        y=3-i;axt.barh(y,original,height=.44,color='#DAE1E6',edgecolor='none')
        if extension:axt.barh(y,min(extension,allowed-original),left=original,height=.44,color=COLORS[i],edgecolor='none')
    axt.axvline(original,color=GRAY,lw=.75,ls=(0,(3,3)))
    axt.set_yticks([3,2,1,0],[f'再调{h:g}小时' for h in c.extension_hours])
    axt.set(xlim=(0,allowed),ylim=(-.65,3.65),xlabel='距这次调用开始（小时）')
    axt.set_xticks([0,original/2,original,allowed]);axt.tick_params(labelsize=6.5,length=2)
    axt.set_title('b  未来继续发出指令，能力才出现差别',loc='left',fontsize=8.1,pad=11,weight='bold')
    axt.text(original,3.72,'原定结束',ha='center',fontsize=6.1,color=GRAY)
    axt.text(.02,-.30,'灰色：原定调用；彩色：仍可提供500兆瓦的后续时段。',transform=axt.transAxes,fontsize=5.8,color=GRAY)
    axt.spines['left'].set_visible(False);axt.tick_params(axis='y',length=0)
    ax=fig.add_axes([.105,.230,.375,.285])
    labels=['原定结束时停止','还能执行1小时','还能执行2小时','还能执行4小时']
    for i,(h,label) in enumerate(zip(c.extension_hours,labels)):
        d=curve[curve['最多续调_小时']==h].sort_values('资源净容量_兆瓦')
        ax.plot(d['资源净容量_兆瓦'],d['全年新增缺供风险_兆瓦时'],color=COLORS[i],lw=1.5,label=label)
    ax.axhline(c.risk_budget_mwh,color=INK,lw=.85,ls=(0,(4,3)),zorder=1)
    ax.text(.99,c.risk_budget_mwh,'允许上限 8760',transform=ax.get_yaxis_transform(),
            ha='right',va='bottom',fontsize=5.9,color=INK)
    annotation_y=[11500,5800,11500,5800]  # 仅错开放置四个已算容量值，避免遮住交点。
    annotation_dx=[-12,0,0,0]  # 展示位移；交点仍取定稿容量，不改变数据。
    for i,r in enumerate(cap.to_dict('records')):
        x=r['净增加_兆瓦'];ax.scatter(x,c.risk_budget_mwh,s=21,color=COLORS[i],edgecolor='white',lw=.6,zorder=5)
        ax.annotate(f'{x:.2f}',xy=(x,c.risk_budget_mwh),xytext=(x+annotation_dx[i],annotation_y[i]),
            ha='center',va='center',fontsize=7.5,weight='bold',color=COLORS[i],
            arrowprops=dict(arrowstyle='-',color=COLORS[i],lw=.7))
    ymax=2*c.risk_budget_mwh  # 展示风险根附近的两倍预算范围；全部原始风险值仍保存在曲线表内。
    ax.set(xlim=(0,curve['资源净容量_兆瓦'].max()),ylim=(0,ymax))
    ax.set_xticks([0,50,100,150,200]);ax.set_xlabel('资源净认证容量（兆瓦）',labelpad=6)
    ax.set_ylabel('最严重5%情形的平均新增缺供（兆瓦时）',labelpad=5,fontsize=6.5)
    ax.tick_params(labelsize=6.8,length=2.5);ax.grid(axis='y',color='#E7EBEF',lw=.5,zorder=0)
    ax.legend(loc='upper left',frameon=False,fontsize=5.9,handlelength=1.4,labelspacing=.4,ncol=2)
    ax.set_title('c  四种能力在相同风险要求下的容量',loc='left',fontsize=8.1,pad=11,weight='bold')
    heat=fig.add_axes([.615,.230,.33,.285])
    cmap=LinearSegmentedColormap.from_list('annual_saving',['#FAFBFC','#DCCFE2','#98779D','#593B65'])
    im=heat.imshow(matrix.to_numpy(),aspect='auto',cmap=cmap,vmin=0,interpolation='nearest')
    heat.set_yticks(range(12),[f'{m}月' for m in month_order]);heat.set_xticks([0,6,12,18,23])
    heat.set_xlabel('一天中的时刻',labelpad=5);heat.tick_params(length=0,labelsize=6.5)
    heat.set_title('d  多调4小时减少缺供的时段',loc='left',fontsize=8.1,pad=11,weight='bold')
    heat.set_xticks(np.arange(-.5,24,.5),minor=True);heat.set_yticks(np.arange(-.5,12,1),minor=True)
    heat.tick_params(which='minor',length=0);heat.grid(which='minor',axis='y',color='white',lw=.3)
    for sp in heat.spines.values():sp.set_visible(False)
    # 原定诊断区间17:00—21:00，取自实验一参数；框中为17、18、19、20四个小时。
    heat.add_patch(Rectangle((c.diagnostic_start_hour-.5,-.5),c.diagnostic_end_hour-c.diagnostic_start_hour,12,
        fill=False,edgecolor='#A77D4D',lw=1.0))
    cbax=fig.add_axes([.615,.142,.33,.012]);bar=fig.colorbar(im,cax=cbax,orientation='horizontal')
    bar.outline.set_visible(False);bar.ax.tick_params(length=2,labelsize=6.2)
    bar.set_label('减少的缺供电量（兆瓦时）',fontsize=6.5,labelpad=3)
    fig.text(.105,.157,'给定0小时与4小时后续能力，容量相差',fontsize=6.4,color=GRAY)
    fig.text(.105,.124,'120.49兆瓦',fontsize=13,color=COLORS[-1],weight='bold')
    fig.text(.615,.083,f'17:00—21:00贡献{100*old["diagnostic_interval_share"]:.1f}%',fontsize=7.2,color='#9A7247',weight='bold')
    fig.text(.065,.049,f'年度容量保留全部40,300个场景与8760小时；净容量扣除同标准系统原有边界{old["no_resource_reference_MW"]:.2f}兆瓦。',fontsize=6.0,color=GRAY)
    fig.text(.065,.021,'时段贡献按相同增载和同一组年度尾部权重计算。0、1、2、4小时均从原定调用结束时刻起算。',fontsize=6.0,color=GRAY)
    for ext in ['png','pdf','svg']:fig.savefig(DEST/(NAME+'.'+ext),dpi=350,facecolor='white')
    plt.close(fig)
    print('已输出全年实验一，原四项容量保持不变。',flush=True)


if __name__=='__main__':draw()
