"""实验四主图：真实冬夏调用、五日留出比较、独立全年风险。

只读取既有事件与年度结果。两日展示选择沿用原事件图，数值不重算。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from common import OUT,AUDIT,write_json

DEST=OUT/'exp4'
NAME='event_risk'
INK='#293744';GRAY='#7B858E';TEAL='#468D89'
NAMES=['正态1','LSE A','LSE B','LSE C']  # 正态参照及三个匿名实测来源。
LABELS=['正态参照','LSE A','LSE B','LSE C']
DAYS=['2022-12-24','2025-06-24']  # 沿用原图：最长冬季日与夏季连续过程的中间日。


def clock(h):
    m=int(round(h*60));return f'{m//60:02d}:{m%60:02d}'


def draw():
    stored=json.loads((DEST/'results.json').read_text(encoding='utf-8'));c=stored['parameters']
    completion=json.loads((DEST/'completion_check.json').read_text(encoding='utf-8'))
    data=pd.read_csv(DEST/'plot/event_trace.csv')
    event=pd.read_csv(DEST/'plot/event_summary.csv')
    comparison=pd.read_csv(DEST/'holdout_capacity.csv').set_index('客户')
    risk=pd.read_csv(DEST/'independent_risk.csv')
    budget=float(risk['风险预算_兆瓦时'].iloc[0]);nomination=c['nomination_mw']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':7.3,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':'#586570','ytick.color':'#586570',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})
    # 双栏183mm宽，158mm高；PNG是阅读预览，PDF/SVG保留可编辑文字。
    fig=plt.figure(figsize=(7.2,6.2))
    fig.text(.060,.960,'实验四：真实调用下的履约与认证检验',fontsize=12.8,weight='bold')
    fig.text(.060,.927,'调用起止来自真实事件，客户响应按既定分布模拟；全年计算沿用2027—2028官方场景',fontsize=7.2,color=GRAY)
    fig.add_artist(FancyBboxPatch((.060,.835),.89,.069,boxstyle='round,pad=0.006,rounding_size=.009',
        transform=fig.transFigure,facecolor='#F3F5F7',edgecolor='none'))
    for x in [.350,.650]:fig.add_artist(Line2D([x,x],[.848,.891],transform=fig.transFigure,color='#DCE2E7',lw=.7))
    headers=[(.080,'五日真实调用','2022年冬季2日、2025年夏季3日'),
        (.375,'客户真实持续能力',f'最多连续{c["true_duration_hours"]:g}小时，申报{nomination:g}兆瓦'),
        (.675,'完整年度检验',f'{completion["annual_atoms"]}个场景，{completion["annual_hours"]}小时')]
    for x,title,body in headers:
        fig.text(x,.883,title,fontsize=7.7,weight='bold');fig.text(x,.852,body,fontsize=7.0)
    fig.text(.060,.797,'a  冬夏代表调用中的15分钟调节',fontsize=8.7,weight='bold')
    cmap=LinearSegmentedColormap.from_list('delivery',['#3D6C95','#D7E5EF','#FBFBFA','#EADFCB','#A97E48'])
    # 展示色标以100%指令功率为中心，范围为上下50个百分点；涵盖两日全部实际数据。
    norm=TwoSlopeNorm(vmin=50,vcenter=100,vmax=150)
    panels=[(.140,.290,.480),(.585,.290,.925)]  # 两个日期的版面坐标及右侧不足比例列。
    event_summary=[]
    for k,(date,(left,width,side_x)) in enumerate(zip(DAYS,panels)):
        sub=data[data['日期']==date]
        times=np.sort(sub['时段开始_小时'].unique())
        e=event[(event['日期']==date)&(event['客户']=='正态1')].iloc[0]
        start=e['调用起点_小时'];end=e['调用终点_小时'];duration=e['调用时长_小时']
        matrix=sub.pivot(index='客户',columns='时段开始_小时',values='实际调节占要求_百分比').reindex(index=NAMES,columns=times)
        if matrix.isna().any().any():raise RuntimeError('代表事件的客户15分钟数据不完整')
        if matrix.min().min()<50 or matrix.max().max()>150:raise RuntimeError('代表事件超出统一展示色标')
        ax=fig.add_axes([left,.616,width,.123])
        im=ax.imshow(matrix.to_numpy(),aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,
                     extent=[start,end,3.5,-.5])
        ax.set_yticks(range(4),LABELS if k==0 else []);ax.tick_params(axis='y',length=0,pad=5,labelsize=7.2)
        ticks=[6,10,14,18,20.5] if k==0 else [15,17,19,21,22]  # 原报告中的当地钟点，未做时区替换。
        ax.set_xticks(ticks,[clock(t) for t in ticks]);ax.tick_params(axis='x',length=2,labelsize=6.5,pad=4)
        for sp in ax.spines.values():sp.set_visible(False)
        for y in [.5,1.5,2.5]:ax.axhline(y,color='white',lw=.7)
        fig.text(left,.765,f'{date}  {clock(start)}—{clock(end)}（{duration:g}小时）',fontsize=7.3,weight='bold')
        fig.text(side_x,.747,'不足电量（%）',ha='center',va='bottom',fontsize=5.8,color=GRAY)
        for i,name in enumerate(NAMES):
            row=event[(event['日期']==date)&(event['客户']==name)].iloc[0]
            value=float(row['逐时不足占要求_百分比'])
            ax.text(1+(side_x-left-width)/width,i,f'{value:.1f}',transform=ax.get_yaxis_transform(),
                    ha='center',va='center',fontsize=7.1,color=INK)
            event_summary.append(dict(日期=date,客户=name,起点=start,终点=end,不足电量占要求百分比=value))
    fig.text(.140,.573,'色彩：实际调节功率 / 指令功率（%）。右列：逐点不足电量占要求电量（%）。',fontsize=6.6,color=GRAY)
    cax=fig.add_axes([.140,.538,.735,.013]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.set_ticks([50,75,100,125,150]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.5,length=2,pad=3)
    cb.ax.axvline(100,color=INK,lw=.8)
    legend=[Line2D([0],[0],color=GRAY,marker='o',mfc='white',mec=GRAY,lw=.8,markersize=4.2,label='不含这五日：15次记录'),
        Line2D([0],[0],color=TEAL,marker='o',mfc=TEAL,mec='white',lw=.8,markersize=4.2,label='加入这五日：20次记录')]
    fig.legend(handles=legend,loc='center',bbox_to_anchor=(.52,.492),ncol=2,fontsize=7.0,handlelength=1.8,columnspacing=2.8)
    fig.text(.060,.455,'b  加入五日记录，核定容量怎样变化',fontsize=8.5,weight='bold')
    fig.text(.600,.455,'c  核定后，年度风险是否超过预算',fontsize=8.5,weight='bold')
    axb=fig.add_axes([.140,.195,.305,.214]);axr=fig.add_axes([.625,.195,.315,.214])
    source=[]
    for i,(name,label) in enumerate(zip(NAMES,LABELS)):
        y=3-i;item=comparison.loc[label]
        caps=[float(item['十五次认证下界_兆瓦']),float(item['二十次认证下界_兆瓦'])]
        rs=[]
        for stage,value in zip(['十五次历史','二十次历史'],caps):
            r=risk[(risk['客户']==name)&(risk['资料']==stage)].iloc[0]
            if abs(r['核定净容量_兆瓦']-value)>c['capacity_tolerance_mw']:raise RuntimeError('核定容量与独立风险不是同一额度')
            actual=float(r['独立年度风险_兆瓦时']);rs.append(actual/budget*100)
            source.append(dict(客户=label,资料=stage,核定净容量_兆瓦=value,独立年度风险_兆瓦时=actual,
                风险占预算_百分比=rs[-1],风险预算_兆瓦时=budget))
        for ax,values in [(axb,caps),(axr,rs)]:
            ax.plot(values,[y,y],color='#A5B4BB',lw=1.2,zorder=1)
            ax.scatter(values[0],y,s=29,facecolors='white',edgecolors=GRAY,lw=1.0,zorder=3,clip_on=False)
            ax.scatter(values[1],y,s=22,color=TEAL,edgecolors='white',lw=.5,zorder=4,clip_on=False)
            ax.axhline(y,color='#EFF2F4',lw=.5,zorder=0)
        axb.text(caps[1]+8,y,f'{caps[1]:.1f}',ha='left',va='center',fontsize=6.4,color=TEAL)
        axb.text(1.19,y,f'+{caps[1]-caps[0]:.1f}',transform=axb.get_yaxis_transform(),ha='center',va='center',fontsize=7,color=TEAL)
        axr.text(rs[1]+2.5,y,f'{rs[1]:.1f}%',ha='left',va='center',fontsize=6.6,color=TEAL)
    axb.set(xlim=(0,310),ylim=(-.6,3.6));axb.set_yticks([3,2,1,0],LABELS);axb.set_xticks([0,100,200,300])
    axb.set_xlabel('资源净认证容量（兆瓦）',labelpad=6)
    fig.text(.503,.427,'增加\n（兆瓦）',ha='center',va='top',fontsize=6.6,color=GRAY,linespacing=1.25)
    # 原平均新增缺供除以同一预算；百分比改变显示单位，不改变风险值或风险要求。
    axr.set(xlim=(0,120),ylim=(-.6,3.6));axr.set_yticks([]);axr.set_xticks([0,25,50,75,100])
    axr.set_xlabel('平均新增缺供占允许预算（%）',labelpad=6,fontsize=6.8)
    axr.axvline(100,color=INK,lw=.9,ls=(0,(3,3)),zorder=1)
    axr.text(100,3.77,'允许上限 100%',ha='center',va='bottom',fontsize=6.5,color=INK)
    for ax in (axb,axr):
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=5,labelsize=7.2)
        ax.tick_params(axis='x',length=2.5,pad=4,labelsize=6.4)
    fig.text(.625,.124,f'100%＝{budget:g}兆瓦时',fontsize=6.7,color=GRAY)
    fig.text(.625,.104,'8项核定额度均满足原风险要求',fontsize=7.3,color=TEAL,weight='bold')
    coverage=completion['coverage']
    fig.text(.060,.073,f'独立模拟{coverage["repetitions"]:,}批历史，误差与功率路径同时保留率{100*coverage["true_complete_physical_history_retained"]:.2f}%（要求95%）。',fontsize=6.3,color=GRAY)
    fig.text(.060,.048,f'成熟工具{completion["mature_tool_cases"]}个全年案例的逐时缺供差为0；物理敏感性和12小时能力误判另表保存。',fontsize=6.7,color=GRAY)
    fig.text(.060,.018,'代表过程展示冬夏各一日；容量比较使用五日留出，独立年度风险另用功率轨迹，风险取最差5%情形的平均新增缺供。',fontsize=6.35,color=GRAY)
    for ext in ['png','pdf','svg']:fig.savefig(DEST/(NAME+'.'+ext),dpi=350,facecolor='white')
    plt.close(fig)
    frame=pd.DataFrame(source);source_path=DEST/'plot/event_risk_figure_data.csv'
    frame.to_csv(source_path,index=False,encoding='utf-8-sig')
    index_path=OUT/'plot_index.csv';index=pd.read_csv(index_path)
    relative=str(source_path.relative_to(OUT)).replace('\\','/')
    index=index[index['文件']!=relative]
    index=pd.concat([index,pd.DataFrame([dict(实验=DEST.name,文件=relative,行数=len(frame),
        数据类别='当前实验结果及过程',字段='；'.join(frame.columns))])],ignore_index=True)
    index.to_csv(index_path,index=False,encoding='utf-8-sig')
    caption=('实验四：真实调用、历史资料及独立年度风险。a展示2022年12月24日与2025年6月24日代表过程。'
        '调用起止来自真实报告，每格为15分钟；客户调节按正态参照和三户既定拟合模拟。'
        '不足比例按逐点正缺额电量之和除以要求电量计算。b先留出2022年12月23—24日、2025年6月23—25日，'
        '再比较其余15次与全部20次记录，绘制核定净容量下界。c在对应核定额度下，采用独立功率轨迹和完整年度场景计算风险，'
        f'风险为最差5%情形的平均新增缺供，按允许预算归一化，100%对应{budget:g}兆瓦时。两套资料使用同一核验规则，主情景客户真实能力固定{c["true_duration_hours"]:g}小时，每类总申报{nomination:g}兆瓦。')
    (DEST/(NAME+'_caption.md')).write_text(caption+'\n',encoding='utf-8')
    p=AUDIT/'plot_metadata.json';meta=json.loads(p.read_text(encoding='utf-8'))
    meta['实验四主图']=dict(active_figure=str((DEST/(NAME+'.png')).relative_to(OUT.parent)),representative_dates=DAYS,
        event_summary=event_summary,capacity_and_risk_rows=len(source),original_results_changed=False,
        maximal_independent_risk_MWh=float(risk['独立年度风险_兆瓦时'].max()),risk_budget_MWh=budget)
    write_json(p,meta)
    print('实验四主图已输出：代表事件、五日留出容量、独立全年风险；原数值未改。',flush=True)


if __name__=='__main__':draw()
