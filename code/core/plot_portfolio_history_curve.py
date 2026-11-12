"""逐次历史资料图：累计认证曲线及每次记录带来的容量增量。"""
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.ticker import MultipleLocator
from common import AUDIT
from portfolio_inputs import DEST,PortfolioParameters


plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Microsoft YaHei','Arial','DejaVu Sans']
plt.rcParams['svg.fonttype']='none'
plt.rcParams.update({'pdf.fonttype':42,'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.65,'axes.labelcolor':'#293744','text.color':'#293744','xtick.color':'#52606B',
    'ytick.color':'#52606B','figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})
COLORS=['#727C85','#4C9691','#446E9A','#8A688D']  # 灰、青、蓝、紫：同一张图固定对应1—4户。
MARKERS=['o','s','^','D']
CMAP=LinearSegmentedColormap.from_list('record_gain',['#FAFBFB','#D5E8E5','#87BFB6','#397F77'])


def series(table,name):
    data=table[table['组合']==name].sort_values('每户历史次数')
    if list(data['每户历史次数'])!=list(range(21)):raise RuntimeError(f'{name}的0—20次数据尚未算齐')
    return data


def curve(ax,data,color,label,marker='o',width=1.8,style='-'):
    part=data[data['每户历史次数']>0];x=part['每户历史次数'].to_numpy()
    lo=part['占总申报量下界_百分比'].to_numpy();hi=part['占总申报量上界_百分比'].to_numpy()
    ax.fill_between(x,lo,hi,color=color,alpha=.10,linewidth=0,step='post')
    ax.plot(x,(lo+hi)/2,color=color,label=label,marker=marker,markersize=3.1,linewidth=width,
        markerfacecolor='white',markeredgewidth=.85,linestyle=style,drawstyle='steps-post')


def axes_style(ax,ymax,all_ticks=False):
    ax.set_xlim(.6,20.4);ax.set_ylim(0,ymax)
    ax.set_xticks(np.arange(1,21) if all_ticks else [1,5,10,15,20])
    ax.yaxis.set_major_locator(MultipleLocator(5));ax.grid(axis='y',color='#E6EBEF',linewidth=.6)
    ax.set_axisbelow(True);ax.tick_params(length=2.5,labelsize=7.5)


def gains(ax,table,names,labels,ymax,label_colors=None,nomination=1000.):
    matrix=np.array([series(table,name).query('每户历史次数>0')['本次新增约值_兆瓦'].to_numpy() for name in names])
    if matrix.min() < -1e-8:raise RuntimeError('递增历史的增量出现负值')  # 原证据集合嵌套要求。
    norm=Normalize(0,float(matrix.max()))  # 色深按图内实际新增容量标度；数字保留兆瓦单位和计算区间。
    ax.imshow(matrix,cmap=CMAP,norm=norm,aspect='auto',interpolation='nearest',extent=(.5,20.5,len(names)-.5,-.5))
    ax.set_xticks(np.arange(1,21));ax.set_yticks(np.arange(len(names)),labels)
    ax.set_xticks(np.arange(.5,21),minor=True);ax.set_yticks(np.arange(-.5,len(names)),minor=True)
    ax.grid(which='minor',color='white',linewidth=.9);ax.tick_params(which='minor',bottom=False,left=False)
    ax.tick_params(length=0,labelsize=7.2,pad=4)
    for spine in ax.spines.values():spine.set_visible(False)
    if label_colors:
        for text,color in zip(ax.get_yticklabels(),label_colors):text.set_color(color)
    for i in range(len(names)):
        data=series(table,names[i]).query('每户历史次数>0')
        lo=data['本次新增下界_兆瓦'].to_numpy();hi=data['本次新增上界_兆瓦'].to_numpy()
        for j,value in enumerate(matrix[i]):
            # 阅读图以整兆瓦显示；完整小数及上下界保留在同一份源数据中。
            if round(float(lo[j]))==round(float(hi[j])):shown=f'{value:.0f}'
            elif lo[j]<.5:shown=f'≤{np.ceil(hi[j]):.0f}'
            else:shown=f'{np.floor(lo[j]):.0f}–{np.ceil(hi[j]):.0f}'
            color='white' if norm(value)>.60 else ('#293F43' if value>=.5 else '#A5AFB3')
            ax.text(j+1,i,shown,ha='center',va='center',fontsize=5.4 if len(shown)>4 else 6.3,color=color)
    ax.set_xlabel('每户提交的完整历史调用记录数（次）',labelpad=7)


def save(fig,stem):
    for ext in ('svg','pdf','png'):
        fig.savefig(DEST/f'{stem}.{ext}',dpi=300,facecolor='white')  # PNG供阅读，SVG及PDF保留可编辑文字与矢量曲线。
    plt.close(fig)


def normal_figure(table,c,anchors):
    names=[f'正态{i}户' for i in range(1,5)]
    ymax=5*np.ceil(max(anchors[n]['full_information']['net_upper_MW'] for n in names)/c.nomination_mw*100/5)
    oracle=np.mean([(anchors[n]['full_information']['net_lower_MW']+anchors[n]['full_information']['net_upper_MW'])/2 for n in names])/c.nomination_mw*100
    fig=plt.figure(figsize=(7.2,7.3))  # 双栏约183毫米；固定每户记录数与固定总记录份数分别展示。
    fig.text(.12,.965,'同样的申报量，资料与聚合改变多少认证容量',fontsize=11.8,weight='bold')
    fig.text(.12,.932,f'各组总申报均为{c.nomination_mw:g}兆瓦，组内等分；认证比例20%对应{c.nomination_mw*.2:g}兆瓦',fontsize=7.3,color='#64727B')
    ax=fig.add_axes([.12,.548,.84,.312])
    for i,name in enumerate(names):curve(ax,series(table,name),COLORS[i],'单户' if i==0 else f'{i+1}户聚合',MARKERS[i],style='--' if i==3 else '-')
    axes_style(ax,ymax,True);ax.set_ylabel('净认证容量占总申报量（%）',labelpad=9)
    ax.set_title('a  每户资料增加时，组合的认证容量',loc='left',fontsize=8.8,pad=10,weight='bold')
    ax.axhline(oracle,color='#AEB7BF',linewidth=.9,linestyle=(0,(3,3)),zorder=0)
    ax.text(1.2,oracle+.65,f'完整信息容量约{oracle:.1f}%',color='#7B8790',fontsize=6.9)
    ax.legend(ncol=4,loc='upper right',bbox_to_anchor=(1.,1.18),handlelength=2.,columnspacing=1.4,fontsize=7.3)
    # 第5次记录对照来自当前已完成数据，用于说明同一横轴位置怎样比较聚合户数。
    one=series(table,names[0]).set_index('每户历史次数').loc[5,'认证约值_兆瓦']
    four=series(table,names[-1]).set_index('每户历史次数').loc[5,'认证约值_兆瓦']
    four_pct=four/c.nomination_mw*100
    ax.annotate(f'每户均提供5次记录\n单户约{one:.1f}兆瓦；4户聚合约{four:.1f}兆瓦',
        xy=(5,four_pct),xytext=(6.4,ymax*.79),fontsize=7.4,color='#52606B',
        arrowprops=dict(arrowstyle='-',color='#9CA8B0',lw=.7),
        bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor='none',alpha=.9))
    heat=fig.add_axes([.12,.344,.84,.108]);gains(heat,table,names,['单户','2户聚合','3户聚合','4户聚合'],ymax,COLORS,c.nomination_mw)
    heat.set_title('b  每户再提交一份记录，多认多少容量（兆瓦）',loc='left',fontsize=8.7,pad=10,weight='bold')
    allocation=pd.read_csv(DEST/'plot/fixed_record_budget.csv').set_index('组合').loc[names]
    same=fig.add_axes([.12,.113,.84,.106]);x=np.arange(len(names))
    lo=allocation['占总申报量下界_百分比'].to_numpy();hi=allocation['占总申报量上界_百分比'].to_numpy();mid=(lo+hi)/2
    for i in range(len(names)):
        same.plot([i,i],[0,mid[i]],color=COLORS[i],lw=1.8,alpha=.55)
        same.errorbar(i,mid[i],yerr=[[mid[i]-lo[i]],[hi[i]-mid[i]]],fmt=MARKERS[i],ms=5,
            color=COLORS[i],mfc='white',mew=1.1,capsize=2,lw=.8)
        same.text(i,mid[i]+1.0,f'{allocation.iloc[i]["认证约值_兆瓦"]:.1f}兆瓦',ha='center',fontsize=7.0,color=COLORS[i])
    same.set(xlim=(-.5,3.5),ylim=(0,16),ylabel='认证比例（%）')
    same.set_yticks([0,5,10,15]);same.set_xticks(x,[f'单户：12次',f'2户：每户6次',f'3户：每户4次',f'4户：每户3次'])
    same.tick_params(labelsize=6.5,length=2);same.grid(axis='y',color='#E6EBEF',lw=.5);same.set_axisbelow(True)
    same.set_title('c  总共只有12份历史时，聚合户数与资料分配怎样影响认证',loc='left',fontsize=8.5,pad=12,weight='bold')
    fig.text(.12,.047,'上两图横轴5表示每户各交5份资料；4户共交20份。下图各组的资料总量均为12份。',fontsize=6.2,color='#64727B')
    fig.text(.12,.020,'阶梯在完整记录提交后变化；阴影及误差线为计算上下界。≤n表示不超过n兆瓦，色深表示新增量。',fontsize=6.0,color='#64727B')
    save(fig,'normal_portfolio_sequence')


def empirical_figure(table,c,anchors):
    singles=['LSE A','LSE B','LSE C'];colors={'LSE A':'#446E9A','LSE B':'#4C9691','LSE C':'#B18D57'}
    combinations=[('LSE A＋LSE B',singles[:2]),('LSE A＋LSE C',[singles[0],singles[2]]),
        ('LSE B＋LSE C',singles[1:]),('LSE A＋LSE B＋LSE C',singles)]
    names=singles+[n for n,s in combinations];ymax=5*np.ceil(max(anchors[n]['full_information']['net_upper_MW'] for n in names)/c.nomination_mw*100/5)
    fig=plt.figure(figsize=(7.2,9.0))  # 保留四个聚合比较及全部7组增量，扩大图注空间。
    fig.text(.11,.973,'三户拟合的客户：单户与聚合的认证容量',fontsize=12.8,weight='bold')
    fig.text(.11,.944,f'每条线总申报均为{c.nomination_mw:g}兆瓦；紫实线为聚合申报，虚线为单户申报',fontsize=8.0,color='#64727B')
    axes=[]
    for i,(name,members) in enumerate(combinations):
        row,col=divmod(i,2);ax=fig.add_axes([.105+.475*col,.665-.275*row,.385,.205]);axes.append(ax)
        for user in members:curve(ax,series(table,user),colors[user],user,MARKERS[singles.index(user)],width=1.25,style='--')
        curve(ax,series(table,name),'#8A688D','聚合申报','D',width=2.)
        title='3户聚合' if i==3 else name.replace('水泥','')+'（2户聚合）'
        axes_style(ax,ymax);ax.set_title(('abcd'[i])+'  '+title,loc='left',fontsize=8.8,pad=7,weight='bold')
        if col==0:ax.set_ylabel('净认证容量占总申报量（%）',fontsize=7.8)
    handles=[];labels=[]
    for user in singles:
        handles.append(plt.Line2D([0],[0],color=colors[user],linestyle='--',linewidth=1.4));labels.append(user.replace('水泥','')+'单户')
    handles.append(plt.Line2D([0],[0],color='#8A688D',linewidth=2.));labels.append('聚合申报')
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.10,.925),ncol=4,columnspacing=1.6,fontsize=8.)
    heat=fig.add_axes([.20,.16,.765,.175])
    short=['LSE A单户','LSE B单户','LSE C单户','LSE A＋LSE B聚合','LSE A＋LSE C聚合','LSE B＋LSE C聚合','3户聚合']
    gains(heat,table,names,short,ymax,[colors[n] for n in singles]+['#8A688D']*4,c.nomination_mw)
    heat.set_title('e  每户再补交一次记录后，增加的认证容量（兆瓦）',loc='left',fontsize=8.8,pad=11,weight='bold')
    fig.text(.105,.085,'横轴5表示每户各提供5次调用的完整15分钟记录；2户聚合共10份，3户聚合共15份。',fontsize=6.4,color='#64727B')
    fig.text(.105,.055,'下表第5列表示从每户4次记录增加到5次后，多认的容量。数字取整；≤n表示不超过n兆瓦。',fontsize=6.4,color='#64727B')
    fig.text(.105,.025,'各聚合组的申报量在组内等分。阴影和数值区间给出计算上下界。',fontsize=6.6,color='#64727B')
    save(fig,'empirical_portfolio_sequence')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--normal-only',action='store_true');parser.add_argument('--use-refined',action='store_true');args=parser.parse_args()
    table=pd.read_csv(DEST/'portfolio_sequence.csv')
    saved=json.loads((AUDIT/'portfolio_capacity.json').read_text(encoding='utf-8'));c=PortfolioParameters(**saved['parameters'])
    if args.use_refined:
        from run_portfolio_history_curve import RESULT,export
        from refine_portfolio_history_curve import REFINED
        groups=json.loads(RESULT.read_text(encoding='utf-8'))['groups']
        if REFINED.exists():groups.update(json.loads(REFINED.read_text(encoding='utf-8'))['groups'])
        table=export(c,groups,path=None)
    normal_figure(table,c,saved['groups'])
    if not args.normal_only:
        empirical_figure(table,c,saved['groups'])
        from run_portfolio_comparison import report
        report(c,saved['groups'])


if __name__=='__main__':main()
