"""实验四的两个代表事件；绘制既有15分钟结果，不改变数值。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment_four import DEST


def main():
    data=pd.read_csv(DEST/'event_trace.csv')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
        'pdf.fonttype':42,'svg.fonttype':'none','font.size':11,'axes.spines.right':False,'axes.spines.top':False,
        'axes.labelcolor':'#34383D','xtick.color':'#555B61','ytick.color':'#555B61','legend.frameon':False})
    colors={'正态1':'#34678A','LSE A':'#439589','LSE B':'#BD8350','LSE C':'#966A8C'}
    labels={'正态1':'正态参照','LSE A':'LSE A拟合','LSE B':'LSE B拟合','LSE C':'LSE C拟合'}
    # 选择最长冬季日和夏季连续过程的中间日，先于图形差异；完整五日与全部任务另表保存。
    panels=[('2022-12-24','冬季调用：14.5小时'),('2025-06-24','夏季调用：7小时')]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.6),sharey=True)
    ceiling=20*np.ceil(data[data['日期'].isin([p[0] for p in panels])]['真实调节_兆瓦'].max()/1000*100/20) # 按实际数据确定上界。
    for ax,(date,title) in zip(axes,panels):
        for name,color in colors.items():
            z=data[(data['日期']==date)&(data['客户']==name)].sort_values('时段开始_小时')
            x=z['时段开始_小时'].to_numpy();y=z['真实调节_兆瓦'].to_numpy()/1000*100 # 原申报量1000MW，百分比换算。
            x=np.r_[x,x[-1]+z['时段长度_小时'].iloc[-1]];y=np.r_[y,y[-1]]
            ax.step(x,y,where='post',color=color,lw=1.35,label=labels[name])
        ax.axhline(100,color='#53575C',ls=(0,(4,3)),lw=1,zorder=0)
        ax.set_title(f'{date}\n{title}',loc='left',fontsize=13,pad=12)
        ax.set_xlabel('调用记录中的时刻');ax.set_ylim(max(0,20*np.floor(data['真实调节_兆瓦'].min()/1000*100/20)),ceiling)
        ticks=[6,10,14,18,20.5] if date.startswith('2022') else [15,17,19,21,22]
        ax.set_xticks(ticks,[f'{int(t):02d}:{int(round((t%1)*60)):02d}' for t in ticks]);ax.set_xlim(x[0],x[-1])
        ax.grid(axis='y',color='#E6E8EB',lw=.7);ax.set_axisbelow(True)
    axes[0].set_ylabel('实际调节 / 申报量（%）')
    handles,labels_=axes[0].get_legend_handles_labels();fig.legend(handles,labels_,ncol=4,loc='lower center',bbox_to_anchor=(.5,.045))
    fig.text(.09,.015,'申报1000兆瓦，每15分钟一个读数；灰色虚线为指令要求。客户响应按既定分布模拟。',fontsize=10,color='#555B61')
    fig.subplots_adjust(left=.085,right=.985,top=.84,bottom=.25,wspace=.14)
    for ext in ('png','svg','pdf'):fig.savefig(DEST/f'event_delivery.{ext}',dpi=240,facecolor='white') # 屏幕预览与矢量交付。
    plt.close(fig)
    # 核定容量对应的独立年度风险；1000MW申报过程另图保留。
    table=pd.read_csv(DEST/'short_duration_risk.csv')
    fig,ax=plt.subplots(figsize=(7.2,3.9))  # 双栏图宽；横向排列使三种核定依据及风险单位直接对应。
    risk=table['独立年度风险_兆瓦时'].to_numpy();capacity=table['核定净容量_兆瓦'].to_numpy();budget=table['风险预算_兆瓦时'].iloc[0]
    y=np.arange(len(table))[::-1]
    ax.barh(y,risk,color=['#446E9A','#4C9691','#B87471'],height=.44)
    ax.axvline(budget,color='#555B61',ls=(0,(4,3)),lw=1.1)
    ax.text(budget,2.63,f'允许上限 {budget:,.0f}',ha='center',color='#555B61',fontsize=7.6)
    for i,(r,k) in enumerate(zip(risk,capacity)):
        ax.text(r+170,y[i],f'{r:,.0f}',ha='left',va='center',fontsize=8.0)
        ax.text(1.03,y[i],f'{k:.1f}兆瓦',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8.0)
    ax.set_yticks(y,['依据15次历史核定','依据20次历史核定','按14.5小时能力核定'])
    ax.set_xlabel('最严重5%情形的平均新增缺供（兆瓦时）',fontsize=8.0,labelpad=8)
    ax.set(ylim=(-.6,2.7),xlim=(0,float(risk.max())*1.18))
    ax.set_title('客户实际只能调12小时，直接按14.5小时核定会怎样',loc='left',fontsize=10.5,pad=22)
    ax.text(1.03,2.6,'资源净认证容量',transform=ax.get_yaxis_transform(),ha='left',fontsize=7.2,color='#555B61')
    ax.grid(axis='x',color='#E6E8EB',lw=.6);ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=8,pad=8);ax.tick_params(axis='x',labelsize=7)
    fig.text(.24,.040,'三项额度均在相同年度场景下，按客户真实12小时能力重新计算风险。',fontsize=6.7,color='#555B61')
    fig.subplots_adjust(left=.24,right=.80,top=.78,bottom=.21)
    for ext in ('png','svg','pdf'):fig.savefig(DEST/f'award_risk.{ext}',dpi=240,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
