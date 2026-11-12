"""展示同一批20个调用日期下，真实持续能力与资料数量如何共同影响认证容量。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from run_duration_sensitivity import DEST,POLICY
from run_core_sensitivity import NAMES,LABEL


def main():
    data=pd.read_csv(DEST/'duration_sensitivity.csv')
    expected=int(round((POLICY.baseline_hours-POLICY.minimum_hours)/POLICY.step_hours))+1
    if any(len(data[data['客户']==LABEL[name]])!=expected for name in NAMES):
        raise RuntimeError('持续能力各档尚未全部完成')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':8.2,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#293744',
        'text.color':'#293744','xtick.color':'#56636E','ytick.color':'#56636E','legend.frameon':False})
    # 183毫米双栏宽度；四幅子图共用容量轴，直接比较客户分布。
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.0),sharex=True,sharey=True)
    colors=['#637F9B','#328B82','#BC9050']
    series=[('五次','5次历史记录',colors[0],'-'),('二十次','20次历史记录',colors[1],'-'),
        ('完整信息','已知真实物理规律',colors[2],'--')]
    for ax,name,letter in zip(axes.flat,NAMES,'abcd'):
        d=data[data['客户']==LABEL[name]].sort_values('真实持续能力_小时')
        x=d['真实持续能力_小时'].to_numpy()
        for key,label,color,style in series:
            lo=d[f'{key}容量下界_兆瓦' if key!='完整信息' else '完整信息下界_兆瓦'].to_numpy()
            hi=d[f'{key}容量上界_兆瓦' if key!='完整信息' else '完整信息上界_兆瓦'].to_numpy()
            ax.plot(x,(lo+hi)/2,color=color,lw=1.6,ls=style,label=label)
            ax.fill_between(x,lo,hi,color=color,alpha=.14,linewidth=0)
        ax.set_title(f'{letter}  {LABEL[name]}',loc='left',fontsize=10.1,pad=10)
        ax.set_xlim(POLICY.minimum_hours,POLICY.baseline_hours)
        ax.set_xticks([4,6,8,10,12,14.5])  # 已计算范围内的主要小时刻度，末点为用户确定的基准。
        ax.set_xticklabels(['4','6','8','10','12','14.5'])
        ax.set_ylim(bottom=0);ax.grid(axis='y',color='#E5E9EC',lw=.55);ax.set_axisbelow(True)
        end=d.iloc[-1]
        gain=(end['资料增量下界_兆瓦']+end['资料增量上界_兆瓦'])/2
        ax.text(.02,.95,f'14.5小时下，多15次历史约增加{gain:.1f}兆瓦',
            transform=ax.transAxes,fontsize=7.0,va='top',color='#56636E')
    handles=[Line2D([],[],color=color,lw=1.6,ls=style,label=label) for _,label,color,style in series]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.939),ncol=3,fontsize=8.0)
    fig.suptitle('客户实际能调多久，历史资料能支持多少认证容量',x=.07,ha='left',y=.994,fontsize=12.2,weight='bold')
    fig.supylabel('资源净认证容量（兆瓦）',x=.015,fontsize=9.3)
    fig.supxlabel('客户实际最多能连续调节多久（小时）',y=.075,fontsize=9.3)
    fig.text(.075,.022,'各档均采用同样的20个历史调用日期；正常用电2000兆瓦，申报1000兆瓦。',fontsize=7.1,color='#56636E')
    fig.subplots_adjust(left=.105,right=.98,top=.85,bottom=.17,wspace=.17,hspace=.30)
    for ext in ('png','pdf','svg'):fig.savefig(DEST/f'duration_evidence.{ext}',dpi=350,facecolor='white')
    plt.close(fig)
    print('已输出持续能力敏感性图：四类客户、22档能力、5次和20次资料及完整信息参照。')


if __name__=='__main__':main()
