"""核心敏感性主图：三个资料条件、四类客户、年度风险与历史覆盖。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from run_core_sensitivity import DEST,LABEL,NAMES,read


def main():
    table=pd.read_csv(DEST/'sensitivity_summary.csv')
    c=read(DEST/'sensitivity_params.json')['parameters'];nomination=c['nomination_mw']
    percent=100/nomination  # 按当前申报量换算容量百分比，保留申报量参数化。
    panels=[('年度风险要求',[('tail_0.9','90%：最严重10%'),('central','95%：最严重5%'),('tail_0.99','99%：最严重1%')]),
        ('历史核验的覆盖要求',[('coverage_0.9','90%'),('central','95%'),('coverage_0.99','99%')])]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':7.0,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
        'text.color':'#293744','axes.labelcolor':'#293744','xtick.color':'#61707C','ytick.color':'#293744',
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.65,
        'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})
    gray,teal,gold='#8C969F','#3C8987','#AE8448'
    fig=plt.figure(figsize=(7.2,5.6))  # 双栏183mm；保留可编辑矢量输出。
    fig.text(.06,.955,'年度风险与历史核验怎样影响认证容量',fontsize=12,weight='bold')
    fig.text(.06,.916,f'正态参照与三户拟合，申报均为{nomination:g}兆瓦；5次、20次历史及完整信息',fontsize=7.4,color='#65717B')
    handles=[Line2D([0],[0],marker='o',mfc='white',mec=gray,color=gray,lw=0,markersize=4,label='5次历史'),
        Line2D([0],[0],marker='o',mfc=teal,mec='white',color=teal,lw=0,markersize=4.5,label='20次历史'),
        Line2D([0],[0],marker='|',color=gold,lw=0,markersize=8,label='完整信息')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.52,.875),ncol=3,fontsize=7.2,columnspacing=2.7)
    source=[]
    for row,(family,levels) in enumerate(panels):
        header=.824 if row==0 else .466
        fig.text(.06,header,f'{"a" if row==0 else "b"}  {family}',fontsize=8.9,weight='bold')
        for col,(key,title) in enumerate(levels):
            left=.145+col*.277;bottom=.583 if row==0 else .225
            ax=fig.add_axes([left,bottom,.245,.190])
            for i,name in enumerate(NAMES):
                label=LABEL[name];selected=table[(table['客户']==label)&(table['情景']==key)]
                if len(selected)!=1:raise RuntimeError(f'核心主图缺少完整结果：{label}/{key}')
                r=selected.iloc[0];y=3-i
                lo=np.asarray([r['五次容量下界_兆瓦'],r['二十次容量下界_兆瓦'],r['完整信息下界_兆瓦']])*percent
                hi=np.asarray([r['五次容量上界_兆瓦'],r['二十次容量上界_兆瓦'],r['完整信息上界_兆瓦']])*percent
                mid=(lo+hi)/2
                ax.axhline(y,color='#EFF2F4',lw=.6,zorder=0)
                ax.plot(mid[:2],[y,y],color=teal,lw=1.4,zorder=1)
                ax.plot(mid[1:],[y,y],color='#C7CDD1',lw=.8,ls=(0,(2,2)),zorder=1)
                ax.errorbar(mid[0],y,xerr=[[mid[0]-lo[0]],[hi[0]-mid[0]]],fmt='o',mfc='white',mec=gray,
                    ms=4,ecolor=gray,lw=.6,capsize=1.6,zorder=3)
                ax.errorbar(mid[1],y,xerr=[[mid[1]-lo[1]],[hi[1]-mid[1]]],fmt='o',mfc=teal,mec='white',
                    ms=4.5,ecolor=teal,lw=.6,capsize=1.6,zorder=4)
                ax.plot(mid[2],y,marker='|',color=gold,ms=8,mew=1.2,zorder=3)
                ax.text(mid[1]+.5,y+.14,f'{mid[1]:.1f}',color=teal,fontsize=6.1,ha='left',va='bottom')
                source.append({'比较':family,'档位':title,**r.to_dict()})
            ax.set(xlim=(0,34),ylim=(-.65,3.65));ax.set_xticks([0,10,20,30])
            ax.set_yticks([3,2,1,0],[LABEL[n] for n in NAMES] if col==0 else [])
            ax.tick_params(axis='y',length=0,pad=5,labelsize=7.2);ax.tick_params(axis='x',length=2.5,pad=4,labelsize=6.5)
            ax.set_title(title,fontsize=8,pad=8,weight='bold')
    fig.text(.54,.145,'净认证容量占申报量（%）',ha='center',fontsize=8)
    fig.text(.06,.094,'实线长度：增加历史后多认的容量。虚线长度：距完整信息容量还差多少。误差线为计算上下界。',fontsize=6.7,color='#65717B')
    fig.text(.06,.061,'上排固定历史覆盖95%；下排固定年度风险95%。客户与原始15分钟记录在所有档位保持相同。',fontsize=6.5,color='#65717B')
    fig.text(.06,.028,f'其余条件统一：正常负荷{c["load_mw"]:g}兆瓦、真实持续能力{c["true_duration_hours"]:g}小时、年度缺供上限{c["risk_budget_mwh"]:g}兆瓦时及同一调用方案池。',fontsize=6.2,color='#65717B')
    name='sensitivity_risk'
    for ext in ('png','pdf','svg'):fig.savefig(DEST/(name+'.'+ext),dpi=350,facecolor='white')
    plt.close(fig)
    pd.DataFrame(source).to_csv(DEST/'figure_data.csv',index=False,encoding='utf-8-sig')
    (DEST/(name+'_caption.md')).write_text('年度风险与历史核验覆盖下的认证容量。每行代表同一类虚拟客户。'
        '空心点、实心点及竖线分别表示5次历史、20次历史和完整信息，连接线显示三个容量位置。'
        '容量均扣除同标准无新增客户基准，并除以申报量。上排改变年度风险尾部，'
        '下排改变整批历史核验的覆盖要求，误差范围及历史功率预测界同时更新。'
        '历史覆盖要求指核验规则保留整批真实误差和功率路径的目标概率。'
        '两种资料保留完整15分钟读数并采用同一932项统计全集，调用方案、年度场景与客户真实能力固定。'
        '误差线为数值上下界。\n',encoding='utf-8')
    print('核心参数主图已输出。',flush=True)


if __name__=='__main__':main()
