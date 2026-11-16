"""Regenerate four publication figures from frozen flight-level evidence.

Figure 1 is a code-derived schematic, not a numerical result. Figures 2--4
use the original 864 condition records (not the exploratory extra models).
"""
from pathlib import Path
import collections,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path('.');OUT=ROOT/'results/publication_figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors=['#b65b17','#12619a'];conditions=['unmonitored','monitored']
def save(fig,n):
    for ext in ['pdf','svg','png','tiff']:fig.savefig(OUT/f'Figure_{n}.{ext}',dpi=600,facecolor='white',**({'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}))
    plt.close(fig)
def load(split):
    prefix='final_sealed_evaluation_v2_1b' if split=='final_sealed_test' else 'final_nonsealed_evaluation_v2_1b'
    return [json.loads(l) for l in (ROOT/'results'/prefix/split/'flight_records.jsonl').read_text().splitlines()]
evidence={};fig,ax=plt.subplots(figsize=(7,3.6));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
def box(x,y,w,h,label):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',fc='#edf2f6',ec='#333333',lw=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
def arrow(a,b,dashed=False):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.2,'linestyle':'--' if dashed else '-'})
box(.2,2.8,2,1,'Low-order\nsimulator');box(2.9,2.8,2,1,'Observation +\nnormalization');box(5.6,2.8,1.7,1,'Neural\npolicy');box(8,2.8,1.7,1,'Runtime\nmonitor');box(5.6,.6,1.7,1,'Deterministic\nfallback');box(8,.6,1.7,1,'Selected\ncommand')
arrow((2.2,3.3),(2.9,3.3));arrow((4.9,3.3),(5.6,3.3));arrow((7.3,3.3),(8,3.3));arrow((8.85,2.8),(8.85,1.6));arrow((7.3,1.1),(8,1.1),True);arrow((3.9,2.8),(6.45,1.6));ax.plot([8.85,8.85,1.2,1.2],[.6,.25,.25,2.75],color='#333333',lw=1.2);arrow((1.2,2.2),(1.2,2.8));ax.text(8.2,2.2,'accept /\nproject',fontsize=8,ha='center');ax.text(1.2,4.25,'Scenario + failures',ha='center',fontsize=9);arrow((1.2,4.08),(1.2,3.8));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98);save(fig,1)
splits=['challenge','ood_stress','final_sealed_test'];records={s:load(s) for s in splits}
fig,ax=plt.subplots(figsize=(7,3.6));x=np.arange(3)
for j,c in enumerate(conditions):
    rate=[];lo=[];hi=[]
    for s in splits:
        rs=[r for r in records[s] if r['controller_condition']==c];n=len(rs);k=sum(r['strict_recovery_success'] for r in rs);p=k/n;z=1.959963984540054;den=1+z*z/n;mid=(p+z*z/(2*n))/den;rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        rate.append(p*100);lo.append((p-mid+rad)*100);hi.append((mid+rad-p)*100);evidence[f'figure2_{s}_{c}']={'n':n,'success':k,'wilson95':[mid-rad,mid+rad]}
    ax.errorbar(x+(j-.5)*.14,rate,yerr=[lo,hi],fmt='o',capsize=5,color=colors[j],label=c.capitalize(),ms=6)
ax.set(xticks=x,xticklabels=['Challenge','Simulator stress','Final test'],ylabel='Strict recovery (%)',ylim=(0,106));ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right',frameon=False);fig.tight_layout();save(fig,2)
fig,(a,b)=plt.subplots(1,2,figsize=(7,3.8),gridspec_kw={'width_ratios':[1.2,1]})
for j,c in enumerate(conditions):
    vv=[[r['strict_recovery_time_seconds'] for r in records['final_sealed_test'] if r['controller_condition']==c and r['difficulty_tier']==t] for t in range(6)];med=np.array([np.median(v) for v in vv]);low=med-np.array([min(v) for v in vv]);high=np.array([max(v) for v in vv])-med
    a.errorbar(np.arange(6)+(j-.5)*.16,med,yerr=[low,high],fmt='o',color=colors[j],capsize=3,label=c.capitalize(),ms=4)
    b.plot(np.arange(1,6)+(j-.5)*.16,med[1:],'o',color=colors[j],ms=4)
    evidence[f'figure3_{c}']=[{'tier':t,'median':float(med[t]),'minimum':min(v),'maximum':max(v)} for t,v in enumerate(vv)]
a.set(title='(a) Medians and ranges',xlabel='Scenario-complexity tier',ylabel='Recovery onset delay (s)',xticks=range(6),ylim=(-.04,1.5));b.set(title='(b) Median detail',xlabel='Scenario-complexity tier',xticks=range(1,6),ylim=(0,.065),yticks=[0,.025,.05]);a.grid(axis='y',alpha=.2);b.grid(axis='y',alpha=.2);fig.legend(*a.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False);fig.tight_layout(rect=(0,.12,1,1));save(fig,3)
counter=collections.Counter()
for r in records['final_sealed_test']:
    if r['controller_condition']=='monitored':counter.update(r['intervention_taxonomy']['violated_property_counts'])
items=sorted(counter.items(),key=lambda kv:kv[1]);fig,ax=plt.subplots(figsize=(7,3.6));ax.barh([k.replace('_',' ') for k,v in items],[v for k,v in items],color=colors[1]);ax.set(xlabel='Recorded rule events',xlim=(0,max(counter.values())*1.18));ax.grid(axis='x',alpha=.2)
for i,(k,v) in enumerate(items):ax.text(v+2,i,str(v),va='center',fontsize=9)
fig.tight_layout();save(fig,4);evidence['figure4']=dict(counter)
(OUT/'plotted_values.json').write_text(json.dumps(evidence,indent=2));print('Four figures regenerated in',OUT)
