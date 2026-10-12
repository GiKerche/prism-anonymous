#!/usr/bin/env python3
"""Regenerate the four added figures from the supplied numerical exports.
Requires NumPy and Matplotlib; no external files, network, or model calls.
Run from any directory: python make_reviewer_figures.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
STAGE=Path(__file__).resolve().parent
MAT=STAGE
KEYS=['source_m1','source_m3','source_m5','receiver_m1','receiver_m3','receiver_m5']
LANGS=['pt','en','es','zh','hi']
def read(p): return json.loads(p.read_text())
def results(key): return read(STAGE/'data/current_final_results.json')[key]
MODEL={'source_m1':'Qwen 9B','source_m3':'Ministral 8B','source_m5':'Llama 8B',
       'receiver_m1':'Qwen 27B','receiver_m3':'Ministral 14B','receiver_m5':'Llama 70B'}

def plots():
    (MAT/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    plus='#B64926';minus='#256899';markers={'native':'o','shadow':'s','transported_native':'D'}
    from make_primary_benchmark_figure import main as primary_figure
    with plt.rc_context({'savefig.bbox': None}):
        primary_figure()
    curves=read(STAGE/'data/development_curves.json')
    fig,axes=plt.subplots(2,3,figsize=(7,4.2),sharex=True)
    for ax,key in zip(axes.flat,KEYS):
        d=curves[key]
        for branch in sorted({v['source'] for v in d['curve']}):
            for sign,color in [(1,plus),(-1,minus)]:
                a=sorted([v for v in d['curve'] if v['source']==branch and np.sign(v['alpha'])==sign],key=lambda v:v['alpha'])
                linestyle='--' if branch=='shadow' else '-'
                ax.plot([v['alpha'] for v in a],[v['call_proxy']['delta_pp'] for v in a],linestyle,color=color,lw=1)
                for v in a:
                    chosen=d['selected_alphas'][branch+('_plus' if sign>0 else '_minus')]
                    chosen=chosen['alpha'] if isinstance(chosen,dict) else chosen
                    ax.scatter(v['alpha'],v['call_proxy']['delta_pp'],marker='*' if v['alpha']==chosen else 'o' if v['admissible'] else 'x',
                        s=45 if v['alpha']==chosen else 12,color=color,zorder=3)
        ax.axhline(0,color='.6',lw=.6);ax.axvline(0,color='.6',lw=.6)
        ax.set_xscale('symlog',linthresh=.25);ax.set_xticks([-8,-1,0,1,8],labels=['−8','−1','0','1','8'])
        ax.set_title(MODEL[key]);ax.grid(alpha=.12)
    for ax in axes[-1]:ax.set_xlabel('Signed dose (Validation II)')
    for ax in axes[:,0]:ax.set_ylabel('Action-call proxy Δ (pp)')
    fig.tight_layout();fig.savefig(MAT/'figures/reviewer_dose_response.pdf');plt.close(fig)
    scarcity=read(STAGE/'data/scarce_labels.json')['rows']
    fig,axes=plt.subplots(1,3,figsize=(7,2.15),sharey=True)
    for ax,slot in zip(axes,['m1','m3','m5']):
        for precision,color in [('bf16','#256899'),('nf4','#B64926')]:
            a=[v for v in scarcity if v['slot']==slot and v['precision']==precision]
            ax.plot([v['families'] for v in a],[v['undefined_percent'] for v in a],'o-',ms=3,color=color,label=precision.upper())
        ax.set(title=MODEL['source_'+slot],xlabel='Construction families',xticks=[2,8,16,24,33],ylim=(-2,85));ax.grid(alpha=.15)
    axes[0].set_ylabel('Undefined fits (%)');axes[-1].legend(frameon=False)
    fig.tight_layout();fig.savefig(MAT/'figures/reviewer_label_scarcity.pdf');plt.close(fig)
    geometry=read(STAGE/'data/language_geometry.json')['rows']
    fig,axes=plt.subplots(2,3,figsize=(7,4.3))
    for ax,row in zip(axes.flat,sorted(geometry,key=lambda v:(v['estimator']=='shadow',v['slot']))):
        a=np.array(row['cosine']);im=ax.imshow(a,vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=range(5),xticklabels=LANGS,yticks=range(5),yticklabels=LANGS,title=MODEL['source_'+row['slot']]+' / '+row['estimator'])
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=6,color='white' if a[i,j]>.7 else '.2')
    fig.tight_layout();fig.savefig(MAT/'figures/reviewer_language_cosines.pdf');plt.close(fig)

if __name__=='__main__': plots()
