#!/usr/bin/env python3
"""Export reviewed v7 observations as a static learning/capacity figure."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'docs/evidence/direct-v7-observations.json').read_text())
cases=report['cases'];main=[c for c in cases if c['phase']=='main']
if len(main)!=10 or any(c['state']!='completed' for c in main):
    raise ValueError('the 10 main trajectories must be complete before exporting this final figure')
colors=dict(fc16='#00876c',hb40='#297cb3',hb88='#8451a1',tf12='#dd8a22',tf28='#c14953')
labels=dict(fc16='Direct FC16',hb40='Direct HB40',hb88='Direct HB88',tf12='Transformer 12',tf28='Transformer 28')
plt.rcParams.update({'font.size':10,'svg.hashsalt':'direct-v7-reviewed','axes.spines.top':False,
                     'axes.spines.right':False,'savefig.facecolor':'white'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,4.9),gridspec_kw={'width_ratios':[1.35,1]})
for c in main:
    model=c['id'].split('-')[0];seed=int(c['id'].rsplit('s',1)[1]);points={}
    for entry in [v for v in cases if v['phase']=='screen' and v['id']==c['id']]+[c]:
        for v in entry['validations']:
            if v['state']=='completed' and v['validation_tokens']==32768:
                points[v['training_tokens']]=v['nll']
    xs=sorted(points)
    ax.plot([x/1e6 for x in xs],[points[x] for x in xs],color=colors[model],
            linestyle='-' if seed==11 else '--',linewidth=1.6,alpha=.9,
            label=labels[model] if seed==11 else None)
for model in colors:
    rows=[c for c in main if c['id'].startswith(model+'-')]
    ys=[next(v['nll'] for v in c['validations'] if v['record'].endswith('validation-4194304.json')) for c in rows]
    x=rows[0]['parameters']/1e6;y=statistics.mean(ys)
    bx.errorbar([x],[y],yerr=[[y-min(ys)],[max(ys)-y]],color=colors[model],marker='o',capsize=4,markersize=6)
    bx.annotate(labels[model],(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
counts=[c['result']['nll']['bigram'] for c in report['analytic_controls'] if c['generation']=='v7-count-controls-r2']
if len(counts)==2:
    value=statistics.mean(counts)
    bx.axhline(value,color='#666666',ls=':',lw=1.3,label='4M train-count bigram (2-seed mean)')
    bx.legend(loc='lower left',frameon=False,fontsize=8)
ax.set(xlabel='Effective training targets (million)',ylabel='Validation NLL (nats / target)',
       title='Learning at fixed batch 32 and context 128',xlim=(0,4.3))
ax.legend(ncol=2,frameon=False,fontsize=8,loc='upper right')
ax.text(.02,.04,'Solid: seed 11; dashed: seed 22',transform=ax.transAxes,fontsize=8)
bx.set(xlabel='Trainable parameters (million)',ylabel='Validation NLL at 4,194,304 targets',
       title='Capacity points: means and range of two seeds',xlim=(1,27))
for axis in (ax,bx):axis.grid(alpha=.2)
fig.suptitle('Direct dense graph language study v7 — different function from TIDE',fontsize=13)
fig.text(.5,.01,'Same 32,768 validation targets; validation also used for recipe selection. Test sealed. Dense execution; no sparse-routing claim.',
         ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.94))
fig.savefig(ROOT/'docs/evidence/direct-v7-learning.svg',metadata={'Date':None})
fig.savefig(ROOT/'docs/evidence/direct-v7-learning.png',dpi=150,metadata={'Software':'Matplotlib; scripts/plot_direct_v7.py'})
plt.close(fig)
