#!/usr/bin/env python3
"""Standalone figure from already recorded validation/gradient observations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'docs/evidence/hb-learning-review-20261111.json').read_text())
chosen=[('scale-fc16-kv-quad','FC16 KV quad','#218c74'),
        ('scale-fc32-kv-quad','FC32 KV quad','#66a61e'),
        ('scale-hb40-kv-quad','HB40 KV quad','#2166ac'),
        ('scale-hb40-shannon','HB40 KV Shannon','#67a9cf'),
        ('scale-hb88-kv-quad','HB88 KV quad','#c51b7d'),
        ('scale-transformer-10m','Transformer 10M','#333333')]
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for key,label,color in chosen:
    matches=[r for r in data['scale'] if r['id']==key]
    row=max(matches,key=lambda r:r['progress'].get('tokens',0))
    validation={int(r['train/tokens']):r['validation/nll'] for r in row['curves']
                if r.get('validation/tokens')==32768 and r['train/tokens']<=1050000}
    x=sorted(validation)
    axes[0].plot(np.asarray(x)/1e6,[validation[t] for t in x],label=label,color=color,marker='o',markersize=3)
    grads=[r for r in row['curves'] if 'train/gradient_norm' in r and r['train/tokens']<=1050000]
    axes[1].plot([r['train/tokens']/1e6 for r in grads],[r['train/gradient_norm'] for r in grads],
                 label=label,color=color,marker='.',markersize=3)
axes[0].set(title='Same validation set: 32,768 target tokens',ylabel='NLL (nats / target token)')
axes[0].legend(fontsize=8,loc='upper right')
axes[1].set(title='Logged gradient norm before global clipping',ylabel='Parameter gradient L2 norm',yscale='log')
for ax in axes:
    ax.set(xlabel='Effective training targets (millions)',xlim=(0,1.06))
    ax.grid(alpha=.2)
fig.suptitle('v6, seed 11, d128 TIDE: negative scaling evidence is recipe- and exposure-specific',fontsize=11)
for suffix in ('svg','png'):
    fig.savefig(ROOT/f'docs/evidence/hb-learning-20261111.{suffix}',dpi=170,
                metadata={'Date':None} if suffix=='svg' else None)
