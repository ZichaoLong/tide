#!/usr/bin/env python3
"""Analytic local initialization scale; does not run or alter a model."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import digest,now

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'.local/foundation-src-74e7bbd/python/tidegraph/weighted'
nodes,weights=np.polynomial.hermite.hermgauss(80)
values=[]
for d in (32,128):
    sigma=.15;ffn=4*d;v=d*sigma*sigma
    normal=(2*v)**.5*nodes
    second=float((weights*(normal/(1+np.exp(-normal)))**2).sum()/np.pi**.5)
    values.append(dict(width=d,ffn_width=ffn,weight_std=sigma,
        normalized_input_projection_variance=v,expected_silu_squared=second,
        predicted_ffn_output_rms=float(np.sqrt(ffn*sigma*sigma*v*second))))
report=dict(schema='tide-hb-initialization-analysis-v1',at=now(),
    source_revision='74e7bbd62341d2ccd382194a2c52ad4fea033665',
    source_files=[dict(path=str(source/f),sha256=digest(source/f)) for f in ('tensors.py','kv.py','full_batch.py','config.py')],
    application_source=dict(path='tide_lab/language.py',sha256=digest(ROOT/'tide_lab/language.py')),
    analysis_source=dict(path='scripts/analyze_hb_initialization.py',sha256=digest(Path(__file__))),
    observations=dict(default_parameter_std=.15,application_embedding_std=.02,
        tide_final_readout_normalization=False,transformer_final_layer_norm=True,
        transformer_initial_position_rms=2**-.5,
        transformer_position_to_embedding_initial_rms_ratio=(2**-.5)/.02,
        gradient_probe='runs/hb-gradient-20261111/result.json'),
    assumptions='Single SwiGLU branch at initialization, RMS-normalized input with squared norm d; independent zero-mean Gaussian gate/up/down matrices of std sigma. Ignore epsilon correction and inter-layer correlations. This estimates one branch, not the whole HB state/gradient.',
    formula='v=d*sigma^2; Var[FFN_j]=d_ff*sigma^2*v*E[SiLU(N(0,v))^2]',
    method='80-point Gauss-Hermite quadrature in FP64; no training or sampling',
    values=values,ratio_128_over_32=values[1]['predicted_ffn_output_rms']/values[0]['predicted_ffn_output_rms'],
    interpretation='Fixed-std initialization is not width-stable. Combined with observed output RMS and large directional input gradients, it is a concrete recipe hypothesis; intervention training is still required for causality.',
    ownership='Application can reinitialize public trainable matrices with a named fan-in policy and add task readout normalization without replacing execution core. All roles/aliases must be explicit and initialized once.')
write_review_json(ROOT/'docs/evidence/hb-initialization-20261111.json',report)
print(json.dumps(values))
