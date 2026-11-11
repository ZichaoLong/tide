#!/usr/bin/env python3
"""Static proposal inventory only: no training, device allocation or job configs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tidegraph.planning.recipes import prepare
from tide_lab.campaign_config import local_candidates
from tide_lab.direct_semantic import DirectSemanticLM
from tide_lab.language import SmallTransformer
from tide_lab.records import now
from scripts.resolve_direct_v8 import recipe
from scripts.review_io import write_review_json


def inspect():
    candidates = {r['id']: r for r in local_candidates()}
    shapes = []
    for topology, depths, ids in (
        ('fc', (4, 8, 16), ('send-direct_attention', 'temperature1')),
        ('hb', (2, 3, 4), ('send-pooled_attention', 'temperature1')),
    ):
        for depth in depths:
            for candidate in ids:
                shapes.append((topology, depth, 128, 4, candidate))
    shapes += [('hb', 2, w, 4, 'send-pooled_attention') for w in (192, 256)]
    shapes += [('hb', 2, 64, 32, 'temperature1'),
               ('hb', 3, 64, 32, 'temperature1'),
               ('hb', 2, 32, 128, 'temperature1')]
    rows = []
    for topology, depth, width, region_width, candidate in shapes:
        spec = recipe(candidates[candidate], topology, width)
        spec.update(depth=depth, region_width=region_width)
        if width == 256:
            spec['memory_overrides']['ffn_width'] = 1280
        plan, _ = prepare(spec, samples=1, positions=1, seed=11)
        graph = plan['graph']
        with torch.device('meta'):
            model = DirectSemanticLM(plan)
        rows.append(dict(
            candidate=candidate, recipe=spec, nodes=graph['nodes'],
            regions=graph['regions'],
            rank_depth=max(graph['ranks']) - min(graph['ranks']) + 1,
            parameters=sum(p.numel() for p in model.parameters()),
            max_source_slots=max(map(len, model.connections)),
            plan_sha256=plan['sha256'],
        ))
    transformer_rows = []
    for width, layers in ((240, 6), (336, 6), (416, 8),
                          (384, 6), (400, 12), (512, 16)):
        with torch.device('meta'):
            model = SmallTransformer(width=width, layers=layers, heads=4, ffn=4*width)
        transformer_rows.append(dict(width=width, layers=layers, heads=4, ffn=4*width,
                                     parameters=sum(p.numel() for p in model.parameters())))
    paths = ('scripts/review_scale_activity_plan.py', 'scripts/resolve_direct_v8.py',
             'tide_lab/campaign_config.py', 'tide_lab/direct_semantic.py',
             'tide_lab/direct_semantic_memory.py', 'tide_lab/direct_semantic_ops.py',
             'tide_lab/language.py')
    return dict(
        schema='tide-scale-activity-static-proposal-v1', observed_at=now(),
        kind='design_only', launchable=False,
        source_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        foundation_selection=json.loads((ROOT/'dependencies/foundation-source.json').read_text()),
        torch_version=torch.__version__, vocabulary=4096,
        scope='Public planning metadata and meta-device parameter shapes only; no forward/backward, training, allocation or runtime qualification.',
        limits=['Parameter counts describe the current direct implementation, not an executed new scale point.',
                'Rank depth counts regional Full stages, not Transformer blocks or equal FLOPs.',
                'Wider regions change locality and source aggregation cost; no low-activity runtime measured.',
                'Future architecture edits require a new inventory and frozen runtime configuration.'],
        rejected_designs=[dict(width=272, reason='Installed public recipe rejects this content width; use legal width256 with FFN1280 for the large shallow-wide control.')],
        direct=rows, transformers=transformer_rows,
    )


if __name__ == '__main__':
    report = inspect()
    write_review_json('docs/evidence/scale-activity-v9-static.json', report)
    print(json.dumps(dict(direct=len(report['direct']), transformers=len(report['transformers']),
                         kind=report['kind'], launchable=report['launchable'])))
