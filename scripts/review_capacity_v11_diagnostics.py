#!/usr/bin/env python3
"""Summarize the fixed diagnostic window; never writes workload records."""
import json,statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import digest,now


def main():
    root=Path(__file__).resolve().parents[1];table=json.loads((root/'configs/studies/language/capacity-v11/diagnostics.json').read_text());rows=[]
    for c in table['cases']:
        path=root/'runs'/('v11-diag-'+c['id'])/'result.json'
        r=json.loads(path.read_text());v=dict(id=c['id'],state=r['state'],record=str(path.relative_to(root)),sha256=digest(path))
        if r['state']=='completed':
            obs=list(r['observations'].values());cov=[s['coverage_summary'] for s in obs]
            v.update(observer_output_equality=r['observer_output_equality'],valid_targets=r['valid_targets'],train_window_nll=r['train_window_nll'],
                average_zero_units=statistics.mean(s['zero'] for s in cov) if cov else None,
                average_effective_units=statistics.mean(s['effective_units'] for s in cov) if cov else None,
                average_max_share=statistics.mean(s['max_share'] for s in cov) if cov else None,
                regions_or_layers=len(obs),window_sequences=512,update_global_batch=64,
                parameter_history_exposures=list(r['historical_changes']),projection='runs/v11-diag-'+c['id']+'/projection.json')
            if c['id'].startswith('hb'):
                norms={role:[z['gradient']['zero_first_axis'] for name,z in r['parameters'].items() if name.startswith('memories.') and name.endswith('.'+role+'.weight') and z['gradient'] is not None] for role in ('query','read_q','ffup')}
                v['mean_zero_gradient_nodes_in_one_update']={k:statistics.mean(ns) for k,ns in norms.items()}
                work=r['work'];pos=work['work/prediction_positions']
                v['logical_parameter_budget']={role:work['work/'+role+'_parameter_uses']/pos for role in ('full','receive','shared')}
                v['full_all']=work['work/full']/work['work/node_slots'];v['receipt_all']=work['work/candidates']/work['work/node_slots']
        else:v['error']=r.get('error')
        rows.append(v)
    report=dict(schema='tide-capacity-v11-diagnostic-review-v1',observed_at=now(),rows=rows,
        reviewed_scope='Fixed train-window Full/assignment coverage and one-update gradient support; all 16 observers preserve same-device output exactly. Historical parameter changes include weight decay/Adam memory, not evidence of task gradients at every node.',
        decision='Choose only candidate-aware routing load-bias family, rates.001/.01,d64 R64,seeds11/22. Do not also modify Read/FFN budget.',
        rationale='HB R64 use concentrated in both widths and both large-width seeds; large MoE E64 has broad coverage. This separates HB opportunity bottleneck from exposure and does not establish a unique cause.',
        counterevidence='Small MoE E64 also shows imbalance; larger MoE has broad use yet weak marginal E20-toE64 gain. More exposure and task difficulty remain plausible.',
        limits=['512 sequences, not cumulative lifetime use','no diagnostic optimizer update is training ancestry','near-equal active parameter totals hide Full/receipt/shared decomposition','no fresh confirmation was used to select intervention','no foundation bug or universal capacity-scaling conclusion'])
    write_review_json('docs/evidence/capacity-v11-diagnostics.json',report)
    print(json.dumps(dict(completed=sum(r['state']=='completed' for r in rows),total=len(rows))))

if __name__=='__main__':main()
