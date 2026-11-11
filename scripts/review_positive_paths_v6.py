#!/usr/bin/env python3
"""Read-only lifecycle/evidence consolidation; never writes run/job records."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.campaign_status import status
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import digest,now


def main():
    root=Path(__file__).resolve().parents[1]
    rows=[];seconds=0.;size=0
    for name in ['direct-received','regional-total-bern','constrained-bern','direct-boundary-context']:
        run=root/'runs'/('v6-positive-'+name)
        job=root/'artifacts/jobs'/('v6-positive-'+name)
        life=inspect(job/'job.json');result=json.loads((run/'result.json').read_text())
        if result['state']!='completed' or not life['termination_verified_now']:
            raise ValueError('diagnostic incomplete or live: '+name)
        projection=run/('projection-retry1.json' if name=='direct-received' else 'projection.json')
        lens=json.loads(projection.read_text())
        if lens.get('status')!='verified' or lens.get('readback_matches') is not True:
            raise ValueError('Lens readback not verified: '+name)
        elapsed=life['state']['elapsed_seconds'];seconds+=elapsed
        size+=sum(p.stat().st_size for base in (run,job) for p in base.rglob('*') if p.is_file())
        artifacts={p.name:digest(p) for p in run.iterdir() if p.suffix in ('.json','.pt','.jsonl')}
        rows.append(dict(id='tl-v6-positive-'+name,record=str((run/'result.json').relative_to(root)),
            job_record=str((job/'job.json').relative_to(root)),unit=life['systemd']['Id'],
            state=result['state'],termination_verified=life['termination_verified_now'],
            elapsed_seconds=elapsed,source=result.get('diagnostic_source',result.get('identity')),
            original_run=result.get('original_run'),original_error=result['error'],
            phase=result.get('phase','captured forward'),progress=result.get('progress'),
            scalar_replay_checks=result.get('observations',[]),
            observed_same_exception=result.get('observed_forward_same_exception',result.get('matching_original_exception')),
            axes=result.get('axes',dict(execution='original exception captured; no update')),
            artifact_sha256=artifacts,lens=dict(report=str(projection.relative_to(root)),
                status=lens['status'],readback_matches=True,event_count=lens['event_count'],
                destination_name=lens['destination_name'])))
    campaign=status(root);campaign.pop('cases')
    evidence=root/'docs/evidence/positive-path-v6-local-analysis.json'
    analysis=json.loads(evidence.read_text())
    if analysis['script_sha256']!=digest(root/'scripts/probe_positive_formulas_v6.py'):
        raise ValueError('local probe evidence does not match current script')
    write_review_json(root/'docs/evidence/positive-path-v6-review.json',dict(
        schema='tide-positive-path-review-v1',reviewed_at=now(),study_id='TL-POSITIVE-v6',
        scope='human-reviewed numerical/semantic failure localization; no repair or learning qualification',
        foundation='74e7bbd62341d2ccd382194a2c52ad4fea033665',
        mathematics='0770593838a37c790a568226930567ed60f99b83',
        source_policy='original fixed application/dependencies; separate diagnostics; foundation unchanged',
        runs=rows,allocated_cpu_core_hours=seconds/3600,new_run_job_bytes=size,
        resource_scope='service allocation includes startup/observation; short local probes and source snapshots separate',
        local_analysis_sha256=digest(evidence),campaign_terminal_snapshot=campaign,
        findings=dict(constrained_bern='representable response prematurely zeroed',
            regional_total_bern='final response below FP32 range; sigmoid stabilization alone insufficient',
            direct_received='positive representable caps and shares multiply below FP32 range',
            silent_witnesses=['strict aggregate content loss','pre-Describe precision cast changes active set',
                              'Bern saturated-tail VJP loss']),
        upstream_request='not submitted; repair design awaits discussion',
        limitations=['local Decimal/API probes do not certify repaired gradients or long trajectory',
                     'static map is not an exhaustive backend/candidate audit',
                     'attention/gate/epsilon formula examples do not establish actual training occurrence',
                     'direct global node/time and full upstream mass lineage remain unrecorded'],
        recording_failure=dict(record='runs/v6-positive-direct-received/projection.json',
            reason='console log name collided with recorder-owned projection.log; failed before events; new report retry verified')))
    print(json.dumps(dict(terminal_runs=len(rows),allocated_cpu_core_hours=seconds/3600,
                          new_run_job_bytes=size,all_lens_verified=True)))


if __name__=='__main__':main()
