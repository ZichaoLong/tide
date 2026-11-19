"""Finite, manually reviewed submissions using the repository's frozen launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.job import atomic


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def result(path):
    data = read(path)
    if not data:
        return None
    part = data.get('subjects', {}).get('tide', data)
    samples = part.get('observations', [])
    summary = dict(state=data['state'], stage=data.get('last_stage'), error=data.get('error'))
    if samples:
        summary.update(samples=len(samples), rate=statistics.median(s['stable_decode']['positions_per_second'] for s in samples),
            step_ms=statistics.median(s['stable_decode']['seconds']/15*1000 for s in samples),
            allocated=max(v['peak_allocated_bytes'] for s in samples for v in s['allocator'].values())/2**30,
            reserved=max(v['peak_reserved_bytes'] for s in samples for v in s['allocator'].values())/2**30)
    return summary


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('submit')
    s.add_argument('kind', choices=('tide','tf5','tf32','tf48'))
    s.add_argument('scale', choices=('9b','17b'))
    s.add_argument('cards', type=int, choices=(1,2,4))
    s.add_argument('batch', type=int)
    s.add_argument('--slot', type=int, choices=(0,1,2), required=True)
    s.add_argument('--confirm', action='store_true')
    s.add_argument('--reason', required=True)
    s.add_argument('--allocated-forecast', type=float)
    s.add_argument('--reserved-forecast', type=float)
    sub.add_parser('summary')
    args = p.parse_args()
    records = read(BASE/'submissions.json') or []
    if args.action == 'summary':
        rows=[]
        for r in records:
            run=ROOT/'artifacts'/r['name']
            lifecycle=read(run/'status.json')
            observed=result(run/'inference.json')
            row={k:r[k] for k in ('name','kind','scale','cards','batch','slot','confirm')}
            row.update(lifecycle=None if lifecycle is None else lifecycle['state'], result=observed)
            rows.append(row)
        atomic(BASE/'progress.json',dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),runs=rows))
        for row in rows:
            obs=row['result'] or {}
            print(row['name'], row['lifecycle'], obs.get('stage'),
                  'samples='+str(obs.get('samples',0)),
                  'rate='+str(round(obs.get('rate',0),4)),
                  'GiB='+str(round(obs.get('allocated',0),3))+'/'+str(round(obs.get('reserved',0),3)),
                  obs.get('error') or '')
        return
    if not 1<=args.batch<=4096:
        p.error('batch outside finite plan')
    if (args.scale,args.cards) not in [('9b',1),('17b',2),('17b',4)]:
        p.error('scale/device pair outside plan')
    for r in records:
        lifecycle=read(ROOT/'artifacts'/r['name']/'status.json')
        if r['slot']==args.slot and (not lifecycle or lifecycle['state'] in ('starting','running','prepared')):
            p.error('slot still owned by '+r['name'])
    same=[r for r in records if (r['kind'],r['scale'],r['cards'],r['confirm'])==(args.kind,args.scale,args.cards,args.confirm)]
    if len(same)>=(2 if args.confirm else 8):
        p.error('series point limit reached')
    mode='confirm' if args.confirm else 'screen'
    name=f'cap-{args.scale}-{args.kind}-c{args.cards}-b{args.batch}-{mode}-v1'
    if any(r['name']==name for r in records):
        p.error('point already submitted')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if source != '98210ae3' and not source.startswith('98210ae3'):
        p.error('review source identity before submission')
    devices=','.join(f'npu:{i}' for i in range(args.cards))
    if args.kind=='tide':
        command=['python','-m','benchmarks.inference','--config',f'configs/scale/hb-{args.scale}.json',
            '--config-overrides',json.dumps(dict(region_batch_size=4,ssd_scan_backend='doubling',ssd_summary_backend='prefix')),
            '--subjects','tide','--no-cold','--reference-output','{run}/reference.json']
        if args.cards>1:
            command += ['--placement','region','--region-concurrency','per_device']
    else:
        command=['python','-m','benchmarks.transformer_depth','--parameters',str(8778358224 if args.scale=='9b' else 16707220176),
            '--layers',args.kind[2:],'--chunk','256']
        # Exact old reference is reusable only at its original batch/schedule.
        if args.batch==8:
            suffix='-four' if args.cards==4 else ''
            reference=ROOT/'artifacts'/f'batch-hb{args.scale}-b8{suffix}-fp32-v1/reference.json'
            if reference.exists(): command += ['--target-reference',str(reference)]
    command += ['--devices',devices,'--transport','host_stage','--precision','fp32','--batch',str(args.batch),
        '--prefill','128','--prefill-chunk','32','--decode','16','--samples','5' if args.confirm else '2',
        '--warmup','2' if args.confirm else '1','--threads','4','--allocated-budget-gib','48',
        '--reserved-budget-gib','56','--output','{run}/inference.json']
    affinity=f'{80+4*args.slot}-{83+4*args.slot}'
    argv=[sys.executable,str(ROOT/'scripts/launch.py'),'--name',name,'--revision',source,'--threads','4','--affinity',affinity,
        '--npu-count',str(args.cards),'--seconds','2700','--rss-gib','112' if args.scale=='9b' else '192',
        '--queue-seconds','1800','--',*command]
    record=dict(vars(args),name=name,source=source,recorded_at_utc=datetime.now(timezone.utc).isoformat(),argv=argv,
        manager_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),state='prepared')
    records.append(record);atomic(BASE/'submissions.json',records)
    launched=subprocess.run(argv,cwd=ROOT,text=True,capture_output=True)
    record.update(state='submitted' if launched.returncode==0 else 'launch_failed',returncode=launched.returncode,
        stdout=launched.stdout,stderr=launched.stderr)
    atomic(BASE/'submissions.json',records)
    print(json.dumps(dict(name=name, state=record['state'], source=source,
                         reason=args.reason, returncode=launched.returncode)))
    if launched.returncode:
        print(launched.stdout, launched.stderr)
    if launched.returncode: raise SystemExit(launched.returncode)


if __name__=='__main__':
    main()
