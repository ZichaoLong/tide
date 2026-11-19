"""Retire only this task's committed-validation scratch and audit final storage."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('cleanup_finish',HERE/'cleanup.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
evidence=ROOT/'docs/evidence/storage-finalization-20261119.json'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
committed=subprocess.check_output(['git','show',revision+':docs/evidence/storage-finalization-20261119.json'],cwd=ROOT)
assert committed==evidence.read_bytes()
review=json.loads(committed)
for service in review['verification']['services']:
    current=subprocess.check_output(['systemctl','--user','show',service['unit'],'-p','MainPID','-p','ControlGroup'],text=True)
    assert 'MainPID=0\n' in current and 'ControlGroup=\n' in current
    for row in service['status']['descendant_identities']:
        proc=Path('/proc')/str(row['pid'])/'stat'
        assert not proc.exists() or int(proc.read_text().rsplit(')',1)[1].split()[19])!=row['start_ticks']
scratch=[HERE/name for name in ('smoke-scratch','record-smoke','record-smoke-v2')]
inventory=[];regular=[];total=0
for directory in scratch:
    assert directory.is_dir() and not directory.is_symlink()
    total+=int(subprocess.check_output(['du','-s','-B1',str(directory)]).split()[0])
    for parent,dirs,files in os.walk(directory,followlinks=False):
        for name in dirs+files:
            p=Path(parent)/name;s=p.lstat();relative=str(p.relative_to(ROOT))
            row=dict(path=relative,bytes=s.st_size,allocated_bytes=s.st_blocks*512,mode=s.st_mode)
            if stat.S_ISLNK(s.st_mode):
                row['symlink_target']=os.readlink(p)
            elif stat.S_ISREG(s.st_mode):
                row['sha256']=helper.sha(p)
                regular.append(dict(path=relative,stat_guard=helper.guard(relative)))
            else:
                assert stat.S_ISDIR(s.st_mode)
            inventory.append(row)
live=helper.live_references(regular);assert not live['hits'],live['hits']
receipt=dict(state='reviewed_for_retirement',reviewed_commit=revision,files=inventory,
    allocated_bytes=total,live_observation=live,paths=[str(p.relative_to(ROOT)) for p in scratch])
helper.atomic(HERE/'scratch-retirement.json',receipt)
for row in regular:helper.same(row)
for directory in scratch:shutil.rmtree(directory)
helper.sync_directory(HERE)
receipt.update(state='retired',finished_utc=datetime.now(timezone.utc).isoformat())
helper.atomic(HERE/'scratch-retirement.json',receipt)
plan=json.loads((HERE/'plan.json').read_text())
for row in plan['protected']:
    helper.same(row);assert helper.sha(ROOT/row['path'])==row['sha256']
assert {p:helper.sha(ROOT/p) for p in plan['user_docs']}==plan['user_docs']
categories=defaultdict(int);counts=defaultdict(int);directories=0
for parent,dirs,files in os.walk(ROOT,followlinks=False):
    parent=Path(parent)
    directories+=parent.lstat().st_blocks*512
    for name in dirs+files:
        p=parent/name;s=p.lstat()
        if stat.S_ISDIR(s.st_mode):continue
        rel=p.relative_to(ROOT)
        if rel.parts[0]!='artifacts':key='outside_artifacts_files'
        elif p.suffix in {'.pt','.f32'}:key='retained_tensors'
        elif p.name.endswith('.json.gz'):key='archived_json'
        elif p.suffix=='.json':key='plain_json'
        else:key='other_artifact_files'
        categories[key]+=s.st_blocks*512;counts[key]+=1
categories['directory_blocks']=directories
du=int(subprocess.check_output(['du','-s','-B1',str(ROOT)]).split()[0])
assert sum(categories.values())==du,(sum(categories.values()),du)
quota=subprocess.check_output(['quota','-w'],text=True)
upstream=subprocess.check_output(['python','scripts/check_upstream.py'],cwd=ROOT,text=True)
final=dict(state='passed',observed_utc=datetime.now(timezone.utc).isoformat(),reviewed_commit=revision,
    repository_allocated_bytes=du,categories_allocated_bytes=dict(categories),file_counts=dict(counts),
    initial_inventory_allocated_bytes=review['initial_inventory_allocated_bytes'],
    net_freed_bytes=review['initial_inventory_allocated_bytes']-du,
    temporary_validation_bytes_retired=total,quota=quota,upstream=upstream,
    all_three_services_and_ten_recorded_descendants_terminal=True,
    protected_files_verified=len(plan['protected']),user_docs_unchanged=True,
    observation_boundary='After validation scratch retirement, before final documentation/commit writes; allocated blocks, not logical file bytes.')
helper.atomic(HERE/'final-audit.json',final)
print(json.dumps(final,indent=2))
