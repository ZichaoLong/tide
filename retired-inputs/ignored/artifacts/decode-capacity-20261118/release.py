"""Check terminal ownership and preserved inputs after this finite search."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.job import atomic


def read(path):
    return json.loads(path.read_text())


def identity(path):
    content = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest())


def command(argv):
    return subprocess.check_output(argv, cwd=ROOT, text=True)


def main():
    submissions = read(BASE / 'submissions.json')
    preflight = read(BASE / 'preflight.json')
    audit = read(BASE / 'audit.json')
    summary = read(BASE / 'summary.json')
    assert summary['all_terminal'] and summary['all_current_points_audited']
    assert summary['all_series_have_confirmed_winner']
    assert summary['validation_error_count'] == audit['summary']['validation_errors'] == 0
    assert audit['inputs']['all_checked']
    assert summary['independent_audit_manifest_matches_current']
    assert not audit['summary']['execution_failures']
    assert len(submissions) == len(summary['runs']) == len(audit['runs'])
    audited = {row['name']: row for row in audit['runs']}
    archives, rows, files = {}, [], []
    for entry in submissions:
        run = ROOT / 'artifacts' / entry['name']
        status, launch = read(run / 'status.json'), read(run / 'launch.json')
        assert status['state'] in ('passed', 'failed')
        unit = launch['unit'] + '.service'
        observed = command(['systemctl', '--user', 'show', unit, '-p',
                            'MainPID,ActiveState,SubState,Slice,ControlGroup,Result'])
        properties = dict(s.split('=', 1) for s in observed.splitlines() if '=' in s)
        assert properties['MainPID'] == '0' and properties['ActiveState'] in ('inactive', 'failed')
        assert 'Slice=background.slice' in launch['unit_observation']
        live = []
        for process in status['descendant_identities']:
            try:
                fields = Path(f"/proc/{process['pid']}/stat").read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue
            if int(fields[19]) == process['start_ticks'] and fields[0] not in ('Z', 'X'):
                live.append(process)
        assert not live, (entry['name'], live)
        source = entry['source']
        assert source == launch['source'] == status['source']
        if source not in archives:
            archive_bytes = subprocess.check_output(['git', 'archive', source], cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
                archives[source] = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                                    for m in archive if m.isfile()}
        expected = archives[source]
        mismatches = [name for name, sha in expected.items()
                      if not (run / 'source' / name).exists()
                      or hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() != sha]
        assert not mismatches, (entry['name'], mismatches)
        queue = read(run / 'queue.json')
        classification = audited[entry['name']]['classification']
        if classification == 'completed_within_budget':
            assert status['state'] == 'passed' and status['exit_code'] == 0
            assert queue['status'] == 'completed'
        else:
            assert classification == 'sampled_budget_refusal'
            assert status['state'] == 'failed' and status['exit_code'] != 0
            assert queue['status'] == 'failed'
        assert queue['return_code'] == status['exit_code']
        assert queue['npu_count'] == entry['cards']
        rows.append(dict(name=entry['name'], state=status['state'], source=source,
                         unit=unit, properties=properties, status=status, queue=queue,
                         live_recorded_descendants=live, frozen_regular_files=len(expected),
                         source_mismatches=mismatches))
        for filename in ('launch.json', 'status.json', 'queue.json', 'environment.json',
                         'inference.json', 'reference.json', 'stdout.log', 'stderr.log', 'run.sh'):
            path = run / filename
            if path.exists():
                files.append(identity(path))
    auxiliary = []
    for name in ('cap-search-controller-v1', 'cap-evidence-audit-v1'):
        run = ROOT / 'artifacts' / name
        status, launch = read(run / 'status.json'), read(run / 'launch.json')
        assert status['state'] == 'passed'
        observed = command(['systemctl', '--user', 'show', launch['unit'], '-p',
                            'MainPID,ActiveState,SubState,Slice,ControlGroup,Result'])
        properties = dict(s.split('=', 1) for s in observed.splitlines() if '=' in s)
        assert properties['MainPID'] == '0' and properties['ActiveState'] == 'inactive'
        assert 'Slice=background.slice' in launch['unit_observation']
        live = []
        for process in status['descendant_identities']:
            try:
                fields = Path(f"/proc/{process['pid']}/stat").read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue
            if int(fields[19]) == process['start_ticks'] and fields[0] not in ('Z', 'X'):
                live.append(process)
        assert not live
        expected = archives[launch['source']]
        mismatches = [f for f, sha in expected.items()
                      if not (run / 'source' / f).exists()
                      or hashlib.sha256((run / 'source' / f).read_bytes()).hexdigest() != sha]
        assert not mismatches
        auxiliary.append(dict(name=name, properties=properties, status=status,
                              launch=launch, live_recorded_descendants=live,
                              source_mismatches=mismatches))
        for filename in ('launch.json', 'status.json', 'environment.json', 'stdout.log', 'stderr.log', 'run.sh'):
            files.append(identity(run / filename))
    controller_start = read(BASE / 'controller-start.json')
    for expected in controller_start['inputs']:
        assert identity(ROOT / expected['path']) == expected
    audit_freeze = read(BASE / 'audit-freeze.json')
    for field in ('original', 'frozen'):
        actual = identity(ROOT / audit_freeze[field])
        assert actual['sha256'] == audit_freeze['sha256'] and actual['bytes'] == audit_freeze['bytes']
        files.append(actual)
    assert audit['auditor']['sha256'] == audit_freeze['sha256']
    protected = []
    for original in preflight['user_files']:
        actual = identity(ROOT / original['path'])
        assert actual == original
        protected.append(dict(actual, unchanged=True))
    for name, sha in preflight['reuse_checks']['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    upstream = json.loads(command([sys.executable, 'scripts/check_upstream.py']))
    assert upstream['state'] == 'passed'
    start = datetime.fromisoformat(preflight['created_at_utc'])
    end = max(datetime.fromisoformat(row['status']['finished']) for row in rows)
    elapsed = (end - start).total_seconds()
    assert elapsed <= preflight['plan']['maximum_project_wall_seconds']
    release_elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    assert release_elapsed <= preflight['plan']['maximum_project_wall_seconds']
    assert all(row['status']['elapsed'] <= 2735 for row in rows)
    events = []
    for entry, row in zip(submissions, rows):
        events.extend([(datetime.fromisoformat(row['status']['started']), entry['cards']),
                       (datetime.fromisoformat(row['status']['finished']), -entry['cards'])])
    concurrent = maximum_concurrent = 0
    for _, change in sorted(events):
        concurrent += change
        maximum_concurrent = max(maximum_concurrent, concurrent)
    assert concurrent == 0 and maximum_concurrent <= preflight['plan']['maximum_concurrent_cards']
    point_counts = Counter((r['scale'], r['kind'], r['cards'], r['confirm']) for r in submissions)
    assert all(count <= (2 if key[3] else 8) for key, count in point_counts.items())
    production_tree = command(['git', 'rev-parse', 'HEAD:src/tide']).strip()
    assert production_tree == preflight['production_tree']
    assert not command(['git', 'diff', '--name-only', '--', 'src/tide', 'math'])
    data = dict(schema='tide-decode-capacity-release-1',
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                source=sorted(archives), production_tree=production_tree,
                upstream=upstream, protected_user_files=protected,
                reused_tested_files=preflight['reuse_checks'], lifecycle=rows,
                auxiliary_lifecycle=auxiliary, controller_start=controller_start, audit_freeze=audit_freeze,
                postprocessing_fixes=read(BASE / 'postprocessing-fixes.json'),
                counts=dict(Counter(row['state'] for row in rows)),
                finite_execution_elapsed_seconds=elapsed, elapsed_through_release_seconds=release_elapsed,
                conservative_maximum_concurrent_requested_cards=maximum_concurrent,
                concurrency_scope='supervisor lifetime includes device queue waits; conservative relative to actual admission',
                point_counts=[dict(scale=k[0], kind=k[1], cards=k[2], confirm=k[3], count=v)
                              for k, v in sorted(point_counts.items())], raw_files=files,
                raw_file_count=len(files), quota=command(['quota', '-s']),
                tools=[identity(BASE / name) for name in ('manage.py', 'audit.py', 'summarize.py', 'release.py', 'continue_search.py')],
                helpers=[identity(path) for path in sorted((BASE / 'helpers').glob('*.py'))],
                input_records=[identity(BASE / name) for name in ('preflight.json', 'submissions.json',
                    'audit.json', 'summary.json', 'search-decisions.json', 'series-decisions.json', 'controller-decisions.json',
                    'postprocessing-fixes.json')])
    atomic(BASE / 'release-audit.json', data)
    print(json.dumps(dict(state='passed', counts=data['counts'], elapsed_seconds=elapsed,
                          raw_files=len(files), protected_files=len(protected))))


if __name__ == '__main__':
    main()
