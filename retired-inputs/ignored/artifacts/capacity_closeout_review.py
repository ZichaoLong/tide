"""Read-only qualification review; writes one small evidence handoff atomically."""
import ast
import collections
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts.job import atomic


def git(*args):
    return subprocess.check_output(['git', *args])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load(path):
    return json.loads(Path(path).read_text())


qualified = git('rev-parse', '8a22530d').decode().strip()
current = git('rev-parse', 'HEAD').decode().strip()
package = []
for name in git('ls-files', 'src/tide').decode().splitlines():
    original = git('show', qualified + ':' + name)
    actual = Path(name).read_bytes()
    package.append(dict(path=name, qualified_sha256=hashlib.sha256(original).hexdigest(),
                        current_sha256=hashlib.sha256(actual).hexdigest(), byte_equal=original == actual,
                        ast_equal=ast.dump(ast.parse(original), include_attributes=False) ==
                        ast.dump(ast.parse(actual), include_attributes=False)))
assert len(package) == 21 and all(row['ast_equal'] for row in package)
archives = []
for run in ('capacity-integrated-cpu-v2', 'capacity-support-catalog-v1'):
    base = Path('artifacts') / run
    source = load(base / 'launch.json')['source']
    mismatches = []
    count = 0
    with tarfile.open(fileobj=io.BytesIO(git('archive', source))) as archive:
        for member in archive:
            if not member.isfile():
                continue
            count += 1
            path = base / 'source' / member.name
            if not path.is_file() or path.read_bytes() != archive.extractfile(member).read():
                mismatches.append(member.name)
    assert not mismatches, (run, mismatches)
    archives.append(dict(run=run, source=source, files=count, mismatches=mismatches))

prior = load('docs/evidence/roadmap-retirement-20261118.json')['inventory']['user_files']
users = []
for path, previous in prior.items():
    now = identity(path)
    now['matches_prior_sha256'] = now['sha256'] == previous['sha256']
    users.append(now)
assert all(row['matches_prior_sha256'] for row in users)

reports = ['docs/evidence/region-capacity-20261118.json',
           'docs/evidence/capacity-installed-20261118.json']
checked, skipped, blobs, seen = [], [], [], set()


def verify(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            path = Path(value['path'])
            key = (str(path), value['sha256'])
            if key not in seen:
                seen.add(key)
                if 'git_blob' in value:
                    content = git('cat-file', 'blob', value['git_blob'])
                    assert hashlib.sha256(content).hexdigest() == value['sha256'], str(path)
                    blobs.append(dict(path=str(path), git_blob=value['git_blob'], sha256=value['sha256']))
                elif path.is_file():
                    if path.stat().st_size <= 8 * 1024**2:
                        actual = identity(path)
                        assert actual['sha256'] == value['sha256'], str(path)
                        checked.append(actual)
                    else:
                        skipped.append(dict(path=str(path), bytes=path.stat().st_size,
                                            reason='Large payload identity remains in originating record; no redundant scan'))
                else:
                    skipped.append(dict(path=str(path), reason='Not an online file; original source identity retained'))
        for item in value.values():
            verify(item)
    elif isinstance(value, list):
        for item in value:
            verify(item)


for report in reports:
    verify(load(report))

files = sorted(Path('artifacts').glob('*/status.json'))
files += sorted(Path('artifacts/worktrees/installed-check/artifacts').glob('capacity-*/status.json'))
jobs, live = [], []
for path in files:
    if not path.parent.name.startswith(('capacity-', 'execution-', 'region-', 'regional-', 'native-regional-')):
        continue
    status = load(path)
    jobs.append(dict(identity=identity(path), state=status.get('state'),
                     exit_code=status.get('exit_code'), source=status.get('source')))
    assert status.get('state') in {'passed', 'failed', 'terminated'}, str(path)
    for owner in status.get('descendant_identities', []):
        proc = Path('/proc') / str(owner['pid']) / 'stat'
        try:
            start = int(proc.read_text().rsplit(')', 1)[1].split()[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if start == owner.get('start_ticks'):
            live.append(dict(status=str(path), **owner))
active = json.loads(subprocess.check_output(['systemctl', '--user', 'list-units',
                    '--state=running,activating', '--type=service', '--output=json', '--no-pager']))
tide_active = [row['unit'] for row in active if row['unit'].startswith('tide-')]
assert not live and not tide_active
suite_base = Path('artifacts/capacity-integrated-cpu-v2')
suite = load(suite_base / 'status.json')
catalog_base = Path('artifacts/capacity-support-catalog-v1')
catalog = load(catalog_base / 'catalog.json')
upstream = json.loads(subprocess.check_output(['python', 'scripts/check_upstream.py']))
targeted = Path('artifacts/capacity-closeout-20261118-targeted.log')
assert '17 passed in 8.89s' in targeted.read_text()
record = dict(
    schema='tide-capacity-validation-1',
    recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Source, CPU tests, support metadata, small evidence identities and terminal handoff; no new device performance',
    review_source=current, qualified_source=qualified, upstream=upstream,
    package=dict(files=package, byte_changed=[row['path'] for row in package if not row['byte_equal']],
                 all_ast_equal=True, qualification_limit='21 package-file ASTs equal, not byte identity or new hardware qualification'),
    frozen_archives=archives,
    integrated_cpu=dict(state=suite['state'], source=suite['source'], passed=544,
                        pytest_seconds=285.99, supervisor_seconds=suite['elapsed'], peak_rss_bytes=suite['peak_rss_bytes'],
                        cpu_affinity='64-71', torch_threads='tests/conftest.py selects 1',
                        resource_identity='pinned CPU; shared host; no NPU',
                        records=[identity(suite_base / name) for name in
                                 ('launch.json', 'status.json', 'environment.json', 'stdout.log', 'stderr.log')],
                        pytest_log=(suite_base / 'stdout.log').read_text(), environment=load(suite_base / 'environment.json')),
    targeted=dict(source=current, command='TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv-instances/bin/python -m pytest -q tests/test_installed_validation.py tests/test_region_batch.py',
                  passed=17, seconds=8.89, record=identity(targeted), log=targeted.read_text(),
                  scope='9 installed-classification tests and 8 region-batch tests; not a 553-test full-suite rerun'),
    help_checks=dict(source=current, record=identity('artifacts/capacity-closeout-20261118-help.json'),
                     results=load('artifacts/capacity-closeout-20261118-help.json')),
    support_catalog=dict(record=identity(catalog_base / 'catalog.json'), implementation=catalog['implementation'],
                         main_menu={key: value for key, value in catalog['main_menu'].items() if key != 'profiles'},
                         optional_bindings=catalog['optional_bindings'], qualification=catalog['qualification'],
                         excluded=catalog['excluded'], lifecycle=identity(catalog_base / 'status.json')),
    reviewed_agent_evidence=dict(reports=[identity(path) for path in reports], checked_small_artifacts=checked,
                                 checked_git_blobs=blobs, skipped_payloads=skipped),
    lifecycle=dict(records=jobs, state_counts=dict(collections.Counter(row['state'] for row in jobs)),
                   live_retained_descendants=live, active_tide_services=tide_active,
                   interpretation='Terminal completion does not promote internal numerical failures to passes'),
    user_files=users, artifact_policy='No cleanup: sources, checkpoints, wheel, environments and witness data retained; external/public/consumer paths unchanged')
atomic('docs/evidence/capacity-validation-20261118.json', record)
print(json.dumps(dict(source=current, package_files=len(package), package_ast_equal=True, archives=archives,
                      checked_evidence_files=len(checked), checked_git_blobs=len(blobs), skipped_payloads=len(skipped),
                      jobs=len(jobs), states=record['lifecycle']['state_counts'], live=live,
                      user_files_unchanged=len(users)), ensure_ascii=False, indent=2))
