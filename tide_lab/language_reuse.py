"""Validate scoped reuse of terminal language admission, without executing models."""
import copy
import json
from pathlib import Path
from .records import digest


def validate_recipe(previous, current):
    """Only the declared exposure admission and descriptive metadata may change."""
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    for value in (old, new):
        for key in ('package_id', 'adjustment', 'exploration'):
            value.pop(key, None)
    new['training']['minimum_valid_tokens'] = old['training']['minimum_valid_tokens']
    if old != new:
        raise ValueError('reused mechanisms require unchanged model, loss, optimizer and checks')
    if current['training']['minimum_valid_tokens'] != 32000:
        raise ValueError('this exploration fixes the smaller exposure admission at 32000')


def verify_source_files(source, identity):
    """Check every prior experiment module and the unchanged stage entry point."""
    verified = {}
    for row in identity['content']['files']:
        name = row['path']
        if (name.startswith('tide_lab/') and name.endswith('.py')) or name in (
                'scripts/language_stage.py', 'scripts/env.sh',
                'dependencies/foundation-source.json', 'dependencies/foundation.lock.json',
                'dependencies/installation.lock.json', 'dependencies/environment.lock.json'):
            if digest(Path(source) / name) != row['sha256']:
                raise ValueError('reused execution source changed: ' + name)
            verified[name] = row['sha256']
    if 'scripts/language_stage.py' not in verified or 'tide_lab/language_training.py' not in verified:
        raise ValueError('incomplete prior source inventory')
    return verified


def admission(plan, route, prior, data, source, installation):
    prior, data = Path(prior), Path(data)
    state = json.loads((prior / 'stage.json').read_text())
    if state['route'] != route or state['state'] not in ('failed', 'completed'):
        raise ValueError('reuse requires a terminal route record')
    kinds = plan[route + '_candidates']
    reference = json.loads((prior / ('check-' + kinds[0]) / 'manifest.json').read_text())
    validate_recipe(reference['config'], plan)
    ident = reference['identity']
    if ident['installation'] != installation:
        raise ValueError('installed execution identity changed')
    if ident['data_lock_sha256'] != digest(data / 'data-lock.json'):
        raise ValueError('reused token data identity changed')
    hashes = verify_source_files(source, ident['source'])
    evidence = {str(prior / 'stage.json'): digest(prior / 'stage.json')}
    for kind in kinds:
        candidate = state['candidates'][kind]
        if not candidate.get('admitted') or candidate['recovery']['state'] != 'passed':
            raise ValueError('candidate lacks mechanism and exact recovery admission: ' + kind)
        for name in ('check-' + kind, 'resume-d1-' + kind + '-full',
                     'resume-d1-' + kind + '-split', 'resume-d1-' + kind + '-continued'):
            path = prior / name
            result = json.loads((path / 'result.json').read_text())
            manifest = json.loads((path / 'manifest.json').read_text())
            if result['state'] != 'completed':
                raise ValueError('reused check did not complete: ' + name)
            current = manifest['identity']
            for key in ('source', 'installation', 'config_sha256', 'data_lock_sha256',
                        'cache_sha256', 'route', 'device', 'dtype', 'threads'):
                if current[key] != ident[key]:
                    raise ValueError('inconsistent admission identity: ' + key)
            if current['kind'] != kind:
                raise ValueError('check candidate identity mismatch')
            for filename in ('manifest.json', 'result.json'):
                evidence[str(path / filename)] = digest(path / filename)
        check = json.loads((prior / ('check-' + kind) / 'result.json').read_text())
        if not check.get('admitted_for_finite_exploration'):
            raise ValueError('mechanism admission is false')
        for suffix, key in (('full', 'full_sha256'), ('continued', 'resumed_sha256')):
            path = prior / ('resume-d1-' + kind + '-' + suffix) / 'checkpoint.pt'
            if digest(path) != candidate['recovery'][key]:
                raise ValueError('recovery checkpoint changed')
            evidence[str(path)] = digest(path)
    cases = []
    for batch in (8, 32):
        rows = {}
        for kind in kinds:
            path = prior / f'cal-d1-{kind}-L32-B{batch}'
            row = json.loads((path / 'result.json').read_text())
            manifest = json.loads((path / 'manifest.json').read_text())
            current = manifest['identity']
            if row['state'] != 'completed' or row['batch'] != batch or row['length'] != 32:
                raise ValueError('missing complete calibrated case')
            for key in ('source', 'installation', 'config_sha256', 'data_lock_sha256',
                        'cache_sha256', 'route', 'device', 'dtype', 'threads'):
                if current[key] != ident[key]:
                    raise ValueError('inconsistent calibration identity: ' + key)
            if (current['kind'], current['batch'], current['length']) != (kind, batch, 32):
                raise ValueError('calibration shape/candidate mismatch')
            if row['npu_peak_allocated_mib'] > plan['batch_calibration']['peak_fraction'] * row['npu_capacity_mib']:
                raise ValueError('reused calibration lacks memory headroom')
            rows[kind] = row
            for filename in ('manifest.json', 'result.json'):
                evidence[str(path / filename)] = digest(path / filename)
        cases.append(dict(batch=batch, length=32, calibrations=rows))
    cache = None
    if route == 'b':
        cache = prior / 'cache-L32/features.pt'
        if digest(cache) != ident['cache_sha256']:
            raise ValueError('frozen prefix cache changed')
        evidence[str(cache)] = ident['cache_sha256']
    return dict(cases=cases, cache=cache, evidence=evidence, source_files=hashes,
                source_commit=ident['source']['commit'], candidates=state['candidates'],
                scope='unchanged execution files and recipe; prior B2/L3 recovery and small checks only')
