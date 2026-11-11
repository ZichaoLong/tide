#!/usr/bin/env python3
"""Review explicit FC suites, jobs and consumers without rerunning computation.

Unlike the historical three-layer collector, this entry accepts arbitrary suite
directories and job names. Terminal state, numerical checks and measured costs
remain separate; incomplete inner commands are preserved rather than hidden.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from collect_fc_comparison import cpu_profile, npu_profile
from durable_records import write_json
from source_identity import digest


def read(path):
    return json.loads(path.read_text())


def identity(path):
    return dict(path=str(path), sha256=digest(path))


def collect(suites, jobs=(), consumers=(), junit=()):
    configurations, builds, installations, rows, profiles, commands = {}, {}, {}, [], [], []

    def intern(table, value):
        if value is None:
            return None
        key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        table[key] = value
        return key

    job_records = []
    for path in jobs:
        record = read(path)
        if record['state'] not in {'passed', 'failed', 'cancelled'}:
            raise ValueError('job is still live: '+str(path))
        job_records.append(identity(path) | dict(record=record))
    for suite in suites:
        path = suite/'commands.json'
        launched = read(path)
        commands.append(identity(path) | dict(commands=launched))
        for command in launched:
            path = suite/command['name']/'result.json'
            if not path.exists():
                rows.append(dict(path=str(path), command_exit_code=command.get('exit_code'),
                                 execution='missing_result'))
                continue
            d = read(path)
            if d.get('schema') != 'tide-dense-query-cost-v1':
                raise ValueError('unexpected cost schema: '+str(path))
            measured = [m for m in d['phases'] if not m['warmup']]
            for m in measured:
                rate = m.get('input_positions_per_second')
                if rate is not None and not math.isclose(rate, m['tokens']/m['complete_wall_seconds']):
                    raise ValueError('throughput numerator mismatch: '+str(path))
            keys = ('foundation_source', 'foundation_dirty', 'execution', 'strict_numerical',
                    'discrete', 'recovery', 'learning', 'baseline', 'mode', 'fc_shape', 'profile',
                    'effective_threads', 'effective_interop_threads', 'precision', 'affinity',
                    'visibility', 'trainable_parameters', 'initial_parameters_sha256',
                    'measured_parameters_sha256', 'effective_window_execution', 'scope',
                    'boundary_scope', 'comparison_scope', 'downstream', 'downstream_archive_sha256')
            row = identity(path) | {k: d.get(k) for k in keys}
            row.update(command_exit_code=command.get('exit_code'),
                       configuration=intern(configurations, d['resolved']),
                       native_build=intern(builds, d.get('native_build')),
                       installed_package=intern(installations, d.get('installed_package')),
                       measurements=measured, construction_seconds=d['construction_seconds'])
            rows.append(row)
            if d['profile'] != 'none':
                profile = identity(path) | dict(kind=d['profile'])
                if d['profile'] == 'cpu':
                    profile['operators'] = cpu_profile(path.parent)
                elif d['profile'] == 'npu':
                    profile['operators'] = npu_profile(path.parent, d)
                else:
                    raise ValueError('unknown profiler: '+d['profile'])
                census = path.parent/'autograd-census.json'
                if census.exists():
                    raw = read(census)['autograd']
                    profile['census'] = identity(census) | dict(
                        python=raw.get('python'), native=raw.get('native'), scope=raw['scope'])
                profiles.append(profile)
    return dict(schema='tide-fc-suites-review-v1', collector_sha256=digest(__file__),
                jobs=job_records, commands=commands, configurations=configurations,
                builds=builds, installations=installations, costs=rows, profiles=profiles,
                consumers=[identity(p) | dict(record=read(p)) for p in consumers],
                junit=[identity(p) | dict(suites=[dict(x.attrib) for x in ET.parse(p).iter('testsuite')])
                       for p in junit],
                scope='Recorded observations only. Completion does not establish equivalence. '
                      'Profile/census timing is separate from uninstrumented throughput. '
                      'Python/native census universes overlap; never add them.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, action='append', required=True)
    parser.add_argument('--job', type=Path, action='append', default=[])
    parser.add_argument('--consumer', type=Path, action='append', default=[])
    parser.add_argument('--junit', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = collect(args.suite, args.job, args.consumer, args.junit)
    write_json(args.output, result)
    print(json.dumps({k: len(result[k]) for k in ('costs', 'profiles', 'consumers', 'junit')}))


if __name__ == '__main__':
    main()
