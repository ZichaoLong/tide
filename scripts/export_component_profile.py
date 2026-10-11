#!/usr/bin/env python3
"""Export a deferred NPU component capture in a fresh, externally bounded process.

No model execution or new measurement. Original timing, capture and raw files
remain immutable. Use a new output directory and the remaining case budget.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

from component_profile_summary import raw_identity, summarize_trace
from durable_records import write_json
from npu_profile_analysis import analyse_raw
from python_environment import working_directory
from source_identity import digest, source_state


def export_capture(capture, output):
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    record = dict(schema='tide-component-export-v1', state='running', source=source,
        dirty=dirty, command=sys.argv, capture=str(capture),
        scope='offline export only; original capture and timing unchanged; no remeasurement')
    write_json(output/'result.json', record)
    before = None
    try:
        measurement = json.loads((capture/'measurement.json').read_text())
        collected = json.loads((capture/'result.json').read_text())
        if collected.get('state') != 'collected' or collected.get('profile_state') != 'collected':
            raise ValueError('offline entry requires a completed deferred capture')
        if not str(measurement.get('device', '')).startswith('npu:'):
            raise ValueError('offline component export requires an NPU capture')
        seconds = measurement.get('seconds')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid original measurement')
        before = dict(raw=raw_identity(capture/'raw'),
            measurement=digest(capture/'measurement.json'), result=digest(capture/'result.json'))
        if before['raw'] != collected['raw_files_sha256'] or before['measurement'] != collected['measurement_sha256']:
            raise ValueError('capture changed after collection')
        sessions = list((capture/'raw').glob('*_ascend_pt'))
        if len(sessions) != 1 or not (sessions[0]/'profiler_info.json').is_file():
            raise ValueError('expected exactly one completed TorchNPU session')
        record.update(input_identity=before, capture_source=measurement['source'],
            capture_dirty=measurement['dirty'], seconds=seconds,
            measurement_state='retained original single measurement')
        write_json(output/'result.json', record)
        raw = output/'raw-copy'
        shutil.copytree(capture/'raw', raw)
        for old_trace in raw.rglob('trace_view.json'):
            old_trace.unlink()  # Derived copies only; never accept a stale export.
        record['analysis'] = analyse_raw(raw, output)
        traces = list(raw.rglob('trace_view.json'))
        if len(traces) != 1:
            raise RuntimeError('offline export did not produce exactly one trace')
        shutil.copyfile(traces[0], output/'trace.json')
        record.update(summarize_trace(output/'trace.json', 'npu'))
        if source_state(root) != (source, dirty):
            raise RuntimeError('exporter source changed')
        record.update(state='passed', profile_state='exported')
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        if before is not None:
            try:
                after = dict(raw=raw_identity(capture/'raw'),
                    measurement=digest(capture/'measurement.json'), result=digest(capture/'result.json'))
                record['original_unchanged'] = before == after
                if before != after:
                    raise RuntimeError('original capture changed during export')
            except BaseException as error:
                record.update(state='failed', original_unchanged=False, error=repr(error))
        write_json(output/'result.json', record)
    if record['state'] != 'passed':
        raise RuntimeError(record['error'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    capture, output = a.capture_dir.resolve(), a.output_dir.resolve()
    if output == capture or capture in output.parents:
        p.error('export output must be outside the immutable capture')
    output.mkdir(parents=True, exist_ok=False)
    with working_directory(output):
        export_capture(capture, output)


if __name__ == '__main__':
    main()
