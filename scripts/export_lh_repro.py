#!/usr/bin/env python3
"""Export the measured a10fdb1 wide graph and portable, source-only LH helpers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from build_lh_original import audit_source, digest
from durable_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepared, out = Path(a.prepared).resolve(), Path(a.output_dir).resolve()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        p.error('export from a clean Tide source commit')
    build = json.loads((prepared/'build-manifest.json').read_text(encoding='utf-8'))
    if build['source_revision'] != 'a10fdb1883fccd63ec21e36cc9cffa294c63c9e2' or build['expected_parameters'] != 17269426339:
        p.error('expected the measured a10fdb1 / 17,269,426,339 parameter wide build')
    source = Path(build['source'])
    if audit_source(source, build['source_revision'], build['parameter_files_sha256']) != build['source_files_sha256']:
        p.error('prepared source changed')
    for name, expected in build['graph_files_sha256'].items():
        if digest(source/'graph-data'/name) != expected:
            p.error('graph changed: ' + name)
    for name, expected in build['vendor_sha256'].items():
        if digest(prepared/'vendor'/name) != expected:
            p.error('vendor header changed: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    packet = out/'lh-a10-wide-repro-kit'
    shutil.copytree(root/'tools/lh_repro', packet, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(prepared/'vendor', packet/'vendor')
    (packet/'graph-data').mkdir()
    for name in build['graph_files_sha256']:
        shutil.copyfile(source/'graph-data'/name, packet/'graph-data'/name)
    graph_config = json.loads((packet/'graph-data/cfg.json').read_text(encoding='utf-8'))
    # GraphConfig follows this path when loading node metadata. Keep it relocatable.
    graph_config['graph_node_info_dir'] = '../test/graph-data'
    write_json(packet/'graph-data/cfg.json', graph_config)
    files = {str(f.relative_to(packet)): digest(f) for f in sorted(packet.rglob('*')) if f.is_file()}
    write_json(packet/'manifest.json', dict(schema='lh-portable-repro-packet-v1',
        lh_revision=build['source_revision'], export_source=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        build_manifest_sha256=digest(prepared/'build-manifest.json'),
        graph_seed=7, static_nodes=232, edges=dict(inputA=984, outputA=984, ioA=232, oiA=8),
        expected_parameters=17269426339, graph_files=sorted(build['graph_files_sha256']),
        original_graph_files_sha256=build['graph_files_sha256'], files_sha256=files,
        graph_portability='original little-endian int64/float64 graph bytes; cfg path normalized only',
        support='source kit; rebuild on target; x86_64 execution remains target-machine work'))
    with (packet/'SHA256SUMS').open('w') as f:
        for path in sorted(p for p in packet.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'):
            f.write(digest(path)+'  '+str(path.relative_to(packet))+'\n')
    archive = out/'lh-a10-wide-repro-kit.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(packet, arcname=packet.name)
    write_json(out/'export.json', dict(archive=archive.name, bytes=archive.stat().st_size,
        sha256=digest(archive), manifest_sha256=digest(packet/'manifest.json')))
    print((out/'export.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
