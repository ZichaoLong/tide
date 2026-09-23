#!/usr/bin/env python3
"""Bundle fixed LH/PDG CPU sources, measured graph, vendor header and one-command runners."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from build_lh_original import audit_source, digest
from durable_records import write_json, replace_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lh-prepared', required=True, help='audited accounting-enabled 17.27B prepared LH copy')
    p.add_argument('--topology', required=True, help='measured PDG graph-only text plus adjacent .json provenance')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--allow-dirty', action='store_true', help='development packet; all copied bytes are hashed')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    if dirty and not a.allow_dirty:
        p.error('export from clean committed source (or explicitly use --allow-dirty for development)')
    prepared = Path(a.lh_prepared).resolve(); topology = Path(a.topology).resolve()
    build = json.loads((prepared/'build-manifest.json').read_text())
    source = Path(build['source'])
    if not build.get('accounting') or build['expected_parameters'] != 17269426339:
        p.error('requires the measured accounting-enabled wide preparation')
    if audit_source(source, build['source_revision'], build['parameter_files_sha256']) != build['source_files_sha256']:
        p.error('prepared LH source changed')
    for name, expected in build['added_files_sha256'].items():
        if digest(source/name) != expected:
            p.error('added LH accounting file changed: '+name)
    info = json.loads(topology.with_suffix(topology.suffix+'.json').read_text())
    if digest(topology) != info['topology_sha256'] or info['static_nodes'] != 232 or info['logical_edges'] != 2208:
        p.error('expected the measured wide graph')
    for name, expected in build['graph_files_sha256'].items():
        if digest(source/'graph-data'/name) != expected:
            p.error('LH graph changed: '+name)
    if any(build['graph_files_sha256'][k] != v for k,v in info['graph_files_sha256'].items()):
        p.error('LH/PDG graph mismatch')
    for name, expected in build['vendor_sha256'].items():
        if digest(prepared/'vendor'/name) != expected:
            p.error('vendor header changed: '+name)
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    packet = out/'cpu-attention-compare'
    shutil.copytree(root/'tools/cpu_compare',packet,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('durable_records.py','experiment_record.py'):
        shutil.copyfile(root/'scripts'/name,packet/name)
    shutil.copytree(root/'cpp',packet/'sources/pdg/cpp')
    shutil.copyfile(root/'CMakeLists.txt',packet/'sources/pdg/CMakeLists.txt')
    for name in {**build['source_files_sha256'], **build['added_files_sha256']}:
        prefix = 'Connectome/cpp/'
        if not name.startswith(prefix):
            continue
        suffix = Path(name[len(prefix):]); dest = packet/'sources/lh'/suffix
        if suffix.parts[0] in ('include','src','test') or suffix.name == 'CMakeLists.txt':
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source/name,dest)
    # Libraries compile exactly the counted source; only build diagnostics added.
    for engine in ('lh','pdg'):
        cmake = packet/'sources'/engine/'CMakeLists.txt'
        replace_text(cmake, cmake.read_text()+'''\nmessage(STATUS "COMPARE compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; CPU: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "COMPARE Torch: ${Torch_VERSION}; flags: ${TORCH_CXX_FLAGS}; prefix: ${CMAKE_PREFIX_PATH}")
message(STATUS "COMPARE Release flags: ${CMAKE_CXX_FLAGS_RELEASE}; global flags: ${CMAKE_CXX_FLAGS}")
''')
    shutil.copytree(prepared/'vendor',packet/'vendor')
    shutil.copyfile(root/'tools/lh_repro/LICENSE.nlohmann',packet/'LICENSE.nlohmann')
    (packet/'graph-data').mkdir()
    for name in build['graph_files_sha256']:
        shutil.copyfile(source/'graph-data'/name,packet/'graph-data'/name)
    cfg = json.loads((packet/'graph-data/cfg.json').read_text())
    cfg['graph_node_info_dir'] = '../test/graph-data'
    write_json(packet/'graph-data/cfg.json',cfg)
    shutil.copyfile(topology,packet/'topology.txt')
    files = {str(f.relative_to(packet)):digest(f) for f in sorted(packet.rglob('*')) if f.is_file()}
    for f in packet.rglob('*'):
        if f.is_file(): f.chmod(0o644)
    write_json(packet/'manifest.json',dict(schema='tide-lh-cpu-compare-v1',
        tide_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_dirty=bool(dirty), source_dirty_status=dirty,
        lh_revision=build['source_revision'], lh_batch=build['batch'],lh_steps=build['steps'],
        lh_preparation_manifest_sha256=digest(prepared/'build-manifest.json'),
        graph_csr_sha256=info['graph_files_sha256'],topology_sha256=info['topology_sha256'],
        files_sha256=files, graph_format='little-endian int64/float64 original LH; PDG portable text',
        support='CPU source kit; target-local build required; no binary portability claim',
        default_config=dict(width=2048,batch=512,vocab=50304,steps=12,warmup=4,seed=7,dtype='float32',grad=False,
                            pdg_attention_packing='exact'),
        expected_parameters=17269426339))
    sums = ''.join(digest(f)+'  '+str(f.relative_to(packet))+'\n' for f in sorted(packet.rglob('*')) if f.is_file())
    replace_text(packet/'SHA256SUMS',sums)
    archive = out/'cpu-attention-compare.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(packet,arcname=packet.name)
    write_json(out/'export.json',dict(archive=archive.name,bytes=archive.stat().st_size,
        sha256=digest(archive),manifest_sha256=digest(packet/'manifest.json')))
    print((out/'export.json').read_text())


if __name__ == '__main__':
    main()
