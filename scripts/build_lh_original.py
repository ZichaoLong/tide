#!/usr/bin/env python3
"""Freeze a named LH revision and build its unchanged original CPU test."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from durable_records import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_source(source, revision, parameter_files=None):
    archive = subprocess.check_output(['git', 'archive', revision], cwd=source)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            path = source / member.name
            expected = tar.extractfile(member).read()
            if member.name != 'Connectome/cpp/CMakeLists.txt':
                allowed = (parameter_files or {}).get(member.name)
                if allowed is not None:
                    if digest(path) != allowed:
                        raise ValueError('configured parameter file changed: ' + member.name)
                elif path.read_bytes() != expected:
                    raise ValueError('original tracked source changed: ' + member.name)
            hashes[member.name] = digest(path)
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--revision', default='a10fdb1')
    p.add_argument('--profile', required=True, choices=['wide', 'narrow'])
    p.add_argument('--width', type=int, help='explicit small bring-up override')
    p.add_argument('--accounting', action='store_true', help='optional audited inference work-count instrumentation')
    p.add_argument('--vocab', type=int, help='small accounting fixtures only')
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--graph-input', required=True, help='existing graph-generation input record; its model is not used')
    p.add_argument('--json-include', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.jobs <= 4:
        p.error('build jobs must be in [1,4]')
    width = args.width if args.width is not None else (2048 if args.profile == 'wide' else 128)
    if not 4 <= width <= 2048 or width % 4 or not 1 <= args.batch <= 512 or not 1 <= args.steps <= 100:
        p.error('outside bounded original-test parameters')
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        p.error('build from a clean frozen Tide commit')
    out = Path(args.output_dir).resolve()
    inputs = Path(args.graph_input).resolve()
    graph_record = json.loads(inputs.read_text())
    graph = Path(graph_record['graph_dir'])
    for name, expected in graph_record['graph_files_sha256'].items():
        if digest(graph / name) != expected:
            p.error('generated graph changed: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(Path(args.source).resolve()), str(source)], check=True)
    revision = subprocess.check_output(['git', 'rev-parse', args.revision], cwd=source, text=True).strip()
    subprocess.run(['git', 'checkout', '--detach', revision], cwd=source, check=True)
    cfg_path = source / 'Connectome/cpp/test/cfg.json'
    model = json.loads(cfg_path.read_text())
    changes = []
    if args.vocab is not None:
        if not args.accounting or not 2 <= args.vocab <= 50304:
            p.error('vocab override requires accounting and range [2,50304]')
        changes.append(dict(file='test/cfg.json', path='/vocab_size', before=model['vocab_size'], after=args.vocab))
        model['vocab_size'] = args.vocab
    for name in ('input_', 'output_', 'iobridge_', 'oibridge_'):
        for key in ('emitD', 'receiveD'):
            changes.append(dict(file='test/cfg.json', path=f'/{name}/{key}', before=model[name][key], after=width))
            model[name][key] = width
    for name in ('input_', 'output_'):
        if any(chal['chal'] != 'attention' for chal in model[name]['chals']):
            raise ValueError('selected baseline is not the expected Attention profile')
    if model['pronounce']['chal'] != 'attention':
        raise ValueError('selected baseline Pronounce is not Attention')
    cfg_path.write_text(json.dumps(model, indent=2) + '\n')
    n, edges = graph_record['static_nodes'], sum(graph_record['edges'].values())
    estimated_parameters = (edges + 4 * (2 * n + 1)) * width**2 + (2 * n + 2 * model['vocab_size']) * width + edges + 1 + model['n_layer']
    test_path = source / 'Connectome/cpp/test/test-cortexnet.cpp'
    test = test_path.read_text()
    selector = 1 if args.profile == 'wide' else 2
    for before, after in [('int64_t batch_size = 512;', f'int64_t batch_size = {args.batch};'),
                          ('int64_t selectnum = 1;', f'int64_t selectnum = {selector};'),
                          ('t<100;', f't<{args.steps};')]:
        if test.count(before) != 1:
            raise ValueError('original test parameter declaration changed: ' + before)
        test = test.replace(before, after)
        if before != after:
            changes.append(dict(file='test/test-cortexnet.cpp', before=before, after=after))
    test_path.write_text(test)
    modified, added = [], []
    if args.accounting:
        from lh_work_instrument import instrument
        modified, added = instrument(source, root)
    parameter_files = {str(f.relative_to(source)): digest(f) for f in (cfg_path, test_path, *modified)}
    cmake = root / 'cpp/lh_original/CMakeLists.txt'
    shutil.copyfile(cmake, source / 'Connectome/cpp/CMakeLists.txt')
    shutil.copytree(graph, source / 'graph-data', dirs_exist_ok=True)
    vendor = out / 'vendor'
    shutil.copytree(Path(args.json_include).resolve() / 'nlohmann', vendor / 'nlohmann')
    before = audit_source(source, revision, parameter_files)
    # Prove that the retained graph generator is from the selected LH revision.
    for name, expected in graph_record['generator_sha256'].items():
        if digest(source / 'PyConnectome' / name) != expected:
            raise ValueError('selected revision differs from the retained graph generator: ' + name)
    build = source / 'Connectome/cpp/build'
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    subprocess.run(['cmake', '-S', str(build.parent), '-B', str(build), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH=' + torch.utils.cmake_prefix_path,
        '-DLH_JSON_INCLUDE=' + str(vendor)], check=True)
    subprocess.run(['cmake', '--build', str(build), '--parallel', str(args.jobs)], check=True)
    if audit_source(source, revision, parameter_files) != before or digest(cmake) != before['Connectome/cpp/CMakeLists.txt']:
        raise ValueError('source changed during build')
    artifacts = {str(f.relative_to(out)): digest(f) for f in [build / 'test-cortexnet-grad',
        build / 'test-cortexnet-nograd', build / 'libConnectome.so']}
    write_json(out / 'build-manifest.json', dict(schema='lh-original-test-build-v1', source=str(source),
        source_revision=revision, source_files_sha256=before, parameter_files_sha256=parameter_files,
        parameter_changes=changes, cmake_sha256=digest(cmake), accounting=args.accounting,
        added_files_sha256={str(f.relative_to(source)): digest(f) for f in added},
        tide_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        build_dir=str(build), artifacts_sha256=artifacts,
        vendor_sha256={str(f.relative_to(vendor)): digest(f) for f in vendor.rglob('*') if f.is_file()},
        torch_version=torch.__version__, cxx11_abi=torch.compiled_with_cxx11_abi(),
        dtype='float32', model=model, config_sha256=digest(cfg_path),
        batch=args.batch, steps=args.steps, selector_budget=selector, width=width, profile=args.profile,
        expected_parameters=estimated_parameters,
        initial_crossbatch_kv_bytes=(2*n+1)*args.batch*16*width*2*4,
        graph_input_path=str(inputs), graph_input_sha256=digest(inputs),
        graph_files_sha256=graph_record['graph_files_sha256'], graph_config=graph_record['graph_config'],
        graph_seed=graph_record['seed'], static_nodes=graph_record['static_nodes'],
        changes='CMake and listed parameters; '+('optional operator counters, seed7/default and fixed token IDs' if args.accounting else 'original C++ algorithms unchanged'),
        timing='original ENABLE_RAIITIMER; inner timer printing included; integer milliseconds',
        mode='Release, DISABLE_CUDA, ENABLE_PARALLEL_FOR; separate NOGRAD test target'))
    print(json.dumps(dict(state='built', manifest=str(out / 'build-manifest.json'))))


if __name__ == '__main__':
    main()
