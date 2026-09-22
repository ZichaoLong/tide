#!/usr/bin/env python3
"""Apply the exported Attention workload to a clean a10fdb1 LH checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys


REVISION = 'a10fdb1883fccd63ec21e36cc9cffa294c63c9e2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('unexpected original test source: ' + before)
    return text.replace(before, after)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lh-root', required=True)
    p.add_argument('--width', type=int, default=2048)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()
    if not 4 <= args.width <= 2048 or args.width % 4 or not 1 <= args.batch <= 512 or not 1 <= args.steps <= 100:
        p.error('width must be a multiple of 4 in [4,2048], batch in [1,512], steps in [1,100]')
    if sys.platform != 'linux' or sys.byteorder != 'little' or struct.calcsize('P') != 8:
        p.error('this kit targets little-endian 64-bit Linux')
    packet = Path(__file__).resolve().parent
    manifest = json.loads((packet/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        if digest(packet/name) != expected:
            p.error('reproduction packet changed: ' + name)
    root = Path(args.lh_root).resolve()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if revision != REVISION:
        p.error('LH checkout must be at ' + REVISION)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root, text=True).strip():
        p.error('use a clean, separate LH checkout; existing changes were not overwritten')
    cpp = root/'Connectome/cpp'
    for target in [cpp/'repro', cpp/'test/test-cortexnet-info.cpp', cpp/'test/lh_run_info.h']:
        if target.exists():
            p.error('refusing to overwrite ' + str(target))
    if any(f.name != '.gitkeep' for f in (root/'graph-data').iterdir()):
        p.error('existing graph data found; use a fresh checkout')
    if not (cpp/'test/graph-data').is_symlink() or (cpp/'test/graph-data').resolve() != root/'graph-data':
        p.error('original test/graph-data link is missing or changed')
    model = json.loads((cpp/'test/cfg.json').read_text(encoding='utf-8'))
    changes = []
    for name in ('input_', 'output_', 'iobridge_', 'oibridge_'):
        for key in ('emitD', 'receiveD'):
            changes.append(dict(path=f'/{name}/{key}', before=model[name][key], after=args.width))
            model[name][key] = args.width
    original = (cpp/'test/test-cortexnet.cpp').read_text(encoding='utf-8')
    original = replace_once(original, 'int64_t batch_size = 512;', f'int64_t batch_size = {args.batch};')
    original = replace_once(original, 't<100;', f't<{args.steps};')
    diagnostic = replace_once(original, '#include "test-connectome.h"',
                              '#include "test-connectome.h"\n#include "lh_run_info.h"')
    diagnostic = replace_once(diagnostic, 'int main()\n{', '''int main(int argc, char **argv)
{
    if (!LHRunInfo::arguments(argc, argv)) return 2;
    LHRunInfo::environment();''')
    diagnostic = replace_once(diagnostic, '    VPtrBatchSignals iacts,oacts;',
        '    LHRunInfo::model(ionet, gd, cfg, batch_size, selectnum);\n    VPtrBatchSignals iacts,oacts;')
    diagnostic = replace_once(diagnostic, '    for (int64_t t=0;',
        '    LHRunInfo::Times times(batch_size);\n    for (int64_t t=0;')
    diagnostic = replace_once(diagnostic, '        Tensor logits = ionet.think(',
        '        const auto info_start = LHRunInfo::Clock::now();\n        Tensor logits = ionet.think(')
    diagnostic = replace_once(diagnostic, '        cout << "t: " << t << endl;',
        '        const auto info_end = LHRunInfo::Clock::now();\n'
        '        cout << "t: " << t << endl;\n'
        '        times.step(t, info_start, info_end, logits);')
    expected = (2208 + 4*(2*232+1))*args.width**2 + (2*232+2*50304)*args.width + 2208+1+2
    # Every check above happens before changing the requested checkout.
    shutil.copytree(packet/'graph-data', root/'graph-data', dirs_exist_ok=True)
    shutil.copytree(packet/'vendor', cpp/'repro/vendor')
    shutil.copyfile(packet/'CMakeLists.txt', cpp/'CMakeLists.txt')
    shutil.copyfile(packet/'lh_run_info.h', cpp/'test/lh_run_info.h')
    (cpp/'test/cfg.json').write_text(json.dumps(model, indent=2)+'\n', encoding='utf-8')
    (cpp/'test/test-cortexnet.cpp').write_text(original, encoding='utf-8')
    (cpp/'test/test-cortexnet-info.cpp').write_text(diagnostic, encoding='utf-8')
    record = dict(schema='lh-portable-repro-config-v1', lh_revision=revision,
        packet_manifest_sha256=digest(packet/'manifest.json'), width=args.width,
        batch=args.batch, steps=args.steps, selectnum=1, dtype='float32', device='cpu',
        expected_parameters=expected, static_nodes=232, edges=manifest['edges'],
        seed_policy='original model/input RNG; no manual seed', width_changes=changes,
        graph_files_sha256={name: digest(root/'graph-data'/name)
            for name in manifest['graph_files']})
    record['configured_source_sha256'] = {str(f.relative_to(root)): digest(f) for f in
        [cpp/'CMakeLists.txt', cpp/'test/cfg.json', cpp/'test/test-cortexnet.cpp',
         cpp/'test/test-cortexnet-info.cpp', cpp/'test/lh_run_info.h']}
    (cpp/'repro/configured.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    subprocess.run(['git', 'diff', '--binary', '--output='+str(cpp/'repro/configuration.patch')], cwd=root, check=True)
    print(json.dumps(record, indent=2))
    print('Configured. Build from Connectome/cpp and run executables from its build directory.')


if __name__ == '__main__':
    main()
