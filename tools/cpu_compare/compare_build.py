"""Relocatable source preparation and target-local CMake configuration."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from compare_config import digest, inventory


def torch_prefix(a, env):
    if a.torch_prefix:
        prefix = str(Path(a.torch_prefix).expanduser().resolve())
        if not Path(prefix).is_dir():
            raise ValueError('missing --torch-prefix directory')
        return prefix, dict(discovery='explicit standalone/CMake prefix', prefix=prefix, label=a.libtorch_label)
    command = [sys.executable, '-c', 'import json,torch; print(json.dumps(dict(prefix=torch.utils.cmake_prefix_path, version=torch.__version__, cxx11_abi=torch.compiled_with_cxx11_abi())))']
    try:
        details = json.loads(subprocess.check_output(command, env=env, text=True, timeout=30))
    except (subprocess.SubprocessError, ValueError) as error:
        raise ValueError('cannot discover LibTorch from this Python; activate a Torch environment or pass --torch-prefix') from error
    details['discovery'] = 'matching Python Torch'
    return details['prefix'], details


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('prepared LH source anchor changed: '+before)
    return text.replace(before, after)


def prepare(engine, a, packet, out):
    kit = Path(a.kit_dir); source = out/'source'
    shutil.copytree(kit/'sources'/engine, source)
    if engine == 'lh':
        cfg = source/'test/cfg.json'
        model = json.loads(cfg.read_text())
        for name in ('input_', 'output_', 'iobridge_', 'oibridge_'):
            for key in ('emitD', 'receiveD'):
                model[name][key] = a.width
        model['vocab_size'] = a.vocab
        if getattr(a, 'memory', 'attention') == 'add':
            for name in ('input_', 'output_'):
                for chal in model[name]['chals']:
                    chal['chal'] = 'add'
            model['pronounce']['chal'] = 'add'
        cfg.write_text(json.dumps(model, indent=2)+'\n')
        test = source/'test/test-cortexnet.cpp'
        value = replace_once(test.read_text(), f"int64_t batch_size = {packet['lh_batch']};", f'int64_t batch_size = {a.batch};')
        value = replace_once(value, f"t<{packet['lh_steps']};", f't<{a.steps};')
        test.write_text(value)
        if getattr(a, 'lh_timer', 'original') == 'outer':
            cmake = source/'CMakeLists.txt'
            cmake.write_text(replace_once(cmake.read_text(), ' ENABLE_RAIITIMER', ''))
        shutil.copytree(kit/'graph-data', source/'test/graph-data')
        shutil.copytree(kit/'vendor', source/'vendor')
    return source


def build_commands(engine, a, source, prefix):
    build = source/'build'
    configure = ['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH='+prefix]
    if engine == 'pdg':
        configure.append('-DTIDE_PYTHON_BINDINGS=OFF')
        target = 'tidegraph-scale-bench'
    else:
        configure.append('-DLH_JSON_INCLUDE='+str(source/'vendor'))
        target = 'test-cortexnet-'+('grad' if getattr(a, 'mode', 'nograd') == 'grad-forward' else 'nograd')
    compile_ = ['cmake', '--build', str(build), '--target', target, '--parallel', str(a.jobs)]
    return configure, compile_, build/target


def command(engine, a, binary, out, run_id):
    if engine == 'lh':
        return [str(binary)], binary.parent
    values = dict(device='cpu', dtype=a.dtype, topology=Path(a.kit_dir)/'topology.txt',
                  run_id=run_id, output_dir=out/'native', width=a.width, batch=a.batch,
                  steps=a.steps, warmup=a.warmup, vocab=a.vocab, seed=a.seed,
                  workers=a.threads, threads=1, head_workers=a.threads, parallel_regions=1,
                  compact_events=1, packed=1, grad=int(getattr(a, 'mode', 'nograd') == 'grad-forward'),
                  memory=getattr(a, 'memory', 'attention'), emission='row', profile=getattr(a, 'phase_profile', 1),
                  work_count=a.work_count, attention_packing=a.attention_packing,
                  operator_profile=getattr(a, 'operator_profile', 0),
                  fiber_pooling=getattr(a, 'fiber_pooling', 'event'), fiber_cache=getattr(a, 'fiber_cache', 'cloned'),
                  projection_layout=getattr(a, 'projection_layout', 'input'), defer_state_release=getattr(a, 'defer_state_release', 0),
                  attention_layout=getattr(a, 'attention_layout', 'event'),
                  full_autograd=getattr(a, 'full_autograd', 'replay'), aggregate_autograd=getattr(a, 'aggregate_autograd', 'replay'), packed_sources=getattr(a, 'packed_sources', 0), batch_next=getattr(a, 'batch_next', 0),
                  check=int(a.smoke and a.width <= 64 and a.batch <= 8 and a.steps <= 12))
    result = [str(binary)]
    for key, value in values.items():
        result += ['--'+key.replace('_', '-'), str(value)]
    return result, out


def build_record(engine, source, binary, original):
    # Build outputs are separately hashed; prepared source identities are checked
    # before/after using the original inventory, excluding generated build files.
    for name, expected in original.items():
        if digest(source/name) != expected:
            raise ValueError('prepared source changed: '+name)
    paths = [binary, source/'build/CMakeCache.txt']
    if engine == 'lh':
        paths.append(source/'build/libConnectome.so')
    return {str(p.relative_to(source)): digest(p) for p in paths}
