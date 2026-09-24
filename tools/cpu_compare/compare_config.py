"""Portable arguments, packet integrity and selected host metadata; no Torch import."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_packet(root):
    path = root/'manifest.json'
    if not path.exists():
        raise ValueError('use the exported cpu-attention-compare kit, or pass --kit-dir KIT')
    manifest = json.loads(path.read_text())
    if manifest['schema'] != 'tide-lh-cpu-compare-v1':
        raise ValueError('unknown packet schema')
    for name, expected in manifest['files_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(root/relative) != expected:
            raise ValueError('packet hash/path mismatch: '+name)
    return manifest


def parse(engine):
    p = argparse.ArgumentParser(description='One-command '+engine.upper()+' CPU comparison: build, run, report.')
    p.add_argument('--device', required=True, choices=['cpu'])
    p.add_argument('--dtype', default='float32', choices=['float32'] if engine == 'lh' else ['float32', 'float64'])
    p.add_argument('--kit-dir', default=str(Path(__file__).resolve().parent))
    p.add_argument('--output-dir', help='new directory; default runs/ENGINE-UTC-ID')
    p.add_argument('--torch-prefix', help='standalone LibTorch root or CMake prefix; otherwise discover from this Python')
    p.add_argument('--libtorch-label', help='optional archive/package identity when using --torch-prefix')
    p.add_argument('--smoke', action='store_true', help='D16/B4/V257/6 steps/warmup2, instead of the 17.27B workload')
    p.add_argument('--memory', choices=['attention', 'add'], default='attention')
    p.add_argument('--mode', choices=['nograd', 'grad-forward'], default='nograd',
                   help='grad-forward retains the forward graph; no backward/optimizer/detach')
    if engine == 'lh':
        p.add_argument('--lh-timer', choices=['original', 'outer'], default='original',
                       help='outer disables original RAII timers and uses the surrounding steady-clock interval')
    for key in ('width', 'batch', 'vocab', 'steps', 'warmup'):
        p.add_argument('--'+key, type=int)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--threads', type=int, default=min(56, len(os.sched_getaffinity(0))))
    p.add_argument('--jobs', type=int, default=2, help='CMake build jobs')
    p.add_argument('--work-count', type=int, choices=[0, 1], default=None)
    p.add_argument('--operator-profile', type=int, choices=[0, 1], default=0,
                   help='exclusive calling-thread timers; do not add to wall intervals')
    if engine == 'pdg':
        p.add_argument('--full-autograd', choices=['replay', 'batched'], default='replay')
        p.add_argument('--phase-profile', type=int, choices=[0, 1], default=1,
                       help='Streaming phase timers; set 0 for an unprofiled comparison')
        p.add_argument('--attention-packing', choices=['exact', 'single'], default='exact',
                       help='exact shape buckets or one padded query batch per node update')
        p.add_argument('--fiber-pooling', choices=['event', 'csr'], default='event')
        p.add_argument('--fiber-cache', choices=['cloned', 'owned'], default='cloned')
        p.add_argument('--projection-layout', choices=['input', 'linear'], default='input')
        p.add_argument('--attention-layout', choices=['event', 'head'], default='event')
        p.add_argument('--defer-state-release', type=int, choices=[0, 1], default=0)
        p.add_argument('--packed-sources', type=int, choices=[0, 1], default=0)
        p.add_argument('--batch-next', type=int, choices=[0, 1], default=0)
    p.add_argument('--timeout-seconds', type=int, default=3600, help='native execution only')
    p.add_argument('--build-timeout-seconds', type=int, default=3600)
    p.add_argument('--memory-gib', type=int, default=0, help='optional address-space bound; 0 means no added limit')
    p.add_argument('--tracking', choices=['best-effort', 'off', 'required'], default='best-effort')
    a = p.parse_args()
    if a.work_count is None:
        a.work_count = int(a.mode == 'nograd')
    if engine == 'lh' and a.mode == 'grad-forward' and (a.work_count or a.operator_profile):
        p.error('LH grad-forward requires work-count=0 and operator-profile=0')
    if engine == 'pdg' and a.memory == 'add' and (a.attention_packing != 'exact'
            or a.fiber_pooling != 'event' or a.fiber_cache != 'cloned'
            or a.projection_layout != 'input' or a.attention_layout != 'event'):
        p.error('fiber attention policies do not apply to Add')
    if sys.platform != 'linux' or sys.byteorder != 'little' or struct.calcsize('P') != 8:
        p.error('requires little-endian 64-bit Linux')
    defaults = dict(width=16, batch=4, vocab=257, steps=6, warmup=2) if a.smoke else dict(width=2048, batch=512, vocab=50304, steps=12, warmup=4)
    for key, value in defaults.items():
        if getattr(a, key) is None:
            setattr(a, key, value)
    if (not 4 <= a.width <= 2048 or a.width%4 or not 1 <= a.batch <= 512
            or not 2 <= a.vocab <= 50304 or not 0 <= a.warmup < a.steps <= 100
            or not 1 <= a.threads <= min(160, len(os.sched_getaffinity(0)))
            or not 1 <= a.jobs <= 8 or not 0 <= a.seed < 2**63
            or a.timeout_seconds < 1 or a.build_timeout_seconds < 1 or a.memory_gib < 0):
        p.error('invalid shape, window, resources or seed; use --help')
    a.kit_dir = str(Path(a.kit_dir).resolve())
    try:
        packet = verify_packet(Path(a.kit_dir))
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    if engine == 'lh' and a.operator_profile and not packet.get('lh_operator_profile'):
        p.error('LH operator profiling requires a freshly prepared capable source kit')
    run_id = engine+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    a.output_dir = str(Path(a.output_dir or Path.cwd()/'runs'/run_id).resolve())
    if Path(a.output_dir).exists():
        p.error('refusing existing output directory: '+a.output_dir)
    return a, packet, run_id


def parameters(a):
    attention = 4*465 if getattr(a, 'memory', 'attention') == 'attention' else 0
    return (2208+attention)*a.width**2+(464+2*a.vocab)*a.width+2211


def host_info():
    selected = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_WAIT_POLICY',
                'OMP_PROC_BIND', 'OMP_PLACES', 'KMP_AFFINITY', 'KMP_BLOCKTIME')
    result = dict(architecture=platform.machine(), platform=platform.platform(), python=sys.version,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), load_average=list(os.getloadavg()),
                  inherited_thread_environment={k: os.environ[k] for k in selected if k in os.environ})
    result['meminfo'] = Path('/proc/meminfo').read_text()
    for name, command in [('lscpu', ['lscpu']), ('cmake', ['cmake', '--version'])]:
        try:
            result[name] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            result[name] = str(error)
    return result


def environment(engine, a):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS=str(a.threads if engine == 'lh' else 1),
               MKL_NUM_THREADS=str(a.threads if engine == 'lh' else 1),
               TIDE_LH_SEED=str(a.seed), TIDE_LH_WORK=str(a.work_count),
               TIDE_LH_OPERATOR_PROFILE=str(getattr(a, 'operator_profile', 0)))
    env.pop('TIDE_LH_AUDIT', None)
    if engine == 'pdg':
        env['OMP_WAIT_POLICY'] = 'PASSIVE'
    else:
        env.pop('OMP_WAIT_POLICY', None)  # Original LH library default, as in the measured pair.
    if engine == 'lh' and a.smoke and a.batch*a.vocab <= 4096:
        env['TIDE_LH_AUDIT'] = '1'
    return env
