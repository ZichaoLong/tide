"""Check a project venv against the explicitly loaded local Ascend baseline.

Metadata only: no Torch import or device initialization. Historical environments
must load their original module; a matching version string alone is insufficient.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
from durable_records import write_json


def validate(environ, base_prefix, prefix, versions, *, require_venv=False):
    required = ('ASCEND_PYTHON_ENV', 'ASCEND_CANN_VERSION', 'ASCEND_TORCH_VERSION',
                'ASCEND_TORCH_NPU_VERSION', 'ASCEND_CONSTRAINTS_FILE')
    for key in required:
        if not environ.get(key):
            raise ValueError(f'{key}: load the matching ascend core/dev module first')
    if Path(base_prefix).resolve() != Path(environ['ASCEND_PYTHON_ENV']).resolve():
        raise ValueError('venv base differs from loaded module; create a new venv or load its original baseline')
    if require_venv and Path(prefix).resolve() == Path(base_prefix).resolve():
        raise ValueError('private venv required; do not install into the public baseline')
    for package, key in (('torch', 'ASCEND_TORCH_VERSION'), ('torch-npu', 'ASCEND_TORCH_NPU_VERSION')):
        actual = versions[package].split('+')[0]
        if actual != environ[key]:
            raise ValueError(f'{package}: {versions[package]} differs from {key}={environ[key]}; recreate the private environment')


def inspect(*, require_venv=False):
    packages = {key: metadata.version(key) for key in ('torch', 'torch-npu')}
    validate(os.environ, sys.base_prefix, sys.prefix, packages, require_venv=require_venv)
    constraints = Path(os.environ['ASCEND_CONSTRAINTS_FILE'])
    return dict(schema='tide-local-ascend-environment-v1', python=sys.executable,
                base_prefix=str(Path(sys.base_prefix).resolve()), prefix=str(Path(sys.prefix).resolve()),
                cann=os.environ['ASCEND_CANN_VERSION'], packages=packages,
                module=os.environ.get('TIDE_ASCEND_MODULE'),
                constraints=str(constraints), constraints_sha256=hashlib.sha256(constraints.read_bytes()).hexdigest(),
                scope='environment identity only; build and device checks are separate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-venv', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect(require_venv=args.require_venv)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2))
