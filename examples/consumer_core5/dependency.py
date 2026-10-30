"""Capture/verify an installed dependency lock; source provenance is declared.

The caller builds the wheel from the stated checkout and retains build evidence.
Byte matching authenticates neither Git provenance nor mathematical correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import zipfile
from recording import fingerprint, json_record, sha256

SCHEMA = 'tide-foundation-lock-v1'


def commit(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('a full immutable Git commit is required')
    return value


def installed_files():
    import tidegraph
    root = Path(tidegraph.__file__).resolve().parent
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    return root, files


def environment():
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                architecture=platform.machine(), cxx11_abi=torch.compiled_with_cxx11_abi())


def native_identity(location):
    if location is None:
        return None
    root = Path(location)
    libraries = list(root.glob('_tide_native*.so'))
    if len(libraries) != 1:
        raise ValueError('native directory must contain exactly one adapter')
    manifest = root/'build-manifest.json'
    m = json.loads(manifest.read_text())
    binary = libraries[0]
    if m['binary_sha256'].get(binary.name) != sha256(binary):
        raise ValueError('native binary/manifest mismatch')
    env = environment()
    if m['backend'] != 'cpu' or any(m[k] != env[k] for k in ('torch', 'architecture', 'cxx11_abi')):
        raise ValueError('starter requires a matching CPU native build')
    return dict(manifest_sha256=sha256(manifest), binary_name=binary.name,
                binary_sha256=sha256(binary), cpp_source_sha256=m['cpp_source_sha256'],
                build_source=commit(m['source']), backend=m['backend'])


def capture(wheel, source, math_revision, *, protocol_commit=None, native_library=None, dirty=False):
    commit(source); commit(math_revision); commit(protocol_commit or source)
    import tidegraph
    root, files = installed_files()
    with zipfile.ZipFile(wheel) as archive:
        expected = {n[len('tidegraph/'):]: hashlib.sha256(archive.read(n)).hexdigest()
                    for n in archive.namelist() if n.startswith('tidegraph/') and not n.endswith('/')}
    if not expected or files != expected:
        raise ValueError('installed package differs from the supplied wheel')
    return dict(schema=SCHEMA, foundation=dict(repository='https://github.com/ZichaoLong/tide',
        commit=commit(source), dirty=dirty, package_version=tidegraph.__version__,
        wheel=dict(name=Path(wheel).name, sha256=sha256(wheel)), installed_files_sha256=fingerprint(files)),
        mathematics=dict(version='tide-core-5', commit=commit(math_revision)),
        protocol=dict(version='tide-downstream-contract-v1', commit=commit(protocol_commit or source),
                      path='docs/downstream-contract.md'),
        native=native_identity(native_library), environment=environment())


def verify(path, *, native_library=None):
    import tidegraph
    lock = json.loads(Path(path).read_text())
    if lock.get('schema') != SCHEMA:
        raise ValueError('unknown foundation lock schema')
    f = lock['foundation']
    for value in (f['commit'], lock['mathematics']['commit'], lock['protocol']['commit']):
        commit(value)
    if type(f['dirty']) is not bool or lock['protocol']['version'] != 'tide-downstream-contract-v1':
        raise ValueError('invalid foundation lock/protocol')
    root, files = installed_files()
    if f['package_version'] != tidegraph.__version__ or f['installed_files_sha256'] != fingerprint(files):
        raise ValueError('installed dependency does not match lock')
    if lock['environment'] != environment():
        raise ValueError('runtime environment differs from lock; qualify before capturing a new lock')
    if native_library is not None and lock['native'] != native_identity(native_library):
        raise ValueError('native dependency does not match lock')
    return lock, dict(lock_sha256=sha256(path), library_path=str(root),
                     installed_files_sha256=fingerprint(files), environment=environment(),
                     native=native_identity(native_library))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--math-commit', required=True)
    parser.add_argument('--protocol-commit')
    parser.add_argument('--native-library')
    parser.add_argument('--development-dirty', action='store_true', help='record a development wheel; not a clean-source delivery')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    record = capture(a.wheel, a.source_commit, a.math_commit, protocol_commit=a.protocol_commit,
                     native_library=a.native_library, dirty=a.development_dirty)
    json_record(a.output, record)


if __name__ == '__main__':
    main()
