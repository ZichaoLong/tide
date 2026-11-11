#!/usr/bin/env bash
# Reviewed one-bundle publication helper. Default: read-only verification.
# The user, not the experiment agent, invokes --publish with sudo.
set -euo pipefail

staging=/mi/data2T/zlong/home/llm/public-models-staging-20261111
destination=/mi/data2T/models/public-checkpoints-20261111
catalog_sha256=c96f4d921b6d3945c355b5c5c7d5abcb45a82f7d940baa4041bfc4604995a069
mode=${1:---check}
fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ $# -le 1 && ( "$mode" == --check || "$mode" == --publish ) ]] || fail 'Use --check or --publish.'
[[ "$mode" != --publish || "$EUID" == 0 ]] || fail '--publish needs sudo.'
[[ -d /mi/data2T/models && ! -L /mi/data2T/models ]] || fail 'Shared parent is missing or a symlink.'

needs_move=false
if [[ -L "$staging" ]]; then
    [[ $(readlink -- "$staging") == "$destination" && -d "$destination" && ! -L "$destination" ]] || fail 'Unexpected staging link or shared destination.'
    bundle=$destination
elif [[ -d "$staging" ]]; then
    [[ ! -e "$destination" && ! -L "$destination" ]] || fail 'Destination exists; refusing to merge or overwrite.'
    bundle=$staging
    needs_move=true
elif [[ ! -e "$staging" && -d "$destination" && ! -L "$destination" ]]; then
    # Recover a prior invocation interrupted after rename but before symlink.
    bundle=$destination
else
    fail 'No prepared bundle was found.'
fi
[[ $(stat -c %d -- "$bundle") == $(stat -c %d -- /mi/data2T/models) ]] || fail 'Different filesystems; refusing an implicit large copy.'

/usr/bin/python3 - "$bundle" "$catalog_sha256" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys

root, expected = Path(sys.argv[1]), sys.argv[2]
if root.resolve() != root:
    raise SystemExit('Bundle ancestors must not be symlinks.')
raw = (root/'CATALOG.json').read_bytes()
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit('Prepared catalog hash differs; refusing publication.')
m = json.loads(raw)
if m['state'] != 'completed' or len(m['models']) != 6 or m['weights'] != 27:
    raise SystemExit('Expected six fully verified models and 27 weight shards.')
allowed = {'CATALOG.json', 'UPSTREAM-CATALOG.json', 'SHA256SUMS', 'README.md'}
for row in m['files']:
    rel = Path(row['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise SystemExit('Unsafe manifest path.')
    p = root/rel
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or (
            s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) != (
            row['inode'], row['device'], row['bytes'], row['mtime_ns']):
        raise SystemExit(f'File identity changed after full hashing: {p}')
    allowed.add(str(rel))
for p in root.rglob('*'):
    if p.is_symlink() or not (p.is_dir() or p.is_file()):
        raise SystemExit(f'Unexpected special file/symlink: {p}')
    if p.is_file() and str(p.relative_to(root)) not in allowed:
        raise SystemExit(f'Unexpected added file: {p}')
if hashlib.sha256((root/'UPSTREAM-CATALOG.json').read_bytes()).hexdigest() != m['catalog_sha256']:
    raise SystemExit('Upstream catalog changed.')
expected_sums = ''.join(f"{r['sha256']}  {r['path']}\n" for r in m['files'])
if (root/'SHA256SUMS').read_text() != expected_sums:
    raise SystemExit('Checksum list differs from verified catalog.')
for model in m['models']:
    old = Path(model['original_path'])
    if not old.is_symlink() or str(old.readlink()) != str(Path(m['staging'])/model['relative_path']):
        raise SystemExit(f'Original model alias changed: {old}')
print(f'Prepared identity verified: {len(m["models"])} models, {len(m["files"])} files, {m["weights"]} shards.')
print('This checks the saved hashes and current inode/size/mtime; it does not rehash 94 GiB.')
PY

if [[ "$mode" == --check ]]; then
    printf 'Read-only check passed. Destination: %s\n' "$destination"
    exit 0
fi
if [[ "$needs_move" == true ]]; then
    mv -T --no-clobber -- "$staging" "$destination"
    [[ ! -e "$staging" && ! -L "$staging" ]] || fail 'Rename did not complete; stopping.'
fi
if [[ ! -L "$staging" ]]; then
    ln -sT -- "$destination" "$staging"
fi
# Root ownership, not pathname relocation, transfers this space out of user quota.
chown -R --no-dereference -- root:root "$destination"
chmod -R -- a+rX,go-w "$destination"
printf 'Published: %s\nOriginal model paths remain valid through symlinks.\n' "$destination"
stat -c '%U:%G %a %n' -- "$destination"
quota -w -u zlong
