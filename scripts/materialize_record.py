"""Restore the exact bytes of a losslessly archived experiment JSON record.

This does not recreate retired tensor payloads or alter qualification results.
Use a new explicit output path; existing files are never overwritten.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def _relative(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
        raise ValueError('archive paths must be normalized repository-relative paths')
    if not relative.parts or relative.parts[0] != 'artifacts':
        raise ValueError('archive is outside artifacts')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('archive path contains a symlink')
    return current


def materialize(index, record, output, *, root):
    root = Path(root).resolve()
    document = json.loads(Path(index).read_text())
    if document.get('schema') != 'tide-json-archive-index-v1':
        raise ValueError('unknown archive index')
    requested = Path(record)
    name = str(requested.relative_to(root)) if requested.is_absolute() else str(requested)
    entries = [row for row in document['records'] if row['path'] == name]
    if len(entries) != 1:
        raise ValueError('record must have exactly one archive index entry')
    row, = entries
    _relative(root, row['path'])
    archive = _relative(root, row['archive_path'])
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(output)
    # Stage beside the requested output. link() publishes without an overwrite
    # race, after both the compressed and decompressed identities are checked.
    descriptor, stage_name = tempfile.mkstemp(prefix='.record-', dir=output.parent)
    stage = Path(stage_name)
    try:
        with os.fdopen(descriptor, 'wb') as destination, archive.open('rb') as source:
            original_stat = os.fstat(source.fileno())
            compressed = hashlib.sha256()
            for block in iter(lambda: source.read(1 << 20), b''):
                compressed.update(block)
            if compressed.hexdigest() != row['archive_sha256']:
                raise ValueError('compressed archive SHA256 mismatch')
            source.seek(0)
            digest, size = hashlib.sha256(), 0
            with gzip.GzipFile(fileobj=source, mode='rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    size += len(block)
                    if size > row['original_bytes']:
                        raise ValueError('decompressed record exceeds recorded size')
                    digest.update(block)
                    destination.write(block)
            after = os.fstat(source.fileno())
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                    original_stat.st_size, original_stat.st_mtime_ns, original_stat.st_ctime_ns):
                raise ValueError('archive changed while reading')
            if size != row['original_bytes'] or digest.hexdigest() != row['original_sha256']:
                raise ValueError('original record size/SHA256 mismatch')
            destination.flush()
            os.fsync(destination.fileno())
        os.link(stage, output)
        stage.unlink()
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return dict(path=row['path'], output=str(output), bytes=size, sha256=digest.hexdigest())
    finally:
        if stage.exists():
            stage.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--record', required=True, help='original repository-relative or absolute JSON path')
    parser.add_argument('--output', type=Path, required=True, help='new explicit file; parent must exist')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(materialize(args.index, args.record, args.output, root=args.root)))


if __name__ == '__main__':
    main()
