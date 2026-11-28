"""Restore byte-verified checkpoints from a project storage-cleanup manifest."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


SCHEMA = "tide-checkpoint-archive-v1"
RUNS = {"pilot-r1", "pilot-r2", "pilot-r3"}


def sync_directory(directory):
    fd = os.open(directory, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def regional_uri(value):
    return re.match(r"^(?:obs://|[\w.-]+:/)", str(value)) is not None


@contextmanager
def local_copy(value, expected=None):
    if not regional_uri(value):
        yield Path(value)
        return
    with tempfile.TemporaryDirectory(prefix="tide-checkpoint-archive-") as tmp:
        local = Path(tmp) / "content"
        command = ["maflow", "fs", "cp", str(value), str(local), "--via", "direct",
                   "--verify", "sha256", "--no-clobber", "--json"]
        if str(value).startswith("east:/"):
            command.extend(["--control", "east"])
        if expected:
            command.extend(["--sha256", expected])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        yield local


def object_path(store, digest):
    if not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("invalid checkpoint SHA-256")
    if regional_uri(store):
        return str(store).rstrip("/") + "/objects/" + digest[:2] + "/" + digest + ".zst"
    return Path(store) / "objects" / digest[:2] / (digest + ".zst")


def verify_stream(stream, entry, output=None):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        if size > entry["size"]:
            raise ValueError("checkpoint exceeds its recorded size")
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
    if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError("checkpoint size or SHA-256 mismatch: " + entry["path"])


def read_object(blob, entry, output=None):
    with local_copy(blob) as local, subprocess.Popen(
        ["zstd", "-q", "-d", "-c", "--", str(local)], stdout=subprocess.PIPE
    ) as decoder:
        try:
            verify_stream(decoder.stdout, entry, output)
        finally:
            decoder.stdout.close()
        if decoder.wait():
            raise OSError("zstd failed for " + str(blob))


def restore(root, entry, blob):
    relative = Path(entry["path"])
    if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 4
            or relative.parts[0] != ".local" or relative.parts[1] not in RUNS or relative.suffix != ".pt"):
        raise ValueError("unexpected checkpoint destination")
    target = root / relative
    if target.resolve() != target:
        raise ValueError("restore destination contains a symlink: " + str(target))
    if target.exists():
        with target.open("rb") as existing:
            verify_stream(existing, entry)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".restoring-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            read_object(blob, entry, output)
            output.flush()
            os.fchmod(output.fileno(), entry["mode"] & 0o777)
            os.utime(output.fileno(), ns=(entry["mtime_ns"], entry["mtime_ns"]))
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            with target.open("rb") as existing:
                verify_stream(existing, entry)
            return False
        sync_directory(target.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="local recovery manifest or maflow OBS URI")
    parser.add_argument("--manifest-sha256", help="pin the recovery manifest to the published SHA-256")
    parser.add_argument("--root", default=Path.cwd(), type=Path, help="repository or empty recovery directory")
    parser.add_argument("--under", required=True, type=Path, help="file or directory relative to the repository")
    parser.add_argument("--store", action="append", default=[], metavar="NAME=LOCATION",
                        help="relocated local directory or maflow OBS prefix such as east:/…")
    parser.add_argument("--verify-only", action="store_true", help="decode the archive without writing files")
    args = parser.parse_args()
    if args.manifest_sha256 and not re.fullmatch("[0-9a-f]{64}", args.manifest_sha256):
        parser.error("invalid recovery manifest SHA-256")
    with local_copy(args.manifest, args.manifest_sha256) as local:
        data = local.read_bytes()
    if args.manifest_sha256 and hashlib.sha256(data).hexdigest() != args.manifest_sha256:
        parser.error("recovery manifest SHA-256 mismatch")
    manifest = json.loads(data)
    if manifest["schema"] != SCHEMA or args.under.is_absolute() or ".." in args.under.parts:
        parser.error("invalid manifest or relative selection")
    stores = dict(manifest["stores"])
    for override in args.store:
        name, separator, directory = override.partition("=")
        if not separator or name not in stores or not directory:
            parser.error("unknown store or missing directory: " + override)
        stores[name] = directory if regional_uri(directory) else str(Path(directory).resolve())
    selected = [entry for entry in manifest["files"] if Path(entry["path"]).is_relative_to(args.under)]
    if not selected:
        parser.error("selection matches no archived files")
    restored = 0
    for entry in selected:
        blob = object_path(stores[entry["store"]], entry["sha256"])
        if args.verify_only:
            read_object(blob, entry)
        else:
            restored += restore(args.root.resolve(), entry, blob)
    print(f"verified {len(selected)} files, restored {restored}")


if __name__ == "__main__":
    main()
