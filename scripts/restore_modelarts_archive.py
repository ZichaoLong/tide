"""从华东归档按原路径查询或恢复 TIDE 实验文件。依赖已配置的 maflow 和 zstd。"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

DEFAULT_INDEX = "tide-learning-lab/production/r3-20261128/recovery/index.json"
BLOCK = 1024**2


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(ref, cache):
    expected = ref.get("sha256")
    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("无效的 SHA-256")
    key = ref["key"]
    if not key or any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("无效的华东对象路径")
    target = cache / (expected or hashlib.sha256(key.encode()).hexdigest())
    if not target.exists():
        cmd = ["maflow", "fs", "cp", "east:/" + key, str(target), "--verify", "sha256",
               "--via", "direct", "--control", "east", "--no-clobber", "--json"]
        if expected:
            cmd.extend(["--sha256", expected])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
    if expected and sha256_file(target) != expected:
        raise ValueError("华东对象内容 SHA-256 不符 " + key)
    return target


def source_key(name, index):
    path = Path(name)
    if path.is_absolute():
        # Old clients may spell the same home through a machine-local symlink.
        for candidate in (path.parent.resolve() / path.name, path):
            if candidate.is_relative_to(index["source_root"]):
                path = candidate.relative_to(index["source_root"])
                break
            if candidate.is_relative_to(index["original_legacy_sfs"]):
                path = Path(".external/legacy-sfs") / candidate.relative_to(index["original_legacy_sfs"])
                break
        else:
            raise ValueError("请输入实验仓相对路径或原归档中的绝对路径")
    if not path.parts or ".." in path.parts:
        raise ValueError("请输入有效的原始文件路径")
    return path.as_posix()


def lookup(index, name, cache):
    bucket = hashlib.sha256(name.encode()).hexdigest()[:2]
    with gzip.open(fetch(index["shards"][bucket], cache), "rt", encoding="utf-8") as stream:
        shard = json.load(stream)
    if shard["schema"] != "tide-r3-recovery-paths/v1" or shard["shard"] != bucket:
        raise ValueError("路径分片身份不符")
    if name not in shard["paths"]:
        raise ValueError("索引没有此路径 " + name)
    return shard["paths"][name]


def copy_verified(source, output, record):
    digest = hashlib.sha256()
    size = 0
    with output.open("xb") as target:
        for block in iter(lambda: source.read(BLOCK), b""):
            size += len(block)
            if size > record["size"]:
                raise ValueError("恢复文件超过登记长度")
            digest.update(block)
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
    if size != record["size"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("恢复文件长度或 SHA-256 不符")


def recover_file(record, name, temp, output, index, cache):
    storage = record["storage"]
    if storage == "object":
        obj = fetch({"key": record["key"], "sha256": record["sha256"]}, cache)
        with obj.open("rb") as stream:
            copy_verified(stream, output, record)
    elif storage == "archive":
        group = index["native_archives"][record["archive"]]
        manifest = json.loads(fetch(group["manifest"], cache).read_bytes())
        member = PurePosixPath(record["member"])
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("归档成员路径无效")
        if manifest["entries"][str(member)]["sha256"] != record["sha256"]:
            raise ValueError("原生包清单中的文件摘要不符")
        dest = temp / "native"
        subprocess.run(["maflow", "admin", "archive", "restore", record["archive"],
                        "--to", str(dest), "--json"], stdout=subprocess.DEVNULL, check=True)
        with (dest / member).open("rb") as stream:
            copy_verified(stream, output, record)
    elif storage in ("legacy-zstd", "tar"):
        ref = record["compressed"] if storage == "legacy-zstd" else index["evidence_packs"][record["group"]]["archive"]
        obj = fetch(ref, cache)
        with subprocess.Popen(["zstd", "-q", "-dc", "--", str(obj)], stdout=subprocess.PIPE) as decoder:
            try:
                if storage == "legacy-zstd":
                    copy_verified(decoder.stdout, output, record)
                else:
                    with tarfile.open(fileobj=decoder.stdout, mode="r|") as archive:
                        for member in archive:
                            if member.name != name:
                                continue
                            if not member.isreg() or member.size != record["size"]:
                                raise ValueError("包内文件类型或长度不符")
                            with archive.extractfile(member) as stream:
                                copy_verified(stream, output, record)
                    for _ in iter(lambda: decoder.stdout.read(BLOCK), b""):
                        pass
            finally:
                decoder.stdout.close()
            if decoder.wait() or not output.exists():
                raise OSError("归档解压或成员恢复失败")
    else:
        raise ValueError("未知存储方式 " + str(storage))


def restore(record, name, destination, index, cache):
    if record["storage"] == "unavailable":
        raise ValueError("迁移前此历史文件已退役，可通过索引查询原结果与摘要。")
    requested = destination.expanduser().absolute()
    dest = requested.parent.resolve() / requested.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    link = record["storage"] == "symlink"
    if os.path.lexists(dest):
        if link and dest.is_symlink() and os.readlink(dest) == record["target"]:
            return "existing"
        if (not link and not dest.is_symlink() and dest.is_file()
                and dest.stat().st_size == record["size"] and sha256_file(dest) == record["sha256"]):
            return "existing"
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(prefix=".tide-recover-", dir=dest.parent) as scratch:
        temp = Path(scratch)
        output = temp / "file"
        if link:
            output.symlink_to(record["target"])
        else:
            recover_file(record, name, temp, output, index, cache)
            output.chmod(record["mode"] & 0o7777)
        os.utime(output, ns=(record["mtime_ns"],) * 2, follow_symlinks=False)
        os.link(output, dest, follow_symlinks=False)
        fd = os.open(dest.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return "restored"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="实验仓相对路径、原实验绝对路径或原私有 SFS 文件的绝对路径")
    parser.add_argument("--to", type=Path, help="输出文件名。省略时只显示命中的索引记录。")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="华东索引的个人前缀相对键或 east:/ URI")
    parser.add_argument("--index-sha256", help="用项目交接的 SHA-256 锁定总索引")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tide-cloud-lookup-") as directory:
        cache = Path(directory)
        key = args.index.removeprefix("east:/")
        index = json.loads(fetch({"key": key, "sha256": args.index_sha256}, cache).read_bytes())
        if index["schema"] != "tide-r3-recovery/v1":
            raise ValueError("归档索引格式不符")
        name = source_key(args.source, index)
        row = lookup(index, name, cache)
        answer = {"source": name, "record": row}
        if args.to:
            answer.update(result=restore(row, name, args.to, index, cache), destination=str(args.to))
        print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
