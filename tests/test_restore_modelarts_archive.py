import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.restore_modelarts_archive import lookup, restore, source_key


class ModelArtsArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.data = b"frozen experiment evidence"
        self.sha = hashlib.sha256(self.data).hexdigest()
        (self.cache / self.sha).write_bytes(self.data)
        self.name = ".local/pilot-r3/old/checkpoint.pt"
        self.row = {"storage": "object", "key": "archive/" + self.sha,
                    "size": len(self.data), "sha256": self.sha, "mode": 0o640,
                    "mtime_ns": 1620304010123456789}
        bucket = hashlib.sha256(self.name.encode()).hexdigest()[:2]
        wire = gzip.compress(json.dumps({"schema": "tide-r3-recovery-paths/v1",
                                         "shard": bucket, "paths": {self.name: self.row}}).encode())
        sha = hashlib.sha256(wire).hexdigest()
        (self.cache / sha).write_bytes(wire)
        self.index = {"source_root": str(self.root), "original_legacy_sfs": "/old-private/legacy",
                      "shards": {bucket: {"key": "archive/shard", "sha256": sha}}}

    def test_resolves_shard_and_recovers_into_symlinked_home(self):
        alias = self.root / "client-home"
        alias.symlink_to(self.root, target_is_directory=True)
        name = source_key(str(alias / self.name), self.index)
        self.assertEqual(name, self.name)
        self.assertEqual(source_key("/old-private/legacy/checkpoint", self.index),
                         ".external/legacy-sfs/checkpoint")
        self.assertEqual(lookup(self.index, name, self.cache), self.row)
        target = alias / "output" / "checkpoint"
        self.assertEqual(restore(self.row, name, target, self.index, self.cache), "restored")
        self.assertEqual(target.read_bytes(), self.data)
        self.assertEqual(target.stat().st_mtime_ns, self.row["mtime_ns"])
        self.assertEqual(target.stat().st_mode & 0o777, self.row["mode"])
        self.assertEqual(restore(self.row, name, target, self.index, self.cache), "existing")
        target.write_bytes(b"new contents")
        with self.assertRaises(FileExistsError):
            restore(self.row, name, target, self.index, self.cache)

    def test_symlink_and_missing_bytes_do_not_replace_existing_files(self):
        target = self.root / "link"
        link = {"storage": "symlink", "target": "../target", "mtime_ns": self.row["mtime_ns"]}
        self.assertEqual(restore(link, self.name, target, self.index, self.cache), "restored")
        self.assertEqual(os.readlink(target), "../target")
        self.assertEqual(restore(link, self.name, target, self.index, self.cache), "existing")
        with self.assertRaises(ValueError):
            restore({"storage": "unavailable"}, self.name, target, self.index, self.cache)
        self.assertEqual(os.readlink(target), "../target")
