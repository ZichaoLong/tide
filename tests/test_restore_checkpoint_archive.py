import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.restore_checkpoint_archive import main, object_path, read_object, restore


class CheckpointArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.payload = b"frozen checkpoint bytes" * 200
        self.entry = {
            "path": ".local/pilot-r3/campaign-r01/checkpoints/model.pt",
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size": len(self.payload),
            "mtime_ns": 1719590473000123456,
            "mode": 0o640,
        }
        self.target = self.root / self.entry["path"]
        self.blob = object_path(self.root / "archive", self.entry["sha256"])
        self.blob.parent.mkdir(parents=True)
        self.compress(self.payload)

    def compress(self, data):
        self.blob.write_bytes(subprocess.check_output(["zstd", "-q", "-c"], input=data))

    def test_restore_and_repeat(self):
        read_object(self.blob, self.entry)
        self.assertTrue(restore(self.root, self.entry, self.blob))
        self.assertEqual(self.target.read_bytes(), self.payload)
        before = self.target.stat()
        self.assertEqual(stat.S_IMODE(before.st_mode), self.entry["mode"])
        self.assertEqual(before.st_mtime_ns, self.entry["mtime_ns"])
        self.assertFalse(restore(self.root, self.entry, self.blob))
        self.assertEqual(self.target.stat().st_ino, before.st_ino)

    def test_conflicting_original_is_preserved(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"a different checkpoint")
        with self.assertRaises(ValueError):
            restore(self.root, self.entry, self.blob)
        self.assertEqual(self.target.read_bytes(), b"a different checkpoint")

    def test_region_manifest_and_store_use_original_bytes_and_hash(self):
        base = "east:/project/archive"
        manifest = json.dumps({
            "schema": "tide-checkpoint-archive-v1", "stores": {"sfs": "/old/private"},
            "files": [dict(self.entry, store="sfs")],
        }).encode()
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        sources = {base + "/recovery-manifest.json": manifest,
                   object_path(base, self.entry["sha256"]): self.blob.read_bytes()}

        def fetch(args, **unused):
            self.assertEqual(args[:3], ["maflow", "fs", "cp"])
            self.assertEqual(args[args.index("--control") + 1], "east")
            Path(args[4]).write_bytes(sources[args[3]])

        argv = ["restore", "--manifest", base + "/recovery-manifest.json",
                "--manifest-sha256", manifest_sha, "--store", "sfs=" + base,
                "--root", str(self.root), "--under", self.entry["path"]]
        with patch("sys.argv", argv), patch("subprocess.run", side_effect=fetch) as transfer:
            main()
            self.assertEqual(self.target.read_bytes(), self.payload)
            self.assertEqual(transfer.call_count, 2)
            main()
            self.assertEqual(transfer.call_count, 3)
            sources[argv[2]] += b" "
            with self.assertRaises(SystemExit):
                main()
            self.assertEqual(transfer.call_count, 4)

    def test_wrong_content_and_truncated_frame_leave_no_output(self):
        self.compress(b"x" * len(self.payload))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            restore(self.root, self.entry, self.blob)
        self.compress(self.payload)
        self.blob.write_bytes(self.blob.read_bytes()[:-1])
        with self.assertRaises((ValueError, OSError)):
            restore(self.root, self.entry, self.blob)
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_symlink_in_destination_is_rejected(self):
        (self.root / "elsewhere").mkdir()
        (self.root / ".local").symlink_to(self.root / "elsewhere", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            restore(self.root, self.entry, self.blob)
        self.assertEqual(list((self.root / "elsewhere").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
