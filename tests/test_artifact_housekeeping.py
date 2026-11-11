import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.retire_priority_artifacts import checked_file
from scripts.stage_public_models import checkpoint_files, inspect_file, relocate


class HousekeepingSafetyTests(unittest.TestCase):
    def test_retirement_rejects_escape_symlink_and_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); p = root/'a'; p.write_bytes(b'old')
            s = p.stat(); row = dict(path='a', logical_bytes=3, mtime_ns=s.st_mtime_ns)
            self.assertEqual(checked_file(root, row)[0], p)
            with self.assertRaises(ValueError):
                checked_file(root, dict(row, path='../a'))
            (root/'alias').symlink_to(p)
            with self.assertRaises(ValueError):
                checked_file(root, dict(row, path='alias'))
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                checked_file(root, row)

    def test_catalog_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'weight'; p.write_bytes(b'abc')
            expected = dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
            self.assertTrue(inspect_file(p, expected)['upstream_verified'])
            with self.assertRaises(ValueError):
                inspect_file(p, dict(expected, sha256='0'*64))

    def test_relocation_preserves_inode_and_original_path_and_refuses_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); src = root/'old'; src.mkdir()
            f = src/'config.json'; f.write_text('{}'); inode=f.stat().st_ino
            dest = root/'stage'/'model'; relocate(src, dest)
            self.assertTrue(src.is_symlink())
            self.assertEqual((src/'config.json').stat().st_ino, inode)
            self.assertEqual((dest/'config.json').read_text(), '{}')
            extra = root/'other'; extra.mkdir()
            with self.assertRaises(FileExistsError):
                relocate(extra, dest)
            self.assertTrue(extra.is_dir())

    def test_checkpoint_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); model = root/'model'; model.mkdir()
            (model/'alias').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                checkpoint_files(model)


if __name__ == '__main__':
    unittest.main()
