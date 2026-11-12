import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.retire_archived_checkpoints import checked, file_row, mutate, scope_check


class ArchivedCheckpointRetirementTests(unittest.TestCase):
    def test_scope_rejects_live_campaign_failed_witness_and_non_checkpoint(self):
        for path in ('runs/v9-main-x/train/latest.pt',
                     'runs/v6-cover-hb-direct-received/train/latest.pt',
                     'runs/v8-cover-x/train/result.json',
                     '../runs/v8-cover-x/train/latest.pt'):
            with self.assertRaises(ValueError):
                scope_check(dict(root_id='tide',path=path,action='delete'))

    def test_changed_file_and_symlink_fail_before_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p=root/'x.pt';p.write_bytes(b'original')
            row=file_row(p,root);p.write_bytes(b'changed content')
            with self.assertRaises(ValueError):checked(root,row)
            alias=root/'alias.pt';alias.symlink_to(p)
            with self.assertRaises(ValueError):checked(root,file_row(alias,root))
            self.assertTrue(p.exists())

    def test_verified_duplicate_keeps_both_paths_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();keep=root/'latest.pt';old=root/'previous.pt'
            keep.write_bytes(b'checkpoint');old.write_bytes(b'checkpoint')
            sha=hashlib.sha256(b'checkpoint').hexdigest()
            row=dict(file_row(old,root),action='deduplicate',keeper='latest.pt',sha256=sha,keeper_sha256=sha)
            mutate(root,row)
            self.assertTrue(old.samefile(keep));self.assertEqual(old.read_bytes(),b'checkpoint')

    def test_nonidentical_duplicate_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();keep=root/'latest.pt';old=root/'previous.pt'
            keep.write_bytes(b'new');old.write_bytes(b'old')
            row=dict(file_row(old,root),action='deduplicate',keeper='latest.pt',sha256='old',keeper_sha256='new')
            with self.assertRaises(ValueError):mutate(root,row)
            self.assertEqual(old.read_bytes(),b'old');self.assertEqual(keep.read_bytes(),b'new')


if __name__=='__main__':unittest.main()
