import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tide_lab.disk_usage import allocated_usage


class DiskUsageTests(unittest.TestCase):
    def test_allocated_bytes_hardlink_symlink_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=p/'state';a.write_bytes(b'x'*8192)
            os.link(a,p/'same');(p/'link').symlink_to('/not/a/scan/target')
            expected=sum(x.lstat().st_blocks*512 for x in (p,a,p/'link'))
            result=allocated_usage([p,p/'vanished.tmp'])
            self.assertEqual(result['bytes'],expected);self.assertEqual(result['vanished'],1)

    def test_atomic_replace_race_does_not_mask_io_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'progress.json.tmp';f.write_text('x')
            actual=os.stat
            def racing(path,**kw):
                if os.fspath(path)==str(f):raise FileNotFoundError(str(f))
                return actual(path,**kw)
            with patch('tide_lab.disk_usage.os.stat',side_effect=racing):
                self.assertEqual(allocated_usage([p])['vanished'],1)
            with patch('tide_lab.disk_usage.os.stat',side_effect=PermissionError('denied')):
                with self.assertRaises(PermissionError):allocated_usage([p])


if __name__=='__main__':unittest.main()
