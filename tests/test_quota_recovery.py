import json
from pathlib import Path
import tempfile
import unittest
from tide_lab.quota import parse_quota
from scripts.resume_direct_v9 import remaining_cases


class QuotaRecoveryTests(unittest.TestCase):
    def test_soft_warning_with_grace_is_not_hard_limit(self):
        r=parse_quota('/dev/example 314581608* 314572800 838860800 2000 1600254 0 0 0',1,at=1000)
        self.assertTrue(r['soft_exceeded'])
        self.assertGreater(r['hard_quota_headroom_gib'],499)

    def test_normal_and_low_headroom_remain_observable(self):
        r=parse_quota('/dev/example 700000000 0 838860800 0 100 0 0 0',0,at=1000)
        self.assertLess(r['hard_quota_headroom_gib'],200)
        self.assertFalse(r['soft_exceeded'])

    def test_expired_grace_and_bad_query_fail(self):
        for output,rc in [('/dev/example 301 300 800 999 100 0 0 0',1),
                          ('/dev/example 301 300 800 0 100 0 0 0',1),
                          ('quota: permission denied',1),
                          ('/dev/example 301 300 800 2000 100 0 0 0',2)]:
            with self.subTest(output=output,rc=rc),self.assertRaises(RuntimeError):
                parse_quota(output,rc,at=1000)

    def test_completed_skipped_and_checkpoint_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cases=[]
            for name,state in [('done','completed'),('cut','budget_truncated'),('new',None)]:
                out=root/name;out.mkdir();cases.append(dict(id=name,output=str(out)))
                if state:
                    (out/'train').mkdir();(out/'train/result.json').write_text(json.dumps(dict(state=state)))
            original={'phases':{'main':{'cases':cases}}}
            with self.assertRaises(ValueError):remaining_cases(original)
            (root/'cut/train/latest.pt').touch()
            self.assertEqual([(r['id'],s) for r,s in remaining_cases(original)],
                             [('cut','budget_truncated'),('new','not_started')])


if __name__=='__main__':unittest.main()
