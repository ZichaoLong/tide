"""Worker ownership must not reserve a case directory or hide its live state."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from scripts.submit_v6 import prepare
from scripts.campaign_status import status
from tide_lab.records import atomic_json


class SubmissionOwnership(unittest.TestCase):
    def test_prepared_workers_and_case_outputs_are_disjoint_and_status_follows_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';cfg=source/'configs/studies/language/v6/resolved'
            cfg.mkdir(parents=True);data=root/'data';data.mkdir()
            atomic_json(data/'state.json',dict(state='completed'));atomic_json(data/'data-lock.json',{})
            atomic_json(source/'.tide-source.json',dict(commit='0'*40));rows=[]
            for topology in ('fc','hb'):
                for i in range(91):
                    cid=f'cover-{topology}-c{i:03d}'
                    atomic_json(cfg/(cid+'.json'),dict(id=cid,resources=dict(max_rss_gib=1),training=dict(timeout_seconds=60)))
                    rows.append(dict(id=cid,line='coverage',device='cpu',threads=1))
            atomic_json(cfg/'catalog.json',dict(launchable=True,cases=rows))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(source=source,root=root,data=data,workers=96))
            manifest=json.loads((root/'.local/v6-submission/submission.json').read_text())
            worker_dirs={Path(j['worker_output']) for j in manifest['jobs']}
            case_dirs={Path(p) for j in manifest['jobs'] for p in j['case_outputs'].values()}
            self.assertEqual(len(worker_dirs),96);self.assertEqual(len(case_dirs),182)
            self.assertFalse(worker_dirs&case_dirs)
            self.assertFalse(any(w in c.parents or c in w.parents for w in worker_dirs for c in case_dirs))
            job=manifest['jobs'][0];cid=job['cases'][0];out=Path(job['case_outputs'][cid])
            atomic_json(root/'artifacts/jobs'/job['name']/'state.json',dict(state='running'))
            atomic_json(Path(job['worker_output'])/'state.json',dict(cases=[dict(id=cid,state='running',stages=[dict(name='train',state='running')])]))
            atomic_json(out/'train/result.json',dict(state='running'))
            atomic_json(out/'train/progress.json',dict(step=2,tokens=512))
            actual=next(c for c in status(root)['cases'] if c['id']==cid)
            self.assertEqual((actual['state'],actual['stage'],actual['tokens']),('running','train',512))
            atomic_json(root/'artifacts/jobs'/job['name']/'state.json',dict(state='completed'))
            pending=next(c for c in status(root)['cases'] if c['id']==job['cases'][1])
            self.assertEqual(pending['state'],'unchecked')
            self.assertIn('before this case started',pending['reason'])


if __name__=='__main__':unittest.main()
