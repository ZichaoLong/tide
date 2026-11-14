import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.run_direct_v10 import continuation_plan,makespan,M
from tide_lab.records import atomic_json,digest
from tide_lab.campaign_data import Corpus


class V10Contracts(unittest.TestCase):
    def test_extension_requires_whole_group_and_does_not_read_quality(self):
        groups=[dict(id='pair',ids=['a','b'])]
        rows=[dict(id=i,eligible=True,training_seconds=1000,nll=n) for i,n in [('a',9.),('b',2.)]]
        self.assertEqual(continuation_plan(groups,rows,128*M,4000)[0],['a','b'])
        bad=copy.deepcopy(rows);bad[1]['eligible']=False
        self.assertEqual(continuation_plan(groups,bad,128*M,100000)[0],[])
        bad=copy.deepcopy(rows);bad[1]['training_seconds']=float('nan')
        self.assertEqual(continuation_plan(groups,bad,128*M,100000)[0],[])
        self.assertEqual(continuation_plan(groups,rows,128*M,2000)[0],[])

    def test_extension_accounts_for_more_cases_than_workers(self):
        groups=[dict(id='a',ids=['a']),dict(id='b',ids=['b'])]
        rows=[dict(id=i,eligible=True,training_seconds=1000) for i in ('a','b')]
        self.assertEqual(makespan([1000,1000],1),2000)
        self.assertEqual(continuation_plan(groups,rows,128*M,3500,workers=1)[0],['a'])
        self.assertEqual(continuation_plan(groups,rows,128*M,3500,workers=2)[0],['a','b'])

    def test_confirmation_is_explicit_and_test_stays_sealed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);splits={}
            for split in ('train','validation','confirmation','test'):
                p=root/(split+'.u16');np.asarray([10,11,12],dtype='<u2').tofile(p)
                splits[split]=dict(file=p.name,sha256=digest(p),tokens=3,documents=[dict(id=split,offset=0,tokens=3)])
            # A bad test hash proves neither construction opens the sealed split.
            splits['test']['sha256']='sealed'
            atomic_json(root/'data-lock.json',dict(schema='tide-language-corpus-v6',dtype='<u2',bos=1,pad=0,splits=splits))
            ordinary=Corpus(root,3);self.assertEqual(set(ordinary.arrays),{'train','validation'})
            confirm=Corpus(root,3,extra_splits=('confirmation',))
            self.assertEqual(confirm.batch('confirmation',[0])['labels'].tolist(),[[10,11,12]])
            with self.assertRaisesRegex(ValueError,'sealed'):confirm.batch('test',[0])
            with self.assertRaises(ValueError):Corpus(root,3,extra_splits=('test',))

    def test_frozen_table_preserves_capacity_exposure_and_comparisons(self):
        root=Path(__file__).resolve().parents[1]/'configs/studies/language/direct-v10/table'
        catalog=json.loads((root/'catalog.json').read_text());self.assertEqual(len(catalog['cases']),20)
        ids=[i for g in catalog['continuation_groups'] for i in g['ids']]
        self.assertCountEqual(ids,catalog['cases']);self.assertEqual(len(set(ids)),20)
        self.assertEqual(catalog['main_target_tokens'],128*M)
        for cid in ids:
            c=json.loads((root/'main'/(cid+'.json')).read_text());t=c['training']
            self.assertLessEqual(c['planned_inventory']['lm_trainable'],200000000)
            self.assertEqual(t['lr_decay_tokens'],256*M);self.assertEqual(t['global_batch'],64)
            self.assertLessEqual(t['batch'],64);self.assertEqual(t['length'],256)
            self.assertIn(c['seed'],(11,22))
            if 'direct' in c:
                self.assertEqual(c['candidate']['depth'],2);self.assertFalse(c['candidate']['tree'])
        self.assertFalse(catalog['confirmation']['selection_feedback'])
        self.assertFalse(catalog['confirmation']['test_access'])


if __name__=='__main__':unittest.main()
