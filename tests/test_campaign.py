"""V6 contracts: candidate union, sealed documents and reproducible data cursor."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tide_lab.records import atomic_json,digest
from tide_lab.campaign_config import local_candidates,anchor
from tide_lab.campaign_data import Corpus


class CampaignContracts(unittest.TestCase):
    def test_all_declared_choices_are_legal_and_families_and_sends_covered(self):
        from tidegraph.planning.recipes import model_recipe
        families=set();sends=set();reads=set();sources=set()
        for item in local_candidates():
            m,_=model_recipe(**item['recipe'])
            families.update(x.get('memory_family','kv') for x in m['memories'])
            sends.update(x['selection_mode'] for x in m['regions'])
            reads.update(x['read_mode'] for x in m['regions'])
            sources.add(item['recipe']['source_mode'])
        self.assertEqual(families,{'kv','affine','delta','gated_delta','ssd'})
        self.assertEqual(len(sends),10);self.assertEqual(reads,{'0','-','+'})
        self.assertEqual(sources,{'region','node','source','adapted'})

    def test_document_boundaries_sealed_test_and_batch_independent_cursor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);splits={}
            for split in ('train','validation','test'):
                f=root/(split+'.u16');np.asarray([10,11,12,13,20,21,22],dtype='<u2').tofile(f)
                splits[split]=dict(file=f.name,sha256=digest(f),tokens=7,documents=[
                    dict(id='a',offset=0,tokens=4),dict(id='b',offset=4,tokens=3)])
            atomic_json(root/'data-lock.json',dict(schema='tide-language-corpus-v6',dtype='<u2',bos=1,pad=0,splits=splits))
            c=Corpus(root,3)
            one,end=c.sample_ids(0,7,11);a,cursor=c.sample_ids(0,2,11);b,cursor=c.sample_ids(cursor,5,11)
            self.assertEqual(one,a+b);self.assertEqual(end,cursor)
            rows=c.batch('train',[1,2])
            self.assertEqual(rows['input_ids'].tolist(),[[1,0,0],[1,20,21]])
            self.assertEqual(rows['labels'].tolist(),[[13,-100,-100],[20,21,22]])
            with self.assertRaisesRegex(ValueError,'sealed'):c.batch('test',[0])
            self.assertEqual(c.evaluation_ids('validation',6),[0,2])

    def test_explicit_recipe_preserves_existing_anchor_and_opens_other_memory(self):
        import torch
        from tide_lab.language import SequenceCore5
        torch.set_num_threads(1)
        legacy=SequenceCore5(width=32,depth=4,send='sparse',temperature=.25,seed=11)
        explicit=SequenceCore5(width=32,recipe=anchor(),seed=11)
        self.assertEqual(legacy.model_id,explicit.model_id)
        x=torch.randn(2,3,32)
        self.assertTrue(torch.equal(legacy(x),explicit(x)))
        other=SequenceCore5(width=32,recipe=anchor(family='ssd'),seed=11)
        y=other(x);y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in other.parameters()))


if __name__=='__main__':unittest.main()
