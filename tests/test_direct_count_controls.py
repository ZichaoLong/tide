import tempfile
import unittest
from pathlib import Path
import numpy as np
from tide_lab.campaign_data import Corpus
from tide_lab.records import atomic_json,digest
from scripts.direct_count_controls import stream_counts


class CountControlTests(unittest.TestCase):
    def test_reusing_corpus_for_another_seed_matches_a_fresh_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);splits={}
            for split in ('train','validation'):
                p=root/(split+'.u16');np.arange(4,204,dtype='<u2').tofile(p)
                splits[split]=dict(file=p.name,sha256=digest(p),tokens=200,
                    documents=[dict(id=split,offset=0,tokens=200)])
            atomic_json(root/'data-lock.json',dict(schema='tide-language-corpus-v6',dtype='<u2',bos=1,pad=0,splits=splits))
            corpus=Corpus(root,2)
            first,_,_=stream_counts(corpus,11,(16,))
            actual,pairs,boundaries=stream_counts(corpus,22,(16,))
            expected,reference,ref_boundaries=stream_counts(Corpus(root,2),22,(16,))
            self.assertFalse(np.array_equal(first,actual))
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(pairs,reference)
            self.assertEqual(boundaries,ref_boundaries)
            self.assertEqual(int(pairs.sum()),16)


if __name__=='__main__':unittest.main()
