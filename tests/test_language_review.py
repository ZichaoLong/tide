import copy
import unittest
from scripts.review_language_results import paired_documents


class PairedDocumentReviewTests(unittest.TestCase):
    def rows(self):
        return [dict(document_id='one',block_id='a',tokens=2,nll_sum=4.),
                dict(document_id='one',block_id='b',tokens=3,nll_sum=6.),
                dict(document_id='two',block_id='c',tokens=5,nll_sum=15.)]

    def test_identical_predictions_have_zero_interval_and_documents_are_clusters(self):
        rows=self.rows();result=paired_documents(rows,rows,draws=100)
        self.assertEqual(result['delta_nll'],0.)
        self.assertEqual(result['bootstrap_two_sided_95'],[0.,0.])
        self.assertEqual(result['documents'],2)

    def test_constant_per_token_shift_has_known_sign_and_magnitude(self):
        old=self.rows();new=copy.deepcopy(old)
        for row in new:row['nll_sum']+=.25*row['tokens']
        result=paired_documents(new,old,draws=100)
        self.assertEqual(result['delta_nll'],.25)
        self.assertEqual(result['bootstrap_two_sided_95'],[.25,.25])

    def test_mismatched_or_repeated_targets_are_rejected(self):
        rows=self.rows()
        for invalid in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):paired_documents(rows,invalid,draws=100)
        invalid=copy.deepcopy(rows);invalid[0]['tokens']+=1
        with self.assertRaises(ValueError):paired_documents(rows,invalid,draws=100)


if __name__=='__main__':unittest.main()
