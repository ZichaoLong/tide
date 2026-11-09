"""The diagnostic must preserve a real failed row and scope its precision probes."""
import unittest
import torch
from tidegraph.weighted import allocate
from scripts.diagnose_allocation_underflow import analyze_failure
from scripts.replay_allocation_row import replay


class AllocationDiagnosisTests(unittest.TestCase):
    def test_positive_support_failure_is_preserved_while_fp64_represents_row(self):
        scores=torch.tensor([-50.,0.,0.,0.])
        try:
            allocate(scores,torch.ones(4),2.,2.,.25,regularizer='ent',ent_alpha=1.)
        except ArithmeticError as error:
            report=analyze_failure(error)
        else:self.fail('synthetic positive support was expected to underflow')
        self.assertEqual(report['captured']['tensors']['scores']['values'],scores.tolist())
        self.assertEqual(report['underflow_coordinates'],[0])
        self.assertEqual(report['probes'][0]['state'],'failed')
        higher=report['probes'][1]
        self.assertEqual(higher['state'],'completed')
        self.assertTrue(higher['all_positive_caps_have_positive_controls'])
        self.assertGreater(higher['controls'][0],0)
        self.assertLess(higher['controls'][0],1e-45)
        self.assertAlmostEqual(sum(higher['controls']),2.,places=12)
        self.assertEqual(report['probes'][-1]['state'],'completed')
        reconstructed=replay(report,torch.float64)
        self.assertEqual(reconstructed['state'],'completed')
        self.assertEqual(reconstructed['controls'],higher['controls'])

    def test_fp64_also_rejects_a_finite_more_extreme_shannon_row(self):
        scores=torch.tensor([-200.,0.,0.,0.],dtype=torch.float64)
        with self.assertRaisesRegex(ArithmeticError,'support underflowed'):
            allocate(scores,torch.ones_like(scores),2.,2.,.25,regularizer='ent',ent_alpha=1.)


if __name__=='__main__':unittest.main()
