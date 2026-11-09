"""The diagnostic must preserve a real failed row and scope its precision probes."""
import unittest
import torch
from tidegraph.weighted import allocate
from scripts.diagnose_allocation_underflow import analyze_failure


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


if __name__=='__main__':unittest.main()
