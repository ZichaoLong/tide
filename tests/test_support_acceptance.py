import unittest
import torch
from tide_lab.language import SequenceCore5
from scripts.accept_positive_support import exact


class SupportConsumerTests(unittest.TestCase):
    def test_new_policy_has_new_identity_and_same_parameters(self):
        torch.set_num_threads(1)
        old=SequenceCore5(seed=11,width=8,depth=2,send='dense')
        new=SequenceCore5(seed=11,width=8,depth=2,send='dense',support_policy='positive-floor-v1')
        self.assertNotEqual(old.model_id,new.model_id)
        self.assertNotEqual(old.binding.graph.identity,new.binding.graph.identity)
        self.assertNotIn('support_policy',old.recipe['region_overrides'])
        self.assertEqual(list(dict(old.named_parameters())),list(dict(new.named_parameters())))
        new.load_state_dict(old.state_dict(),strict=True)
        exact(old.state_dict(),new.state_dict())

    def test_exact_checker_detects_optimizer_tensor_and_scalar_changes(self):
        original={'state':{0:{'exp_avg':torch.tensor([.1]),'step':3}}}
        with self.assertRaises(AssertionError):
            exact(original,{'state':{0:{'exp_avg':torch.tensor([.2]),'step':3}}})
        with self.assertRaises(AssertionError):
            exact(original,{'state':{0:{'exp_avg':torch.tensor([.1]),'step':4}}})


if __name__=='__main__':unittest.main()
