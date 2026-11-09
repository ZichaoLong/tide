import json
from pathlib import Path
import unittest
import torch
from scripts.prepare_language_confirmation import select_confirmation
from scripts.run_on_light_device import candidates
from tide_lab.language import SequenceCore5
from tide_lab.language_checks import core_axes


class DepthTests(unittest.TestCase):
    def test_heldout_excludes_all_previously_observed_batches(self):
        docs=[dict(id=str(i),split='test') for i in range(62)]
        old={'selected_documents':{'test':[dict(id=str(i)) for i in range(16)]}}
        extra={'selected_documents':{'test':[dict(id=str(i)) for i in range(16,32)]}}
        chosen=select_confirmation(docs,old,seed=20261110,additional_locks=[extra])
        self.assertEqual(len(chosen),16)
        self.assertTrue(all(int(d['id'])>=32 for d in chosen))

    def test_shared_admission_uses_two_observations_and_headroom(self):
        def row(util,used):return dict(health='OK',aicore_percent=util,hbm_used_mib=used,hbm_total_mib=65536,processes=[{'pid':12}])
        a={0:row(0,3000),1:row(0,20000),2:row(0,60000)}
        b={0:row(90,3000),1:row(3,20000),2:row(0,60000)}
        self.assertEqual(candidates(a,b,dict(candidate_ids=[0,1,2],max_aicore_percent=20,minimum_free_hbm_mib=16384)),[1])

    def test_serial_and_fork_join_have_live_cross_region_gradients(self):
        torch.set_num_threads(1)
        for topology,depth in [('fc',4),('hb',1)]:
            with self.subTest(topology=topology):
                options=dict(width=8,topology=topology,depth=depth,region_width=4,send='dense',temperature=.25)
                model=SequenceCore5(seed=11,**options)
                report=core_axes(model,options,11,torch.device('cpu'))
                facts=report['depth_facts'];self.assertEqual(facts['regions'],4)
                self.assertGreater(facts['internal_messages'],0)
                self.assertTrue(all(r['node_parameter_vjp_l1']>0 for r in facts['region_events']))


if __name__=='__main__':unittest.main()
