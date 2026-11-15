import copy
import unittest
from scripts.capacity_v11_extension_stage import validate_extension

class ExtensionGuardTests(unittest.TestCase):
    def setUp(self):
        self.old={'kind':'moe','moe':{'experts':64},'training':{'target_tokens':128*2**20,'lr_decay_tokens':256*2**20,'milestones':[8*2**20,128*2**20],'lr':.001}}
        self.new=copy.deepcopy(self.old)
        self.new['training'].update(target_tokens=256*2**20,milestones=[8*2**20,128*2**20,256*2**20])
    def test_only_endpoint(self):validate_extension(self.old,self.new)
    def test_reject_changes(self):
        for key,value in [('lr',.002),('lr_decay_tokens',512*2**20),('target_tokens',128*2**20)]:
            c=copy.deepcopy(self.new);c['training'][key]=value
            with self.assertRaises(ValueError):validate_extension(self.old,c)
        c=copy.deepcopy(self.new);c['moe']['experts']=20
        with self.assertRaises(ValueError):validate_extension(self.old,c)
    def test_preserve_original_schedule(self):
        self.old['training']['lr_decay_tokens']=128*2**20
        with self.assertRaises(ValueError):validate_extension(self.old,self.new)
