import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.resume_hb_v12_six_card import DEVICES, MODEL_COMMIT, SixCardWorker, validate_plan
from tide_lab.records import digest


class SixCardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'config.json'; self.config.write_text(json.dumps({'training': {'target_tokens': 128*2**20}, 'study_id': 'test'}))
        parent = self.root/'parent.pt'; parent.write_bytes(b'fixed checkpoint stand-in')
        self.plan = dict(model_source=MODEL_COMMIT, deadline_epoch=1000, administrative_mode='resume_train',
                         original_config_sha256=digest(self.config), cases=[dict(id='test', config=str(self.config),
                         output=str(self.root/'run'), administrative_parent='old', parent_checkpoint=str(parent), parent_sha256=digest(parent))])
        self.policy = dict(candidate_ids=DEVICES, max_new_concurrent_npus=6, slot_namespace='tide-direct-v10', deadline_epoch=1000)

    def test_same_source_and_physical_set_required(self):
        validate_plan(self.plan, dict(commit=MODEL_COMMIT), self.policy, 1000)
        for key, value in [('candidate_ids', list(range(6))), ('max_new_concurrent_npus', 12), ('deadline_epoch', 1001)]:
            policy = dict(self.policy, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(self.plan, dict(commit=MODEL_COMMIT), policy, 1000)
        with self.assertRaises(ValueError):
            validate_plan(self.plan, dict(commit='new model source'), self.policy, 1000)

    def test_changed_parent_or_config_rejected(self):
        Path(self.plan['cases'][0]['parent_checkpoint']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'parent changed'):
            validate_plan(self.plan, dict(commit=MODEL_COMMIT), self.policy, 1000)
        self.config.write_text(json.dumps({'training': {'target_tokens': 256*2**20}}))
        with self.assertRaisesRegex(ValueError, 'endpoint required'):
            validate_plan(self.plan, dict(commit=MODEL_COMMIT), self.policy, 1000)

    def test_failed_recovery_cannot_enter_training(self):
        worker = object.__new__(SixCardWorker)
        worker.plan = copy.deepcopy(self.plan); worker.state = {'cases': []}
        worker.a = SimpleNamespace(deadline_epoch=1000)
        worker.publish = lambda: None; worker.guard = lambda: None
        with patch.object(worker, 'stage', return_value={'state': 'failed'}) as stage:
            with self.assertRaisesRegex(RuntimeError, 'recovery-full'):
                worker.run()
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(stage.call_args.args[1:3], ('recovery-full', 'segment'))


if __name__ == '__main__': unittest.main()
