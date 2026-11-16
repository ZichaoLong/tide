import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.resume_hb_v12_six_card import DEVICES, MODEL_COMMIT, SixCardWorker, validate_plan, try_preparation_lock
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

    def test_four_card_policy_cannot_admit_old_six_card_set(self):
        plan = dict(self.plan, card_limit=4, run_prefix='tl-v12-four-')
        policy = dict(self.policy, candidate_ids=[4, 9, 12, 14], max_new_concurrent_npus=4)
        validate_plan(plan, dict(commit=MODEL_COMMIT), policy, 1000)
        with self.assertRaisesRegex(ValueError, 'placement'):
            validate_plan(plan, dict(commit=MODEL_COMMIT), self.policy, 1000)
        with self.assertRaisesRegex(ValueError, 'namespace'):
            validate_plan(dict(plan, run_prefix='tl-v12-six-'), dict(commit=MODEL_COMMIT), policy, 1000)

    def test_four_card_recording_keeps_distinct_segment_names(self):
        worker = object.__new__(SixCardWorker)
        worker.plan = dict(self.plan, card_limit=4, run_prefix='tl-v12-four-')
        case = dict(train_arguments=['--until-tokens', str(128*2**20), '--run-id', 'tl-v12-six-main-test'])
        with patch('scripts.run_hb_v12_repair_worker.RepairWorker.stage') as stage:
            worker.stage(case, 'prefix', 'prefix', 1200, ['--run-id', 'tl-v12-repair-prefix-test'])
        self.assertEqual(case['train_arguments'][-1], 'tl-v12-four-main-test')
        self.assertEqual(stage.call_args.args[4], ['--run-id', 'tl-v12-four-prefix-test'])
        self.assertEqual(case['train_arguments'][1], str(128*2**20))

    def test_all_cards_require_bounded_temporary_checkpoint_phases(self):
        plan = dict(self.plan, card_limit=16, run_prefix='tl-v12-all-', preparation_slots=2)
        policy = dict(self.policy, candidate_ids=list(range(16)), max_new_concurrent_npus=16)
        validate_plan(plan, dict(commit=MODEL_COMMIT), policy, 1000)
        with self.assertRaisesRegex(ValueError, 'preparation slots'):
            validate_plan(dict(plan, preparation_slots=12), dict(commit=MODEL_COMMIT), policy, 1000)

    def test_preparation_lock_bounds_concurrency_and_releases(self):
        directory = self.root/'locks'
        left, lslot = try_preparation_lock(directory, 2)
        right, rslot = try_preparation_lock(directory, 2)
        self.addCleanup(left.close); self.addCleanup(right.close)
        self.assertNotEqual(lslot, rslot)
        self.assertEqual(try_preparation_lock(directory, 2), (None, None))
        left.close()
        replacement, slot = try_preparation_lock(directory, 2)
        self.addCleanup(replacement.close); self.assertEqual(slot, lslot)

    def test_temporary_recovery_slots_do_not_limit_formal_training(self):
        worker = object.__new__(SixCardWorker)
        worker.plan = dict(self.plan, preparation_slots=2, card_limit=16, run_prefix='tl-v12-all-')
        with patch.object(worker, 'enter_preparation') as enter, patch.object(worker, 'leave_preparation') as leave, \
                patch('scripts.run_hb_v12_repair_worker.RepairWorker.stage'):
            worker.stage({}, 'recovery-full', 'segment', 1200)
            enter.assert_called_once(); leave.assert_not_called()
            worker.stage({}, 'prefix', 'prefix', 1200)
            worker.stage({}, 'train', 'train', 1200)
            self.assertEqual(leave.call_count, 2)

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
