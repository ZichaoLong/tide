import unittest

import torch

from tide_lab.tasks import RetrievalTask, NoMemoryClassifier
from tide_lab.diagnostics import swap_query, extend_delay


class RetrievalTests(unittest.TestCase):
    def test_decoding_and_split_membership(self):
        task = RetrievalTask()
        maps = {}
        for split in ("train", "val", "test"):
            tokens, targets = task.batch(split, 128, step=0, seed=29)
            maps[split] = set()
            for row, target in zip(tokens, targets):
                pairs, _, answer = task.decode(row)
                self.assertEqual(answer, int(target))
                self.assertEqual(task.mapping_split(pairs), split)
                maps[split].add(tuple(sorted(pairs)))
        self.assertFalse(maps["train"] & maps["val"])
        self.assertFalse(maps["train"] & maps["test"])
        self.assertFalse(maps["val"] & maps["test"])

    def test_resume_cursor_and_rng_isolation(self):
        task = RetrievalTask()
        before = torch.get_rng_state().clone()
        one = task.batch("train", 8, step=31, seed=19)
        two = task.batch("train", 8, step=31, seed=19)
        for a, b in zip(one, two):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.equal(one[0], task.batch("train", 8, step=32, seed=19)[0]))

    def test_query_only_control_cannot_see_history(self):
        task = RetrievalTask()
        model = NoMemoryClassifier(task.vocab_size, task.value_vocab_size)
        x, _ = task.batch("test", 4, step=0, seed=17)
        changed = x.clone()
        changed[:, :-1] = 0
        torch.testing.assert_close(model(x), model(changed), atol=0, rtol=0)

    def test_paired_interventions_preserve_mapping_and_split(self):
        task=RetrievalTask()
        x,y=task.batch('val',32,0,20261107)
        indices,changed,target=swap_query(task,x)
        for index,row,label in zip(indices,changed,target):
            before,query,answer=task.decode(x[index])
            after,new_query,new_answer=task.decode(row)
            self.assertEqual(before,after)
            self.assertNotEqual(query,new_query)
            self.assertNotEqual(answer,new_answer)
            self.assertEqual(int(label),new_answer)
            self.assertEqual(task.mapping_split(after),'val')
        for delay in (4,8):
            longer=extend_delay(task,x,delay)
            other=RetrievalTask(dict(task.config.to_dict(),delay=delay))
            for before,row,label in zip(x,longer,y):
                self.assertEqual(task.decode(before),other.decode(row))
                self.assertEqual(other.decode(row)[2],int(label))


if __name__ == "__main__":
    unittest.main()
