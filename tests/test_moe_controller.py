import unittest
from scripts.run_moe_controls import eligible_result,select_within_shape,M


class ControllerTests(unittest.TestCase):
    def test_inner_failure_and_incomplete_validation_block(self):
        r=dict(state='completed',progress=dict(tokens=8*M),inference=dict(state='completed'))
        v=dict(state='completed',tokens=131072,nll=3.5)
        self.assertTrue(eligible_result(r,v,8*M))
        self.assertFalse(eligible_result(dict(r,state='failed'),v,8*M))
        self.assertFalse(eligible_result(r,dict(v,state='budget_truncated'),8*M))
        self.assertFalse(eligible_result(r,dict(v,nll=float('nan')),8*M))
        self.assertFalse(eligible_result(r,v,32*M))

    def test_stratified_slots_and_deterministic_ties(self):
        rows=[dict(id=f'{s}-{i}',shape=s,nll=n+i,eligible=True) for s,n in [('e20',3),('e64',5),('dense',2)] for i in range(3)]
        rows.append(dict(id='failed',shape='e64',nll=1,eligible=False))
        selected=select_within_shape(rows,2)
        self.assertEqual(len(selected),6)
        self.assertEqual(sum(r['shape']=='e64' for r in selected),2)
        self.assertNotIn('failed',[r['id'] for r in selected])
        rows=[dict(id=x,shape='a',nll=3,eligible=True) for x in ('b','a')]
        self.assertEqual(select_within_shape(rows,1)[0]['id'],'a')


if __name__=='__main__':unittest.main()
