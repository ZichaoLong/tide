"""Target leakage, causal LM and independent public readout checks."""
import copy
from types import SimpleNamespace
import unittest
import torch
from tide_lab.language_data import parse_documents,deduplicate,shifted_block
from tide_lab.language import sequence_readout,SmallTransformer,trainable_inventory
from tidegraph.weighted import readout


class LanguageDataTests(unittest.TestCase):
    def test_articles_not_subsections_and_original_bytes(self):
        rows=['\n',' = One = \n','first\n',' = = Section = = \n','more\n',' = Two = \n','last\n']
        docs=parse_documents(rows,'train')
        self.assertEqual(len(docs),2)
        self.assertEqual(docs[0]['text'],''.join(rows[1:5]))
        self.assertEqual(docs[1]['title'],'Two')

    def test_shift_and_padding_have_no_target_leakage(self):
        x,y=shifted_block([6,9,2],length=5,bos=1,pad=0)
        self.assertEqual(x,[1,6,9,0,0]);self.assertEqual(y,[6,9,2,-100,-100])
        self.assertEqual(shifted_block([15,2],length=2,bos=1,pad=0),([1,15],[15,2]))

    def test_cross_split_overlap_is_not_silently_accepted(self):
        docs={s:parse_documents([f' = {s} = \n',s+' text\n'],s) for s in ('train','validation','test')}
        docs['train'].append(dict(docs['validation'][0],id='train-duplicate'))
        kept,removed=deduplicate(docs)
        self.assertEqual(len(kept['train']),1);self.assertEqual(removed[0]['id'],'train-duplicate')
        docs['test'].append(dict(docs['validation'][0],id='test-overlap'))
        with self.assertRaisesRegex(ValueError,'validation/test'):deduplicate(docs)


class LanguageReadoutTests(unittest.TestCase):
    def test_replacement_keeps_public_transport_frozen(self):
        torch.set_num_threads(1)
        from tide_lab.language import FrozenPrefixTail
        lm=SimpleNamespace(model=SimpleNamespace(norm=torch.nn.Identity(),layers=[]),lm_head=torch.nn.Linear(896,32))
        cfg=__import__('json').loads(__import__('pathlib').Path('configs/studies/language/initial.json').read_text())['core']
        stats=dict(x_mean=torch.zeros(896),y_mean=torch.zeros(896),x_scale=torch.tensor(1.),y_scale=torch.tensor(1.))
        model=FrozenPrefixTail(lm,'tide-dense',cfg,stats)
        graph=model.selected.core
        graph.graph_model.validate_binding(graph.binding.graph)
        fixed=[p for p in graph.parameters() if not p.requires_grad]
        self.assertGreater(len(fixed),0)
        self.assertEqual(sum(p.numel() for p in model.selected.parameters() if p.requires_grad),591412)

    def test_vectorized_public_payload_matches_independent_scalar_vjp(self):
        torch.set_num_threads(1)
        encoder=SimpleNamespace(binding=SimpleNamespace(graph=SimpleNamespace(outputs=(0,1))),
                                spec=SimpleNamespace(stride=5,rank=lambda node:3))
        features=torch.randn(2,3,4,dtype=torch.float64)
        values=torch.randn(10,5,dtype=torch.float64,requires_grad=True)
        # Include negative strengths and missing output coordinates.
        outputs=[];i=0
        for b in range(2):
            for p in range(3):
                for port in range(2):
                    if (b,p,port) in [(0,1,0),(1,2,1)]:continue
                    outputs.append((b,p*5+3,port,values[i]));i+=1
        result=SimpleNamespace(outputs=outputs)
        vector=sequence_readout(encoder,result,features)
        scalar=torch.stack([torch.stack([readout(outputs,[(p*5+3,0),(p*5+3,1)],features[b,p],batch=b)
                                         for p in range(3)]) for b in range(2)])
        torch.testing.assert_close(vector,scalar,rtol=0,atol=0)
        weights=torch.randn_like(vector)
        vg=torch.autograd.grad((vector*weights).sum(),values,retain_graph=True)[0]
        sg=torch.autograd.grad((scalar*weights).sum(),values)[0]
        torch.testing.assert_close(vg,sg,rtol=1e-12,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            sequence_readout(encoder,SimpleNamespace(outputs=outputs+[outputs[0]]),features)

    def test_transformer_future_tokens_cannot_change_prefix(self):
        torch.set_num_threads(1)
        model=SmallTransformer(vocab_size=32,width=16,layers=2,heads=4,ffn=64).eval()
        x=torch.tensor([[1,6,8,9],[1,3,7,10]])
        changed=x.clone();changed[:,2:]=torch.tensor([[11,12],[13,14]])
        torch.testing.assert_close(model(x)[:,:2],model(changed)[:,:2],rtol=0,atol=0)
        prefix=model(x[:,:2]);torch.testing.assert_close(model(x)[:,:2],prefix,rtol=1e-5,atol=1e-6)
        F=torch.nn.functional
        F.cross_entropy(model(x).flatten(0,1),torch.tensor([6,8,9,2,3,7,10,2])).backward()
        self.assertIs(model.embedding.weight,model.head.weight)
        self.assertGreater(float(model.embedding.weight.grad.norm()),0)
        self.assertGreater(float(model.blocks[0].qkv.weight.grad.norm()),0)


if __name__=='__main__':unittest.main()
