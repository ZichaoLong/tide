"""Mechanism tests use random tiny models, not pre-trained model qualification."""
from copy import deepcopy
import pytest
import torch
from tidegraph.lm.config import resolve
from tidegraph.lm.inference import generate
from tidegraph.lm.inputs import create
from tidegraph.lm.checkpoint import host_tree, identity, inference_next, cache_state
from tidegraph.lm.training import update, optimizer_for


@pytest.fixture
def tiny():
    transformers = pytest.importorskip('transformers')
    torch.set_num_threads(2)
    torch.manual_seed(19)
    c = transformers.Qwen2Config(vocab_size=1024, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                                 attention_dropout=0, eos_token_id=2)
    return transformers.Qwen2ForCausalLM(c).eval()


def request(**task):
    return resolve(dict(schema='tide-lm-experiment-v1', model='qwen05', task=dict(batch=2,input_tokens=12,output_tokens=4,**task),
                        execution=dict(dtype='float32', prefill_chunk=4)))['request']


def test_independent_online_generation_matches_full_sequence(tiny):
    c = request(); ids = torch.tensor([[10,11,12,13,14,15,16,17,18,19,20,21],[4,5,7,8,10,11,12,13,14,15,16,17]])
    mask = torch.ones_like(ids)
    actual = generate(tiny, ids, mask, c, torch.device('cpu'), None)
    expected = []; sequence = ids
    with torch.no_grad():
        for _ in range(4):
            value = tiny(sequence, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            expected.append(value); sequence = torch.cat((sequence,value),dim=1)
    assert torch.equal(actual['tokens'], torch.cat(expected,dim=1))
    assert actual['work']['prefill_calls'][0]['rows'] == 8
    state = host_tree(dict(cache=cache_state(actual['cache']), mask=actual['mask'], last_token=actual['last_token']))
    assert inference_next(tiny,state,c,torch.device('cpu'),None) == inference_next(deepcopy(tiny),state,c,torch.device('cpu'),None)


def test_accumulation_preserves_global_optimizer_boundary(tiny):
    c = request(); c['task'].update(mode='full', batch=4, global_batch=4)
    ids = torch.arange(48).reshape(4,12)+10; mask = torch.ones_like(ids)
    second = deepcopy(tiny)
    first = update(tiny,optimizer_for(tiny),ids,mask,c,torch.device('cpu'),None)
    c['task']['microbatch'] = 4
    other = update(second,optimizer_for(second),ids,mask,c,torch.device('cpu'),None)
    assert first['accumulation_steps']==4 and other['accumulation_steps']==1
    assert first['loss']==pytest.approx(other['loss'],abs=1e-6)
    for a,b in zip(tiny.parameters(),second.parameters()):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
        assert a.grad is None and b.grad is None


def test_distinct_varied_samples_and_masks():
    class Tokenizer:
        pad_token_id=0
        def encode(self,s,**kwargs):return list(s.encode())
    c=request(); c['task'].update(batch=128,input_kind='varied_text',input_tokens=32)
    ids,mask,meta=create(c['task'],dict(vocab_size=1024),Tokenizer())
    assert set(meta['lengths'])=={8,16,32}
    assert mask.dtype==torch.int64 and ids.dtype==torch.int64
    assert len({tuple(x) for x in ids.tolist()})>100


def test_configuration_does_not_accept_underfilled_training_or_unknown_engine():
    c=request(); c['task']['mode']='full'
    with pytest.raises(ValueError,match='task.batch'):resolve(c)
    c=request(); c['execution']['engine']='not-implemented'
    with pytest.raises(ValueError,match='execution.engine'):resolve(c)


def test_checkpoint_full_training_restores_all_slots_and_next_step(tiny,tmp_path):
    from tidegraph.lm.checkpoint import prepare_training, training_state
    c=request();c['task'].update(mode='full',batch=4,global_batch=4)
    ids=torch.arange(48).reshape(4,12)+10;mask=torch.ones_like(ids)
    original=deepcopy(tiny);opt=optimizer_for(tiny)
    update(tiny,opt,ids,mask,c,torch.device('cpu'),None)
    recorded=prepare_training(tiny,opt,ids,mask,c,tmp_path,torch.device('cpu'),None)
    saved=torch.load(tmp_path/'checkpoint.pt',weights_only=True)['state']
    with torch.no_grad():
        for name,p in original.named_parameters():p.copy_(saved['parameters'][name])
    restored=optimizer_for(original);restored.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng'])
    update(original,restored,ids,mask,c,torch.device('cpu'),None)
    assert identity(training_state(original,restored))==recorded['expected']


@pytest.mark.parametrize('architecture',['olmoe','qwen3_moe'])
def test_moe_online_cache_and_logits_to_keep(architecture):
    import transformers
    if architecture=='olmoe':
        config=transformers.OlmoeConfig(vocab_size=1024,hidden_size=32,intermediate_size=32,num_hidden_layers=2,
             num_attention_heads=4,num_key_value_heads=4,num_experts=4,num_experts_per_tok=2,max_position_embeddings=128)
        owner=transformers.OlmoeForCausalLM(config).eval()
    else:
        config=transformers.Qwen3MoeConfig(vocab_size=1024,hidden_size=32,intermediate_size=64,moe_intermediate_size=32,
             num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
             num_experts=4,num_experts_per_tok=2,max_position_embeddings=128)
        owner=transformers.Qwen3MoeForCausalLM(config).eval()
    c=request();ids=torch.arange(24).reshape(2,12)+10;mask=torch.ones_like(ids)
    actual=generate(owner,ids,mask,c,torch.device('cpu'),None,diagnostics=True)
    from tidegraph.lm.verification import independent_prefix
    expected=independent_prefix(owner,ids,mask,4)
    assert torch.equal(actual['tokens'],expected['tokens'])
    for a,b in zip(actual['logits'],expected['logits']):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def test_natural_eos_counts_only_live_rows(tiny):
    from types import SimpleNamespace
    class Owner:
        config=SimpleNamespace(eos_token_id=2)
        calls=0
        def __call__(self,**kwargs):
            self.calls+=1
            logits=torch.zeros(2,1,5)
            logits[0,0,2]=1
            logits[1,0,2 if self.calls>3 else 3]=1
            return SimpleNamespace(logits=logits,past_key_values=None)
    c=request();c['task'].update(input_kind='varied_text',output_tokens=6)
    c['execution']['prefill_chunk']=12
    actual=generate(Owner(),torch.ones(2,12,dtype=torch.int64),torch.ones(2,12,dtype=torch.int64),c,torch.device('cpu'),None)
    assert actual['metrics']['output_tokens']==5
    assert actual['metrics']['physical_output_rows']==8
