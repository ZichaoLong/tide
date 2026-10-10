"""Packed retained journals preserve independent CPU VJPs and update lifetimes."""
import json
import pytest
import torch
from resident_test_target import owner_devices
from tidegraph import ResidentPlacement
from resident_training_cases import runtime
from test_resident_training import target, training_case


@pytest.mark.parametrize("family,cards,schedule,full,aggregation,mode", [
    ("pdg",1,"streaming","tanh","weighted_mean","hst"),
    ("pdg",2,"greedy","swiglu","all_softmax","softp"),
    ("timed-dag",1,"greedy","lh-silu-rms-v1","sum","hard"),
    ("timed-dag",2,"streaming","tanh","active_softmax","hst"),
    ("settle",1,"streaming","tanh","sum","softp"),
    ("settle",2,"greedy","lh-silu-rms-v1","all_softmax","hard"),
])
def test_compact_journal_training(target,family,cards,schedule,full,aggregation,mode,tmp_path):
    start=torch.device(target).index
    placement=ResidentPlacement(devices=owner_devices(target, cards)) if cards>1 else None
    # Includes CPU autograd, aliases, None/zero roots, SGD/AdamW state and a
    # checkpoint suffix. Different windows have different actual tape extents.
    records=training_case(target,family,schedule,"adamw" if cards>1 else "sgd",tmp_path,
        full=full,aggregation=aggregation,emit_mode=mode,zeta=.75,placement=placement,
        model_device="cpu",emission="slot_affine" if mode=="hard" else "broadcast",
        chunk_policy="aggressive")
    assert all(r['retained_compact_journals']==1 for r in records)
    assert all(0 < r['retained_bytes'] < r['retained_dense_bytes'] for r in records)
    (tmp_path/'retention.json').write_text(json.dumps(records))


def test_empty_compact_journal_window(target):
    r=runtime('pdg',target,model_device='cpu',chunk_policy='aggressive')
    with torch.no_grad(),r.training_session(2) as session:
        windows=[session.advance_device([],stop=t,sealed_until=t) for t in (1,2)]
        gradients=session.backward([session.cotangents(w) for w in windows])
        assert gradients.statistics['retained_compact_journals']==1
        assert gradients.statistics['retained_bytes'] < gradients.statistics['retained_dense_bytes']
        assert not gradients.connected.any()
        assert session.step().applied
