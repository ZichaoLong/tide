import torch
from tidegraph import External
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.reference import run
from attention_cases import fixture


def shared_case(dtype):
    g, m, q, xs, x, _ = fixture(dtype, cyclic=True)
    m.nodes[1].extra["attn_q"] = m.nodes[0].extra["attn_q"]
    m.edge_scale[1] = m.edge_scale[0]
    return g, m, q, xs, x


def train(dtype, native):
    g, m, q, xs, x = shared_case(dtype)
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.003)
    engine = Native(g, m, workers=3, packed=True, mode="hst") if native else None
    records = []
    for stop in (12, 16):
        optimizer.zero_grad(set_to_none=True)
        result = engine.run(q, xs, stop, sealed_until=stop) if engine else run(g, m, q, xs, stop, sealed_until=stop, mode="hst")
        objective(result).backward()
        grads = {k: None if p.grad is None else p.grad.clone() for k, p in m.named_parameters()}
        optimizer.step()
        q = result.continuation.detach()
        records.append((q, grads, {k: p.clone() for k, p in m.state_dict().items()}))
        xs = [External(b, p, q.ledger[b, p][0] + 1, stop + 1, x[b, p, 0]) for b in range(3) for p in range(2)]
    return records, g, m, q, optimizer


def test_attention_shared_training_checkpoint(dtype, tmp_path):
    expected, _, _, _, _ = train(dtype, False)
    actual, g, m, q, optimizer = train(dtype, True)
    equivalent(expected, actual)
    path = tmp_path / "trained-attention.pt"; save(path, g, m, q, optimizer)
    _, restored, _, _, _ = shared_case(dtype)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003)
    rq = load(path, g, restored, restored_optimizer)
    equivalent(q, rq); equivalent(m.state_dict(), restored.state_dict())
    equivalent(optimizer.state_dict(), restored_optimizer.state_dict())
    assert restored.nodes[0].extra["attn_q"] is restored.nodes[1].extra["attn_q"]
