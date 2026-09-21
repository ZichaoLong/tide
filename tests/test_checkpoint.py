import pytest
import torch
from cases import ring
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.reference import run


def test_optimizer_and_value_checkpoint(dtype, tmp_path):
    results = []
    for native in (False, True):
        g, m, q, xs, _, _ = ring(dtype)
        opt = torch.optim.AdamW(m.parameters(), lr=0.001)
        result = Native(g, m, workers=3, packed=True, mode="hst").run(q, xs, 7, sealed_until=7) if native else run(g, m, q, xs, 7, sealed_until=7, mode="hst")
        objective(result).backward()
        opt.step()
        results.append({k: v.clone() for k, v in m.state_dict().items()})
        path = tmp_path / f"checkpoint-{native}.pt"
        save(path, g, m, result.continuation, opt)
        with pytest.raises(FileExistsError):
            save(path, g, m, result.continuation, opt)
        _, restored, _, _, _, _ = ring(dtype)
        restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.8)
        rq = load(path, g, restored, restored_opt)
        equivalent(result.continuation, rq)
        equivalent(m.state_dict(), restored.state_dict())
        equivalent(opt.state_dict(), restored_opt.state_dict())
        assert all(not s.value.requires_grad for s in rq.states.values())
        expected = run(g, m, result.continuation, [], 9, sealed_until=9)
        actual = Native(g, restored, workers=2, packed=True).run(rq, [], 9, sealed_until=9)
        equivalent(expected, actual)
    equivalent(*results)
