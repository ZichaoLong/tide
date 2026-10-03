"""Capacity policy for actual FP32 KV journals, independent of payload dtype."""
from dataclasses import replace
import pytest
import torch
from test_consumer_capacity import geometry, cpp_plan, capacity_probe
from tools.online_bench.capacity import Capacities, Chunks, plan


@pytest.mark.parametrize('payload', [2, 4])
@pytest.mark.parametrize('windows', [1, 2, 4])
def test_journal_storage_inventory_and_window_lifetime(payload, windows, capacity_probe):
    _, g = geometry(width=128)
    g = replace(g, payload=payload, windows=windows)
    caps, chunks = Capacities(kv_trace=3073), Chunks(1, 1, 1, 1, 1, 1, 1)
    # DeviceJournal's actual tensor types/shapes, including a non-aligned row
    # count; no model/payload-dtype guess or summing API workspace ceilings.
    tensors = [torch.empty((caps.kv_trace, 5), dtype=torch.int64),
               torch.empty((caps.kv_trace, 2*g.width+1), dtype=torch.float32),
               torch.empty((1,), dtype=torch.int64)]
    bank = sum(x.nbytes for x in tensors)
    current = plan(g, caps, chunks, [64*1024**3]*g.devices, True)
    assert cpp_plan(capacity_probe, g, caps, chunks, [64*1024**3]*g.devices, True) == current
    old = plan(g, caps, chunks, [64*1024**3]*g.devices, False)
    for before, after in zip(old['devices'], current['devices']):
        c = after['components']
        assert bank <= c['kv_journal_bank_bytes'] <= bank+4096
        assert c['journals'] >= 2*bank
        assert windows*bank <= c['retained_kv_journal_bytes'] <= windows*(bank+4096)
        assert c['retained_pack_workspace'] == before['components']['retained_pack_workspace'] > 0
        assert c['retained'] >= c['retained_kv_journal_bytes']
        assert c['journals'] < before['components']['journals']
        assert after['usable_bytes'] == after['budget_bytes']-after['budget_bytes']//10-128*1024**2


@pytest.mark.parametrize('variant', ['single', 'conservative', 'inference', 'add'])
def test_undeclared_modes_keep_the_legacy_journal_bound(variant, capacity_probe):
    _, g = geometry(memory='add' if variant == 'add' else 'attention', width=16,
                    devices=1 if variant == 'single' else 3, training=variant != 'inference')
    caps, chunks = Capacities(kv_trace=2049), Chunks(1, 1, 1, 1, 1, 1, 1)
    aggressive = variant != 'conservative'
    result = plan(g, caps, chunks, [64*1024**3]*g.devices, aggressive)
    assert cpp_plan(capacity_probe, g, caps, chunks, [64*1024**3]*g.devices, aggressive) == result
    for card in result['devices']:
        c = card['components']
        assert c['kv_journal_bank_bytes'] == 0
        assert c['journals'] == (24*caps.kv_trace*(2*g.width+8) if g.attention else 0)

