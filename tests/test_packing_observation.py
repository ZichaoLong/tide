"""Hand-counted buckets and inert instrumentation, independent of any GPU."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tidegraph.weighted import packing_observation as observation


def test_scoped_buckets_reset_and_bound():
    assert not observation.enabled()
    observation.bucket("ignored", 9, 9)
    with observation.observe() as outer:
        observation.bucket("cache", 2, 3)
        with observation.observe() as inner:
            observation.projection("matrix", 4)
        observation.bucket("cache", 2, 3)
        observation.bucket("cache", 0, 1)
        observation.skipped("budget", 2)
    metrics = outer.metrics()
    assert metrics["packing/bucket/cache/calls"] == 3
    assert metrics["packing/bucket/cache/rows"] == 7
    assert metrics["packing/bucket/cache/elements"] == 12
    assert metrics["packing/bucket/cache/length/2/rows/3/calls"] == 2
    assert metrics["packing/skipped/budget/rows"] == 2
    assert not any("matrix" in key for key in metrics)
    assert inner.metrics()["packing/projection/matrix/rows"] == 4
    assert not observation.enabled()
    with observation.observe() as bounded:
        for length in range(observation.MAX_KEYS+4):
            observation.bucket("cache", length, 1)
    assert len(bounded.counts) == observation.MAX_KEYS
    assert bounded.metrics()["packing/dropped_updates"] > 0


def test_observation_unwinds_on_failure_and_rejects_tensor_metadata():
    with pytest.raises(RuntimeError):
        with observation.observe():
            raise RuntimeError("test")
    assert not observation.enabled()
    with observation.observe():
        with pytest.raises(ValueError, match="metadata"):
            observation.bucket("cache", torch.tensor(2), 1)


class DispatchCounts(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def test_ragged_kv_hand_counts_without_extra_tensor_operations(dtype):
    from test_weighted_packed import binding
    from tidegraph.content import Content
    from tidegraph.packing import PackedSequence
    from tidegraph.weighted import query_batch, full_batch
    from tidegraph.compare import equivalent
    bound = binding(dtype, "kv", memory_read="qry")
    weights = bound.model.nodes[0]
    memory = weights.kernel
    raw = torch.tensor([[.4, .2, -.1], [.7, .3, .4], [0., .8, .9], [.3, -.2, .7]], dtype=dtype)
    batch = PackedSequence(raw, [0, 2, 3, 4], [(0, 0), (1, 0), (2, 0)], [0, 2, 1, 3], [Content(v) for v in raw])
    initial = [memory.initial(weights) for _ in range(3)]
    def run():
        states, _ = memory.packed_sequence(weights, initial, batch)
        final = [states[i] for i in (1, 2, 3)]
        u = raw[[1, 2, 3], 1:]
        return states, query_batch.kv(memory, final, [4, 4, 4], u), full_batch.kv(weights.full_program.content, memory, final, [4, 4, 4], u)
    with torch.no_grad(), DispatchCounts() as baseline_ops:
        expected = run()
    with torch.no_grad(), observation.observe() as counted, DispatchCounts() as observed_ops:
        actual = run()
    equivalent(actual, expected, atol=0., rtol=0.)
    assert baseline_ops.counts == observed_ops.counts  # Includes scalar/copy/nonzero operations.
    metrics = counted.metrics()
    assert metrics["packing/bucket/positive_write_segments/length/2/rows/1/calls"] == 1
    assert metrics["packing/bucket/positive_write_segments/length/1/rows/1/calls"] == 1
    assert metrics["packing/skipped/zero_write_owners/rows"] == 1
    assert metrics["packing/projection/kv_write_unprepared_rows/rows/3/calls"] == 1
    for phase in ("kv_visible_read", "kv_visible_full"):
        for length in (0, 1, 2):
            assert metrics[f"packing/bucket/{phase}/length/{length}/rows/1/calls"] == 1
    assert metrics["packing/bucket/attention_positive/calls"] == 9


def test_independent_projection_budget_and_observer_preserve_roots(dtype):
    from test_weighted_independent_preparation import memory
    from tidegraph.weighted.kv_independent_batch import prepare
    memories = [memory(dtype) for _ in range(3)]
    inputs = [torch.tensor([.4, value], dtype=dtype, requires_grad=True) for value in (2., -3., 4.)]
    per_row = 4*sum(t.numel()*t.element_size() for t in
                    (memories[0].base, memories[0].adapter_v, memories[0].adapter_u, memories[0].summary))
    def run(enabled, budget):
        with observation.observe() if enabled else nullcontext() as counted:
            rows = prepare(memories, inputs, {}, budget=budget)
            grads = torch.autograd.grad(rows[0]["values"].sum(), inputs, allow_unused=True) if rows[0] else None
        return rows, grads, counted
    expected, gradients, _ = run(False, per_row*2)
    actual, observed_gradients, counted = run(True, per_row*2)
    assert torch.equal(actual[0]["values"], expected[0]["values"])
    assert torch.equal(observed_gradients[0], gradients[0])
    assert observed_gradients[1:] == gradients[1:] == (None, None)
    assert counted.metrics()["packing/projection/private_bmm/rows/2/calls"] == 4
    assert counted.metrics()["packing/skipped/independent_projection_singleton/rows"] == 1
    _, _, rejected = run(True, per_row)
    assert rejected.metrics()["packing/skipped/independent_projection_budget/rows"] == 3


def test_cpp_counter_contract_and_parallel_collection(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++17 compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path/"counter.cpp"
    source.write_text('''#include "tide/packing_work.h"
#include <iostream>
#include <thread>
int main(){using namespace tide::work::packing;
  bucket("ignored",9,9);reset(true);
  std::thread a([]{for(int i=0;i<5;++i)bucket("cache",2,3);});
  std::thread b([]{for(int i=0;i<5;++i)bucket("cache",2,3);});
  a.join();b.join();projection("matrix",0);skipped("budget",2);
  for(const auto& [key,value]:metrics())std::cout<<key<<"="<<value<<"\\n";
  reset(false);bucket("ignored",9,9);
  return metrics().size()==2?0:1;
}
''')
    binary = tmp_path/"counter"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(root/"cpp/include"), str(source), "-o", str(binary)], check=True, timeout=30)
    output = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10).stdout
    actual = {key: float(value) for key, value in (line.split("=") for line in output.splitlines())}
    with observation.observe() as counted:
        for _ in range(10):
            observation.bucket("cache", 2, 3)
        observation.projection("matrix", 0)
        observation.skipped("budget", 2)
    assert actual == counted.metrics()
