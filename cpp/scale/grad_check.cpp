#include "scale.h"
#include "../bench/streaming.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>

namespace pdg_scale {
namespace {
struct Run { tide::Result result; std::vector<at::Tensor> leaves, roots; };
Run execute_grad(const Config& c, const Topology& topology, bool batched) {
  portable_torch::seed_runtime(at::Device(at::kCPU), c.runtime.seed);
  auto f = fixture(c, topology);
  tide::Options opts; opts.packed = batched; opts.workers = batched ? 3 : 1;
  opts.full_autograd = batched ? c.full_autograd : "replay";
  opts.aggregate_autograd = batched ? c.aggregate_autograd : "replay";
  opts.parallel_regions = batched; opts.packed_sources = batched && c.packed_sources;
  opts.batch_next = batched && c.batch_next;
  tide::Streaming engine(f.graph, f.model, opts);
  tide::Continuation q; q.identity = engine.graph().identity; q.batch_size = c.batch;
  tide::StreamingCursor cursor(engine, q);
  Run all; all.leaves = f.owners;
  for (Index token = 0; token < c.steps; ++token) {
    std::vector<tide::External> inputs;
    for (Index b = 0; b < c.batch; ++b) {
      auto x = at::zeros({c.width}, f.embedding.options()).set_requires_grad(true);
      all.leaves.push_back(x);
      inputs.push_back({b, 0, token, token*(topology.layers+1), f.embedding[(token*7+b*3)%c.vocab]+x});
    }
    auto part = cursor.advance(inputs, (token+1)*(topology.layers+1), (token+1)*(topology.layers+1));
    all.result.trace.insert(all.result.trace.end(), part.trace.begin(), part.trace.end());
    all.result.messages.insert(all.result.messages.end(), part.messages.begin(), part.messages.end());
    all.result.outputs.insert(all.result.outputs.end(), part.outputs.begin(), part.outputs.end());
  }
  all.result.continuation = cursor.snapshot();
  for (const auto& e : all.result.trace) if (e.batch == 0 && e.active && !e.emitted.empty()) {
    all.roots.push_back(e.full); all.roots.push_back(e.emitted.front().value); break;
  }
  for (const auto& x : all.result.outputs) if (x.batch == 0) { all.roots.push_back(x.value); break; }
  for (const auto& x : all.result.continuation.pending) if (x.batch == 0) { all.roots.push_back(x.value); break; }
  const auto& state = all.result.continuation.states.begin()->second;
  all.roots.push_back(state.value);
  for (const auto& [name, value] : state.slots) all.roots.push_back(value);
  return all;
}
}
void check_grad(const Config& c, const Topology& t) {
  if (c.width > 64 || c.batch > 8 || c.steps > 12)
    throw std::invalid_argument("gradient check requires small width/batch/steps");
  at::AutoGradMode grad(true);
  auto local = c; local.emission = "row";
  auto expected = execute_grad(local, t, false), actual = execute_grad(local, t, true);
  tide_bench::compare(actual.result, expected.result, true, c.runtime.dtype);
  if (expected.roots.size() != actual.roots.size() || expected.leaves.size() != actual.leaves.size())
    throw std::runtime_error("scale grad root/owner count mismatch");
  const auto atol = c.runtime.dtype == at::kDouble ? 1e-10 : 1e-6;
  const auto rtol = c.runtime.dtype == at::kDouble ? 1e-8 : 1e-5;
  for (size_t r = 0; r < expected.roots.size(); ++r) for (bool zero : {false, true}) {
    if (expected.roots[r].requires_grad() != actual.roots[r].requires_grad())
      throw std::runtime_error("scale grad root presence mismatch");
    if (!expected.roots[r].requires_grad()) continue;
    auto a = torch::autograd::grad({expected.roots[r].square().sum()*(zero ? 0. : .7)}, expected.leaves, {}, true, false, true);
    auto b = torch::autograd::grad({actual.roots[r].square().sum()*(zero ? 0. : .7)}, actual.leaves, {}, true, false, true);
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i].defined() != b[i].defined() || (a[i].defined() && !at::allclose(a[i], b[i], rtol, atol)))
        throw std::runtime_error("scale isolated gradient mismatch: root="+std::to_string(r)+" owner="+std::to_string(i));
  }
}
}  // namespace pdg_scale
