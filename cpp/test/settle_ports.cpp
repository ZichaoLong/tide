// Core-5 Settle §2/§6/§9/§10: literal parallel-path coefficients, no Python.
#include "tide/settle.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>

namespace {
using namespace tide;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void close(const Tensor& a, const Tensor& b) {
  require(a.defined() == b.defined(), "ports gradient presence mismatch");
  if (a.defined()) require(at::allclose(a, b, a.scalar_type() == at::kDouble ? 1e-8 : 1e-5,
                                            a.scalar_type() == at::kDouble ? 1e-10 : 1e-6),
                           "ports literal formula mismatch");
}
}

void check_settle_ports(at::TensorOptions options) {
  using namespace tide;
  Graph graph;
  graph.nodes = {{0, false, true}, {1, false, true}, {2, false, true}};
  graph.regions = {{1}, {1}, {1}};
  graph.edges = {{0, 2, 2}, {0, 2, 2}, {1, 2, 2}};
  graph.inputs = {0, 1, 2, 2}; graph.outputs = {0, 2, 2}; graph.compile();
  SettleGraph spec(graph, {1, 1, 3}, "ports", 5);
  Model m;
  for (int n = 0; n < 3; ++n)
    m.nodes.push_back({at::zeros({2}, options), at::zeros({2,2}, options),
                       at::zeros({2}, options), at::zeros({2}, options)});
  for (double c : {1.,2.,3.,4.}) m.input_scale.push_back(at::full({}, c, options));
  for (double c : {5.,7.,11.}) m.agg_scale.push_back(at::full({}, c, options));
  for (double c : {2.,3.,4.}) m.edge_scale.push_back(at::full({}, c, options));
  for (double c : {13.,17.,19.}) m.output_scale.push_back(at::full({}, c, options));
  auto x = at::full({2}, .2, options).set_requires_grad(true);
  auto y = at::full({2}, .3, options).set_requires_grad(true);
  auto z = at::full({2}, .1, options).set_requires_grad(true);
  auto unused = at::full({2}, .4, options).set_requires_grad(true);
  std::vector<External> inputs;
  for (const auto& [port, value] : std::vector<std::pair<Index,Tensor>>{{0,x},{1,y},{2,z}}) {
    auto rows = spec.port_external(value.unsqueeze(0), {0}, port);
    inputs.insert(inputs.end(), rows.begin(), rows.end());
  }
  auto later = spec.port_external(x.unsqueeze(0), {2}, 0, 0, 1);
  inputs.insert(inputs.end(), later.begin(), later.end());
  Continuation q; q.identity = graph.identity;
  require(spec.encoded_graph().identity == graph.identity, "general embedding changed graph");
  for (const std::string algorithm : {"streaming", "frontier", "greedy"}) {
    Options opts; opts.packed = true;
    SettleExecutor engine(spec, m, opts, algorithm);
    auto out = spec.project(engine.run_ports(q, inputs, 15, 15));
    require(out.outputs.size() == 6 && out.messages.size() == 5, "ports record counts");
    auto h = 31*x + 88*y + 3*z;
    for (const auto& row : out.outputs) {
      auto expected = row.port == 0 ? 13*x : (row.time < 5 ? h : 31*x) * (row.port == 1 ? 17 : 19);
      close(row.value, expected);
    }
    const auto gradient = torch::autograd::grad({out.outputs.at(1).value.sum()}, {x,y,z,unused}, {}, true, false, true);
    close(gradient[0], at::full_like(x, 17*31)); close(gradient[1], at::full_like(y, 17*88));
    close(gradient[2], at::full_like(z, 17*3)); require(!gradient[3].defined(), "unused root connected");
    require(out.continuation.ledger.at({0,0}) == Owner{1,11}, "occurrence confused with global position");
    for (Index cut = 0; cut <= 15; ++cut) {
      std::vector<External> left, right;
      for (const auto& row : inputs) (row.time < cut ? left : right).push_back(row);
      auto first = spec.project(engine.run_ports(q, left, cut, cut));
      if (cut == 2) require(first.continuation.pending.size() == 3, "partial cut lost pending");
      auto last = engine.run_ports(first.continuation, right, 15, 15);
      for (const auto& [owner, state] : out.continuation.states)
        close(last.continuation.states.at(owner).value, state.value);
      require(last.continuation.ledger == out.continuation.ledger, "resumed port ledger mismatch");
    }
    bool rejected = false;
    auto bad = inputs; bad[0].time = 2;
    try { engine.run_ports(q, bad, 15, 15); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "accepted misaligned ports");
  }
}
