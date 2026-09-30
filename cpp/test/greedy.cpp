#include "portable_torch/runtime.hpp"
#include "tide/greedy.h"
#include "tide/stream.h"
#include "../bench/streaming.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tide;
void require(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}
Tensor objective(const Result& result, const Tensor& reference) {
  auto loss = at::zeros({}, reference.options());
  const auto cotangent = at::tensor({.125, -.375}, reference.options());
  for (const auto& output : result.outputs) loss = loss + (output.value * cotangent).sum();
  for (const auto& [owner, state] : result.continuation.states)
    loss = loss + (state.value * cotangent).sum() * .125;
  for (const auto& atom : result.continuation.pending)
    loss = loss + (atom.value * cotangent).sum() * .25;
  return loss;
}
void check(at::ScalarType dtype) {
  const auto options = at::TensorOptions().dtype(dtype).device(at::kCPU);
  Graph graph;
  graph.nodes = {{0}, {0}, {1}}; graph.regions = {{1}, {1}};
  graph.edges = {{0, 2, 2}, {0, 2, 2}, {1, 2, 1}, {2, 0, 3}, {2, 1, 4}};
  graph.inputs = {0, 1}; graph.outputs = {2}; graph.compile();
  auto trainable = [](Tensor x) { return x.set_requires_grad(true); };
  NodeWeights weights{trainable(at::zeros({2}, options)), trainable(at::eye(2, options)*.2),
                      trainable(at::full({2}, .03125, options)), trainable(at::ones({2}, options))};
  auto scale = trainable(at::full({}, .5, options));
  Model model; model.nodes = {weights, weights, weights};
  model.input_scale = {scale, scale}; model.output_scale = {scale};
  model.agg_scale.assign(graph.edges.size(), scale); model.edge_scale = model.agg_scale;
  auto x = trainable(at::arange(24, options).reshape({2, 3, 2, 2}) / 32);
  auto unused = trainable(at::ones({}, options));
  std::vector<Tensor> leaves{x, weights.decay, weights.weight, weights.bias, weights.read, scale, unused};
  std::vector<External> inputs;
  for (Index b=0; b<2; ++b) for (Index p=0; p<2; ++p) for (Index t=0; t<3; ++t)
    inputs.push_back({b, p, t, 2*t, x[b][t][p]});
  Continuation q; q.identity = graph.identity; q.batch_size = 2;
  Options scalar; scalar.mode = "hst";
  Streaming oracle(graph, model, scalar);
  const auto expected = oracle.run(q, inputs, 9, 9);
  for (bool packed : {false, true}) {
    auto execution = scalar; execution.packed = packed; execution.workers = packed ? 2 : 1;
    execution.compact_events = packed; execution.parallel_regions = packed;
    execution.packed_sources = packed; execution.batch_next = packed;
    if (packed) execution.full_autograd = execution.aggregate_autograd = "batched";
    Greedy candidate(graph, model, execution);
    auto actual = candidate.run(q, inputs, 9, 9);
    tide_bench::compare(actual, expected, true, dtype);
    require(actual.stats.at("greedy_stages") > 0, "greedy schedule did not run");
    require(q.cut == 0 && q.states.empty() && q.pending.empty(), "initial continuation mutated");
    for (double factor : {1., 0.}) {
      auto wanted = torch::autograd::grad({objective(expected, x)*factor}, leaves, {}, true, false, true);
      auto got = torch::autograd::grad({objective(actual, x)*factor}, leaves, {}, true, false, true);
      for (size_t i=0; i<leaves.size(); ++i) {
        require(got[i].defined() == wanted[i].defined(), "greedy VJP connectivity");
        if (got[i].defined()) require(at::allclose(got[i], wanted[i], dtype==at::kDouble ? 1e-8 : 1e-5,
                                                  dtype==at::kDouble ? 1e-10 : 1e-6), "greedy VJP values");
      }
      require(!got.back().defined(), "disconnected owner acquired a gradient");
    }
    std::vector<External> prefix, suffix;
    for (const auto& input : inputs) (input.time < 3 ? prefix : suffix).push_back(input);
    auto first = candidate.run(q, prefix, 3, 3);
    require(!first.continuation.pending.empty(), "feedback cut lost pending messages");
    auto resumed = oracle.run(first.continuation, suffix, 9, 9);
    Result boundary = expected; boundary.trace.clear(); boundary.messages.clear(); boundary.outputs.clear();
    resumed.trace.clear(); resumed.messages.clear(); resumed.outputs.clear();
    tide_bench::compare(resumed, boundary, false, dtype);
    execution.max_events = 1;
    bool refused = false;
    try { Greedy limited(graph, model, execution); limited.run(q, inputs, 9, 9); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused, "greedy ignored live-fiber capacity");
  }
}
}  // namespace
int main(int argc, char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kFloat && args.dtype != at::kDouble))
      throw std::invalid_argument("greedy independent CPU check requires CPU FP32/FP64");
    at::set_num_threads(1); at::set_num_interop_threads(1);
    check(args.dtype);
    std::cout << "standalone-greedy: passed; feedback, parallel edges, observables, VJPs, continuation, capacity\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
