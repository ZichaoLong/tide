#include "portable_torch/runtime.hpp"
#include "tide/settle.h"
#include "tide/parameters.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
void close(const Tensor& a, const Tensor& b) {
  require(a.defined() == b.defined(), "Settle VJP connectivity mismatch");
  if (!a.defined()) return;
  const bool fp64 = a.scalar_type() == at::kDouble;
  require(a.sizes() == b.sizes() && at::allclose(a, b, fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6),
          "Settle standalone formula/value/VJP mismatch");
}
template <class F> void rejects(F operation) {
  bool rejected = false;
  try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "Settle accepted an invalid contract");
}

void check(at::TensorOptions options) {
  Graph graph;
  graph.nodes = {{0}, {1}}; graph.regions = {{1}, {1}};
  graph.edges = {{0, 1, 2}}; graph.inputs = {0}; graph.outputs = {0, 1};
  graph.compile();
  SettleGraph spec(graph, {1, 3});
  require(spec.stride() == 5 && spec.encoded_graph().nodes.size() == 4, "Settle rank encoding mismatch");
  auto trainable = [](Tensor value) { return value.set_requires_grad(true); };
  NodeWeights node{trainable(at::zeros({2}, options)), trainable(at::eye(2, options) * .2),
                   trainable(at::full({2}, .03, options)), trainable(at::ones({2}, options))};
  auto input_scale = trainable(at::full({}, .7, options));
  auto edge_scale = trainable(at::full({}, .8, options));
  Model model;
  model.nodes = {node, node};
  model.input_scale = {input_scale}; model.agg_scale = {at::full({}, .6, options)};
  model.edge_scale = {edge_scale}; model.output_scale = {input_scale, at::full({}, .9, options)};
  auto embedded = spec.embed_model(model);
  require(embedded.nodes[0].weight.is_same(embedded.nodes[1].weight) &&
          embedded.agg_scale[1].is_same(input_scale) && embedded.agg_scale[2].is_same(input_scale),
          "Settle encoding changed parameter owners");
  require(model.parameters().owners().size() == embedded.parameters().owners().size(),
          "Settle adapters added a trainable owner");
  auto x = trainable(at::arange(12, options).reshape({2, 3, 2}) / 30 - .1);
  auto initial = trainable(at::full({2, 2, 2}, .12, options));
  Continuation q; q.identity = graph.identity; q.batch_size = 2;
  for (Index b = 0; b < 2; ++b) for (Index v = 0; v < 2; ++v) q.states[{b, v}] = {initial[b][v]};
  auto eq = spec.embed_initial(q);

  // Literal two-layer recurrence: no executor, planner or local kernel calls.
  auto state0 = initial.select(1, 0), state1 = initial.select(1, 1);
  Tensor reference_loss = at::zeros({}, options);
  std::vector<Tensor> reference_outputs;
  for (Index t = 0; t < 3; ++t) {
    const auto content0 = x.select(1, t) * input_scale;
    state0 = at::sigmoid(node.decay) * state0 + content0;
    const auto full0 = content0 + at::tanh(at::matmul(state0, node.weight) + node.bias);
    const auto content1 = full0 * edge_scale * .6;
    state1 = at::sigmoid(node.decay) * state1 + content1;
    const auto full1 = content1 + at::tanh(at::matmul(state1, node.weight) + node.bias);
    auto output = full0 * input_scale + full1 * .9;
    reference_outputs.push_back(output);
    reference_loss = reference_loss + output.square().sum();
  }
  reference_loss = reference_loss + .07 * (state0.square().sum() + state1.square().sum());
  std::vector<Tensor> variables{x, initial, node.decay, node.weight, node.bias, node.read, input_scale, edge_scale};
  auto reference_vjp = torch::autograd::grad({reference_loss}, variables, {}, true, false, true);
  for (const std::string algorithm : {"frontier", "streaming"}) for (bool packed : {false, true}) {
    Options execution; execution.workers = packed ? 3 : 1; execution.packed = packed;
    if (packed) {
      execution.full_autograd = "batched"; execution.aggregate_autograd = "batched";
      execution.packed_sources = true; execution.batch_next = true;
      execution.parallel_regions = true; execution.compact_events = true;
      execution.defer_state_release = true;
    }
    SettleExecutor executor(spec, model, execution, algorithm);
    auto encoded = executor.run(eq, x);
    auto result = spec.project(encoded);
    require(result.outputs.size() == 6 && result.continuation.pending.empty(), "Settle output count/cut mismatch");
    Tensor loss = at::zeros({}, options);
    for (const auto& output : result.outputs) {
      close(output.value, reference_outputs.at(output.time / spec.stride())[output.batch]);
      loss = loss + output.value.square().sum();
    }
    for (Index b = 0; b < 2; ++b) {
      close(result.continuation.states.at({b, 0}).value, state0[b]);
      close(result.continuation.states.at({b, 1}).value, state1[b]);
    }
    for (const auto& item : result.continuation.states) loss = loss + .07 * item.second.value.square().sum();
    auto actual_vjp = torch::autograd::grad({loss}, variables, {}, true, false, true);
    for (size_t i = 0; i < variables.size(); ++i) close(actual_vjp[i], reference_vjp[i]);
    require(!actual_vjp[5].defined(), "HARD unused Read unexpectedly connected");
    if (algorithm == "frontier") require(encoded.stats.at("max_state_sequence") == 3, "Settle did not execute time prefill");
    auto first = executor.run(eq, x.slice(1, 0, 1));
    SettleExecutor stream(spec, model, execution, "streaming");
    auto last = stream.run(first.continuation, x.slice(1, 1));
    auto projected = spec.project(last);
    for (const auto& [owner, state] : result.continuation.states) close(state.value, projected.continuation.states.at(owner).value);
    require(projected.continuation.ledger == result.continuation.ledger, "Settle chunk ledger mismatch");
    auto empty = executor.run(last.continuation, x.slice(1, 3));
    require(empty.outputs.empty() && empty.continuation.cut == last.continuation.cut, "empty Settle window changed time");
  }
  rejects([&] { SettleGraph bad(graph, {1, 1}); });
  rejects([&] { SettleGraph bad(graph, {1, 2}); });
  rejects([&] { SettleGraph bad(graph, {1, std::numeric_limits<Index>::max()}); });
  rejects([&] { spec.external(x, std::numeric_limits<Index>::max()); });
  rejects([&] { SettleExecutor bad(spec, model, {}, "invalid"); });
  rejects([&] { auto bad = q; bad.cut = 1; spec.embed_initial(bad); });
}
}  // namespace

int main(int argc, char** argv) {
  try {
    auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kFloat && args.dtype != at::kDouble))
      throw std::invalid_argument("Settle check requires CPU FP32/FP64");
    if (!args.output_dir.empty() && std::filesystem::exists(args.output_dir))
      throw std::invalid_argument("output exists");
    at::set_num_threads(1); at::set_num_interop_threads(1);
    check(at::TensorOptions().device(device).dtype(args.dtype));
    const std::string report = "standalone-settle: passed; native encoding, formula, VJP, aliases, prefill, chunk, rejection\n";
    if (!args.output_dir.empty()) {
      if (!std::filesystem::create_directories(args.output_dir)) throw std::runtime_error("failed to create output");
      std::ofstream file(std::filesystem::path(args.output_dir) / "result.txt"); file << report;
      if (!file) throw std::runtime_error("failed to write result");
    }
    std::cout << report;
    return 0;
  } catch (const c10::Error& error) { std::cerr << error.what_without_backtrace() << '\n'; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; }
  return 2;
}
