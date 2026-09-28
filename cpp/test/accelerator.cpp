// Standalone LibTorch contract: no Python imports, callbacks or serialization.
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/specialized.h"
#include "tide/checkpoint.h"
#include "../bench/streaming.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <fstream>
#include <iostream>

namespace {
using namespace tide;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(const Tensor& a, const Tensor& b) {
  require(a.defined() == b.defined(), "gradient connectivity mismatch");
  if (!a.defined()) return;
  const auto x = a.detach().to(at::kCPU), y = b.detach().to(at::kCPU);
  require(x.sizes() == y.sizes() && x.scalar_type() == y.scalar_type(), "tensor metadata mismatch");
  require(at::isfinite(x).all().item<bool>() && at::isfinite(y).all().item<bool>(), "nonfinite value");
  require(at::allclose(x, y, x.scalar_type() == at::kDouble ? 1e-8 : 1e-5,
                      x.scalar_type() == at::kDouble ? 1e-10 : 1e-6), "tensor value mismatch");
}
Model model_for(const Graph& graph, at::Device device, at::ScalarType dtype) {
  auto host = at::TensorOptions().dtype(dtype);
  auto leaf = [&](const Tensor& value) { return value.to(device).clone().set_requires_grad(true); };
  Model model;
  for (size_t n = 0; n < graph.nodes.size(); ++n) {
    NodeWeights w;
    w.bias = leaf(at::arange(4, host)*.015 + .01*n);
    w.decay = leaf(at::arange(4, host)*.02);
    w.read = leaf(at::arange(4, host)*.03 + .01);
    w.weight = leaf(at::eye(4, host)*.2 + .01*n);
    model.nodes.push_back(w);
  }
  model.nodes[2].weight = model.nodes[1].weight;
  for (size_t i = 0; i < graph.inputs.size(); ++i) model.input_scale.push_back(leaf(at::full({}, .25, host)));
  for (size_t i = 0; i < graph.outputs.size(); ++i) model.output_scale.push_back(leaf(at::full({}, .25, host)));
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    model.agg_scale.push_back(leaf(at::full({}, .25, host)));
    model.edge_scale.push_back(leaf(at::full({}, .25, host)));
  }
  return model;
}
void detach(Continuation& q) {
  for (auto& [owner, state] : q.states) {
    state.value = state.value.detach();
    for (auto& [name, value] : state.slots) value = value.detach();
  }
  for (auto& [owner, h] : q.history) for (auto& [name, value] : h.tensors) value = value.detach();
  for (auto& a : q.pending) a.value = a.value.detach();
}
Tensor loss_of(const Result& result) {
  std::vector<Tensor> terms;
  for (const auto& value : result.outputs) terms.push_back(value.value.square().sum());
  for (const auto& [owner, state] : result.continuation.states) terms.push_back(state.value.square().sum());
  for (const auto& value : result.continuation.pending) terms.push_back(value.value.square().sum());
  require(!terms.empty(), "empty graph loss");
  return at::stack(terms).sum()/terms.size();
}
void compare_owners(const ParameterRegistry& a, const ParameterRegistry& b, bool gradients) {
  require(a.alias_partitions() == b.alias_partitions(), "parameter alias mismatch");
  for (const auto& owner : a.owners())
    close(gradients ? owner.value.grad() : owner.value,
          gradients ? b.value(owner.canonical).grad() : b.value(owner.canonical));
}
void check_graph(const std::string& topology, at::Device device, at::ScalarType dtype,
                 const std::filesystem::path& directory) {
  Graph graph;
  graph.nodes = {{0},{1},{2},{3}};
  graph.regions = {{1},{1},{1},{1}};
  graph.edges = topology == "ring" ? std::vector<Edge>{{0,1,1},{1,2,1},{2,3,1},{3,0,1}}
                                    : std::vector<Edge>{{0,1,1},{0,2,1},{1,3,1},{2,3,1}};
  graph.inputs = {0}; graph.outputs = {3}; graph.compile();
  auto cpu = model_for(graph, at::Device(at::kCPU), dtype), target = model_for(graph, device, dtype);
  auto cpu_parameters = cpu.parameters(), target_parameters = target.parameters();
  OptimizerGroup group; group.lr = .0002; group.eps = 1e-5; group.weight_decay = .01;
  group.amsgrad = true;
  for (const auto& owner : cpu_parameters.owners()) group.parameters.push_back(owner.canonical);
  AdamW cpu_optimizer(cpu_parameters, {group}), target_optimizer(target_parameters, {group});
  Options reference_options; reference_options.mode = "hst";
  Options options = reference_options; options.packed = true; options.workers = 2;
  options.parallel_regions = true; options.compact_events = true;
  options.packed_sources = true; options.batch_next = true;
  options.full_autograd = "batched"; options.aggregate_autograd = "batched";
  Streaming reference(graph, cpu, reference_options);
  Specialized candidate(graph, target, options, topology);
  Continuation a, b; a.identity = b.identity = graph.identity;
  for (Index cycle = 0; cycle < 3; ++cycle) {
    cpu_optimizer.zero_grad(); target_optimizer.zero_grad();
    auto x = (at::arange(4, at::TensorOptions().dtype(dtype))*.05+.1).set_requires_grad(true);
    auto y = x.detach().to(device).clone().set_requires_grad(true);
    auto expected = reference.run(a, {{0,0,cycle,cycle*6,x}}, (cycle+1)*6, (cycle+1)*6);
    auto actual = candidate.run(b, {{0,0,cycle,cycle*6,y}}, (cycle+1)*6, (cycle+1)*6);
    require(actual.continuation.identity == expected.continuation.identity, "graph identity mismatch");
    portable_torch::synchronize(device);
    tide_bench::compare(actual, expected, true, dtype, device);
    auto expected_loss = loss_of(expected), actual_loss = loss_of(actual);
    close(actual_loss, expected_loss);
    expected_loss.backward(); actual_loss.backward();
    close(y.grad(), x.grad()); compare_owners(target_parameters, cpu_parameters, true);
    a = expected.continuation; b = actual.continuation; detach(a); detach(b);
    cpu_optimizer.step(); target_optimizer.step();
    compare_owners(target_parameters, cpu_parameters, false);
  }
  const auto checkpoint = directory/(topology+".tide");
  Checkpoint::save(checkpoint, target_parameters, &target_optimizer, graph.identity);
  auto restored = model_for(graph, device, dtype), restored_cpu = model_for(graph, at::Device(at::kCPU), dtype);
  auto restored_parameters = restored.parameters(), restored_cpu_parameters = restored_cpu.parameters();
  AdamW restored_optimizer(restored_parameters, {group}), restored_cpu_optimizer(restored_cpu_parameters, {group});
  Checkpoint::load(checkpoint, restored_parameters, &restored_optimizer, graph.identity);
  Checkpoint::load(checkpoint, restored_cpu_parameters, &restored_cpu_optimizer, graph.identity);
  compare_owners(restored_parameters, target_parameters, false);
  for (const auto& [name, state] : restored_optimizer.state()) {
    require(state.exp_avg.device() == device && state.exp_avg_sq.device() == device
            && state.max_exp_avg_sq.device() == device, "optimizer checkpoint stayed on host");
    close(state.exp_avg, target_optimizer.state().at(name).exp_avg);
  }
  for (auto* registry : {&target_parameters, &restored_parameters, &restored_cpu_parameters})
    for (const auto& owner : registry->owners()) owner.value.mutable_grad() = at::ones_like(owner.value)*.01;
  target_optimizer.step(); restored_optimizer.step(); restored_cpu_optimizer.step();
  compare_owners(restored_parameters, target_parameters, false);
  compare_owners(restored_cpu_parameters, target_parameters, false);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv, true);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (args.dtype != at::kFloat && args.dtype != at::kDouble) throw std::invalid_argument("FP32/FP64 required");
    if (args.output_dir.empty() || std::filesystem::exists(args.output_dir))
      throw std::invalid_argument("a new output directory is required");
    std::filesystem::create_directories(args.output_dir);
    at::set_num_threads(1); at::set_num_interop_threads(1);
    portable_torch::seed_runtime(device, args.seed);
    std::optional<c10::Stream> stream;
    if (!device.is_cpu()) {
      c10::impl::VirtualGuardImpl implementation(device.type());
      stream = implementation.getNewStream(device);
    }
    c10::OptionalStreamGuard stream_guard(stream);
    if (stream) {
      tide::NodePool pool(2); pool.set_device(device);
      auto check = [&] {
        c10::impl::VirtualGuardImpl implementation(device.type());
        require(implementation.getStream(device) == *stream, "worker did not inherit caller stream");
      };
      pool.run({check, check});
    }
    for (const auto& topology : {"ring", "diamond"}) check_graph(topology, device, args.dtype, args.output_dir);
    portable_torch::synchronize(device);
    std::ofstream report(std::filesystem::path(args.output_dir)/"result.json");
    report << "{\"state\":\"passed\",\"device\":\"" << device.str()
           << "\",\"dtype\":\"" << portable_torch::dtype_name(args.dtype)
           << "\",\"cases\":2,\"training_steps\":3,\"workers\":2,\"checkpoint_handoff\":true,\"nondefault_stream\":"
           << (stream ? "true" : "false") << "}\n";
    if (!report) throw std::runtime_error("could not write result");
    std::cout << "standalone accelerator parity, gradients, optimizer and checkpoint: passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
