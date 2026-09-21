#include "portable_torch/runtime.hpp"
#include "tide/kernel.h"
#include "tide/stream.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
class CustomAccumulator final : public tide::StateKernel {
 public:
  tide::State initial(const tide::NodeWeights& w) const override {
    return {at::zeros_like(w.bias), -1, 0, {{"sum", at::zeros_like(w.bias)}}};
  }
  tide::State step(const tide::NodeWeights&, const tide::State& old, const tide::Tensor& h,
                   tide::Index time, const std::vector<tide::Atom>& fiber) const override {
    if (fiber.empty()) throw std::invalid_argument("custom kernel requires a real fiber");
    return {old.value + 2 * h, time, old.observations + 1, {{"sum", old.slots.at("sum") + 3 * h}}};
  }
  void validate_weights(const tide::NodeWeights&) const override {}
  void validate_state(const tide::NodeWeights& w, const tide::State& s) const override {
    if (s.slots.size() != 1 || !s.slots.count("sum") || s.slots.at("sum").sizes() != w.bias.sizes())
      throw std::invalid_argument("invalid custom state");
  }
};
}
int main(int argc, char** argv) {
  try {
    auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    auto device = portable_torch::resolve_device(args);
    if (args.dtype != at::kFloat && args.dtype != at::kDouble) throw std::invalid_argument("FP32/FP64 required");
    if (!args.output_dir.empty() && std::filesystem::exists(args.output_dir)) throw std::invalid_argument("output exists");
    at::set_num_threads(1);
    auto options = at::TensorOptions().device(device).dtype(args.dtype);
    tide::Graph g; g.nodes = {{0, false, false, "custom_accumulator", "tanh"}};
    g.edges = {{0, 0, 1}}; g.regions = {{1}}; g.inputs = {0}; g.outputs = {0}; g.compile();
    tide::Model m;
    m.nodes = {{at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::ones({2}, options)}};
    m.nodes[0].kernel = std::make_shared<CustomAccumulator>();
    m.input_scale = m.agg_scale = m.edge_scale = m.output_scale = {at::ones({}, options)};
    auto x = at::ones({2, 2}, options) * 0.25; x.set_requires_grad(true);
    tide::Continuation q; q.identity = g.identity; q.batch_size = 2;
    tide::Options runtime; runtime.packed = true; runtime.workers = 2;
    tide::Streaming engine(g, m, runtime);
    auto result = engine.run(q, {{0, 0, 0, 0, x[0]}, {1, 0, 0, 0, x[1]}}, 2, 2);
    auto loss = at::zeros({}, options);
    for (const auto& [owner, state] : result.continuation.states) loss = loss + state.value.sum() + state.slots.at("sum").sum();
    auto grad = torch::autograd::grad({loss}, {x}).at(0);
    if (!at::equal(grad, at::full_like(x, 10))) throw std::runtime_error("custom kernel VJP mismatch");
    auto idle = engine.run(result.continuation, {}, 2, 2);
    if (idle.continuation.states.size() != 2) throw std::runtime_error("custom kernel continuation mismatch");
    const std::string report = "custom-state-kernel: passed\n";
    if (!args.output_dir.empty()) {
      if (!std::filesystem::create_directories(args.output_dir)) throw std::runtime_error("failed to create output");
      std::ofstream file(std::filesystem::path(args.output_dir) / "result.txt"); file << report;
      if (!file) throw std::runtime_error("failed to write result");
    }
    std::cout << report;
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
