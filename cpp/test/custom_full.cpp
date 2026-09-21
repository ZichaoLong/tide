#include "portable_torch/runtime.hpp"
#include "tide/full.h"
#include "tide/kernel.h"
#include "tide/stream.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
class SlotState final : public tide::StateKernel {
 public:
  tide::State initial(const tide::NodeWeights& w) const override {
    return {at::zeros_like(w.bias), -1, 0, {{"memory", at::zeros_like(w.bias)}}};
  }
  tide::State step(const tide::NodeWeights&, const tide::State& old, const tide::Tensor& h, tide::Index time,
                   const std::vector<tide::Atom>&) const override {
    return {old.value + h, time, old.observations + 1, {{"memory", old.slots.at("memory") + 2*h}}};
  }
  void validate_weights(const tide::NodeWeights&) const override {}
  void validate_state(const tide::NodeWeights& w, const tide::State& state) const override {
    if (state.slots.size() != 1 || !state.slots.count("memory") || state.slots.at("memory").sizes() != w.bias.sizes())
      throw std::invalid_argument("custom state slots mismatch");
  }
};
class ClockFull final : public tide::FullKernel {
 public:
  tide::FullResult step(const tide::NodeWeights&, const tide::FullInput& input, tide::Index slots,
                        const tide::Options&) const override {
    if (slots != 2 || input.comparison->last_time != input.time)
      throw std::invalid_argument("custom Full state/slot mismatch");
    auto value = input.comparison->value + input.comparison->slots.at("memory") + (input.time + input.comparison->observations);
    tide::FullResult result;  // No auxiliary Full value: only the tagged family.
    if (input.time % 2 == 0) result.emitted.push_back({0, value});
    result.emitted.push_back({1, value * 2});
    return result;
  }
  void validate_weights(const tide::NodeWeights&, tide::Index slots) const override {
    if (slots != 2) throw std::invalid_argument("custom Full requires two output slots");
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
    tide::Graph g; g.nodes = {{0, true}}; g.nodes[0].emission = "clock-full-v1";
    g.nodes[0].memory = "custom-slot-state-v1";
    g.edges = {{0, 0, 1}}; g.regions = {{1}}; g.inputs = {0}; g.outputs = {0}; g.compile();
    tide::Model m;
    m.nodes = {{at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::ones({2}, options)}};
    m.nodes[0].full_kernel = std::make_shared<ClockFull>();
    m.nodes[0].kernel = std::make_shared<SlotState>();
    m.input_scale = m.agg_scale = m.edge_scale = m.output_scale = {at::ones({}, options)};
    auto x = at::full({2}, 0.25, options).set_requires_grad(true);
    auto unused = at::full({2}, 0.4, options).set_requires_grad(true);
    tide::Continuation q; q.identity = g.identity; q.batch_size = 2;
    tide::Options runtime; runtime.packed = true; runtime.workers = 2;
    tide::Streaming engine(g, m, runtime);
    auto result = engine.run(q, {{0, 0, 0, 0, x}, {1, 0, 0, 0, unused}}, 3, 3);
    if (result.trace.size() != 4 || result.messages.size() != 2 || result.outputs.size() != 4
        || !result.continuation.pending.empty() || result.stats.at("full_scalar_fallback_steps") != 4)
      throw std::runtime_error("custom Full sparse/fallback mismatch");
    auto loss = result.outputs[0].value.sum() + result.outputs[2].value.sum();
    if (!at::equal(loss, at::full({}, 40, options))) throw std::runtime_error("custom Full did not read pre-clear state/clock");
    auto gradients = torch::autograd::grad({loss}, {x, unused}, {}, false, false, true);
    if (!at::equal(gradients[0], at::full_like(x, 24)) || gradients[1].defined())
      throw std::runtime_error("custom Full isolated VJP mismatch");
    for (const auto& event : result.trace) if (event.full.defined()) throw std::runtime_error("unexpected auxiliary value");
    for (const auto& [owner, state] : result.continuation.states)
      if (at::count_nonzero(state.value).item<int64_t>() || at::count_nonzero(state.slots.at("memory")).item<int64_t>()
          || state.observations != 2)
        throw std::runtime_error("custom Full changed persistent state");
    const std::string report = "custom-full-kernel: passed\n";
    if (!args.output_dir.empty()) {
      if (!std::filesystem::create_directories(args.output_dir)) throw std::runtime_error("failed to create output");
      std::ofstream file(std::filesystem::path(args.output_dir) / "result.txt"); file << report;
      if (!file) throw std::runtime_error("failed to write result");
    }
    std::cout << report;
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
