#include "portable_torch/runtime.hpp"
#include "tide/kernel.h"
#include "tide/stream.h"
#include "tide/cursor.h"
#include <atomic>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <filesystem>
#include <fstream>
#include <iostream>

void check_content_programs(const at::TensorOptions&);
void check_read_programs(const at::TensorOptions&);
void check_next_programs(const at::TensorOptions&);
void check_region_programs(const at::TensorOptions&);

namespace {
class CustomAccumulator final : public tide::StateKernel {
 public:
  mutable std::atomic<int> validations{0};  // Diagnostic only; no execution state in the program.
  tide::State initial(const tide::NodeWeights& w) const override {
    return {at::zeros_like(w.bias), -1, 0, {{"sum", at::zeros_like(w.bias)}}};
  }
  tide::State step(const tide::NodeWeights&, const tide::State& old, const tide::ContentView& content,
                   tide::Index time) const override {
    const auto& h = content.value;
    if (content.sources.empty()) throw std::invalid_argument("custom kernel requires a real fiber");
    return {old.value + 2 * h, time, old.observations + 1, {{"sum", old.slots.at("sum") + 3 * h}}};
  }
  void validate_weights(const tide::NodeWeights&) const override {}
  void validate_state(const tide::NodeWeights& w, const tide::State& s) const override {
    ++validations;
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
    auto kernel = std::make_shared<CustomAccumulator>();
    m.nodes[0].kernel = kernel;
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
    // Public packed contract: ragged independent sequences and the default scalar fallback.
    std::vector<tide::SourceInput> sources{{0, {0, 0, 0, 0, 0, 0, x[0]}, m.input_scale[0]}};
    tide::PackedSequence packed{at::ones({3, 2}, options) * 0.25, {0, 1, 3}, {{0, 0}, {1, 0}},
                                {0, 0, 2}, {}};
    for (tide::Index i = 0; i < 3; ++i) packed.views.push_back({packed.contents[i], sources, {}});
    auto state = m.nodes[0].kernel->initial(m.nodes[0]);
    auto batch = m.nodes[0].kernel->packed_sequence(m.nodes[0], {state, state}, packed);
    if (batch.calls != 2 || batch.max_batch != 1 || batch.max_length != 2
        || !at::equal(batch.states.back().value, at::ones({2}, options)))
      throw std::runtime_error("custom kernel packed fallback mismatch");
    for (int invalid = 0; invalid < 3; ++invalid) {
      auto malformed = packed;
      if (invalid == 0) malformed.offsets = {0, 0, 3};
      if (invalid == 1) malformed.owners[1] = malformed.owners[0];
      if (invalid == 2) malformed.times[2] = malformed.times[1];
      bool rejected = false;
      try { malformed.validate(); } catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected) throw std::runtime_error("invalid packed metadata was accepted");
    }
    auto imported = result.continuation; imported.batch_size = 512;
    for (tide::Index b = 2; b < imported.batch_size; ++b) imported.states[{b, 0}] = state;
    const auto checked = kernel->validations.load();
    tide::StreamingCursor cursor(engine, imported);
    if (kernel->validations != checked + 512) throw std::runtime_error("cursor import validation mismatch");
    auto advanced = cursor.advance({}, 3, 3);
    if (advanced.cut != 3 || advanced.stats.at("candidate_events") != 2)
      throw std::runtime_error("cursor sparse advance mismatch");
    cursor.advance({}, 5, 5);
    if (kernel->validations != checked + 512) throw std::runtime_error("cursor rescanned imported state");
    if (cursor.snapshot().states.size() != 512) throw std::runtime_error("cursor snapshot lost idle state");
    check_content_programs(options);
    check_read_programs(options);
    check_next_programs(options);
    check_region_programs(options);
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
