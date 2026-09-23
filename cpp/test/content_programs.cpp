#include "tide/stream.h"
#include "tide/frontier.h"
#include "tide/kernel.h"
#include "tide/full.h"
#include "tide/read.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>

namespace {
using namespace tide;
class SourceMemory final : public StateKernel {
 public:
  State initial(const NodeWeights& w) const override {
    return {at::zeros_like(w.bias), -1, 0, {{"sum", at::zeros_like(w.bias)}}};
  }
  State step(const NodeWeights& w, const State& old, const ContentView& content, Index time) const override {
    Tensor value, total;
    for (const auto& s : content.sources) {
      auto term = (s.slot+1)*(s.atom.kind+1)*(s.atom.position+1)*s.atom.value*s.scale;
      value = value.defined() ? value + term : term;
    }
    for (const auto& term : content.contributions) {
      auto x = (term.slot+1)*term.value;
      total = total.defined() ? total + x : x;
    }
    return {old.value+w.extra.at("source_gain")*value, time, old.observations+1, {{"sum", old.slots.at("sum")+total}}};
  }
  bool exact_sequence() const override { return true; }  // The base sequence is a counted scalar fallback.
  void validate_weights(const NodeWeights& w) const override {
    if (!w.extra.count("source_gain")) throw std::invalid_argument("missing source gain");
  }
  void validate_state(const NodeWeights& w, const State& state) const override {
    if (state.slots.size() != 1 || !state.slots.count("sum") || state.slots.at("sum").sizes() != w.bias.sizes())
      throw std::invalid_argument("source memory slot mismatch");
  }
};
class SourceRead final : public ReadKernel {
 public:
  Tensor step(const NodeWeights&, const ReadInput& r) const override {
    if (!r.state) throw std::invalid_argument("source Read requires a state");
    return r.state->value.sum()+r.content.contributions[0].value.sum();
  }
  void validate_weights(const NodeWeights&) const override {}
};
class SourceFull final : public FullKernel {
 public:
  FullResult step(const NodeWeights&, const FullInput& request, Index slots, const Options&) const override {
    if (slots != 1 || request.content.contributions.size() != 2) throw std::invalid_argument("source Full domain");
    Tensor source;
    for (const auto& s : request.content.sources) {
      auto value = (s.atom.kind+s.atom.position)*s.atom.value;
      source = source.defined() ? source+value : value;
    }
    return {{}, {{0, request.comparison->value+source+request.content.contributions[1].value}}};
  }
  void validate_weights(const NodeWeights&, Index slots) const override {
    if (slots != 1) throw std::invalid_argument("source Full slots");
  }
};
}

void check_content_programs(const at::TensorOptions& options) {
  using namespace tide;
  for (bool clear : {false, true}) for (bool encoded : {false, true}) for (int algorithm : {0, 1, 2, 3}) {
    Graph g; g.nodes = {{0, clear}}; g.nodes[0].memory = "source-memory-v1";
    g.nodes[0].emission = "source-full-v1"; g.nodes[0].readout = "source-read-v1"; g.regions = {{1}}; g.inputs = {0, 0}; g.outputs = {0};
    Model m; m.nodes = {{at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::ones({2}, options)}};
    auto gain = at::full({}, 2, options).set_requires_grad(true);
    m.nodes[0].extra["source_gain"] = gain; m.nodes[0].kernel = std::make_shared<SourceMemory>();
    m.nodes[0].read_kernel = std::make_shared<SourceRead>();
    m.nodes[0].full_kernel = std::make_shared<SourceFull>();
    m.input_scale = {at::ones({}, options), at::ones({}, options)}; m.output_scale = {at::ones({}, options)};
    if (encoded) {
      g.nodes.push_back({1, false, true}); g.regions.push_back({1}); g.edges = {{1, 0, 1}, {1, 0, 1}};
      g.inputs = {1}; g.origins = {{0, 0, 3}, {1, 1, 3}};
      m.nodes.push_back({at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::zeros({2}, options)});
      m.agg_scale = m.edge_scale = {at::ones({}, options), at::ones({}, options)}; m.input_scale.resize(1);
    }
    g.compile();
    auto x = at::ones({2}, options).set_requires_grad(true), y = at::full({2}, 3, options).set_requires_grad(true);
    auto unused = at::full({2}, .4, options).set_requires_grad(true);
    std::vector<External> xs;
    for (Index b = 0; b < 2; ++b) for (Index port = 0; port < (encoded ? 1 : 2); ++port) {
      xs.push_back({b, port, 0, encoded ? 0 : 1, b == 0 ? x : unused});
      xs.push_back({b, port, 1, encoded ? 3 : 4, b == 0 ? y : unused});
    }
    Continuation q; q.identity = g.identity; q.batch_size = 2;
    Options runtime; runtime.packed = algorithm != 0; runtime.workers = algorithm == 0 ? 1 : 2;
    runtime.packed_sources = runtime.batch_next = algorithm == 3;
    Result result = algorithm == 2 ? Frontier(g, m, runtime).run(q, xs, 5, 5) : Streaming(g, m, runtime).run(q, xs, 5, 5);
    auto loss = result.outputs[0].value.sum()+result.outputs[2].value.sum();
    if (!at::equal(loss, at::full({}, clear ? 104 : 116, options))) throw std::runtime_error("complete content forward mismatch");
    auto grad = torch::autograd::grad({loss}, {x, y, unused, gain}, {}, false, false, true);
    if (!at::equal(grad[0], at::full_like(x, clear ? 7 : 13)) || !at::equal(grad[1], at::full_like(y, 15))
        || grad[2].defined() || !at::equal(grad[3], at::full_like(gain, clear ? 42 : 48)))
      throw std::runtime_error("complete content isolated VJP mismatch");
    for (const auto& event : result.trace) if (event.node == 0 && event.batch == 0) {
      auto expected = event.time == 1 ? 14 : clear ? 78 : 90;
      if (!at::equal(event.descriptor, at::full({}, expected, options))) throw std::runtime_error("complete content Read mismatch");
    }
    if (algorithm == 1 && result.stats.at("state_scalar_batch_steps") != 4)
      throw std::runtime_error("state batch fallback was not counted");
    if (algorithm == 2 && !clear && result.stats.at("state_scalar_sequence_steps") != 4)
      throw std::runtime_error("state sequence fallback was not counted");
  }
}
