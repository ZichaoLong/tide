#include "tide/region.h"
#include "tide/counters.h"
#include "tide/full.h"
#include "tide/stream.h"
#include "tide/frontier.h"
#include "tide/cursor.h"
#include <torch/csrc/autograd/autograd.h>
#include <atomic>
#include <stdexcept>

namespace {
using namespace tide;
class ClockSelector final : public RegionKernel {
 public:
  explicit ClockSelector(int invalid = -1) : invalid_(invalid) {}
  mutable std::atomic<Index> validations{0};
  History initial(const RegionWeights&, const RegionLayout&, const Tensor& ref) const override {
    return {-1, {{"calls", 0}, {"signed", -2}}, {{"seen", {}}}, {{"memory", at::ones({}, ref.options())}}};
  }
  Selection step(const RegionWeights& w, const RegionInput& r) const override {
    Selection out; out.history = r.history; out.history.last_time = r.time;
    out.history.scalars["calls"] = increment(r.history.scalars.at("calls"));
    auto memory = w.extra.at("gain")*r.history.tensors.at("memory")+r.time;
    for (const auto& c : r.candidates) {
      auto& count = out.history.node_maps.at("seen")[c.node]; count = increment(count);
      memory = memory+c.descriptor;
      out.controls[c.node] = at::stack({r.history.tensors.at("memory"), c.descriptor});
    }
    out.history.tensors["memory"] = memory;
    if (r.history.scalars.at("calls") % 2) out.active.insert(r.candidates[0].node);
    if (invalid_ == 0) out.active.insert(9);
    if (invalid_ == 1) for (const auto& c : r.candidates) out.active.insert(c.node);
    if (invalid_ == 2) out.controls.erase(r.candidates[0].node);
    if (invalid_ == 3) out.controls[r.candidates[0].node] = memory/0;
    if (invalid_ == 4) out.history.last_time = r.time+1;
    if (invalid_ == 5) out.history.node_maps["seen"][9] = 0;
    if (invalid_ == 6) out.history.tensors["memory"] = at::ones({2}, memory.options());
    return out;
  }
  void validate_weights(const RegionWeights&, const RegionLayout&) const override {}
  void validate_history(const History& h, const RegionLayout&) const override {
    ++validations;
    if (h.scalars.size() != 2 || !h.scalars.count("calls") || !h.scalars.count("signed")
        || h.node_maps.size() != 1 || !h.node_maps.count("seen") || h.tensors.size() != 1
        || !h.tensors.count("memory") || h.tensors.at("memory").dim() != 0)
      throw std::invalid_argument("custom history layout mismatch");
  }
 private:
  int invalid_;
};
class VectorFull final : public FullKernel {
 public:
  FullResult step(const NodeWeights&, const FullInput& r, Index slots, const Options&) const override {
    auto value = r.comparison->value+r.content.value+r.control.sum();
    FullResult out{value, {}};
    for (Index slot = 0; slot < slots; ++slot) out.emitted.push_back({slot, value});
    return out;
  }
  void validate_weights(const NodeWeights&, Index) const override {}
};
void equal(const Tensor& actual, double value) {
  if (!at::allclose(actual, at::full_like(actual, value), 1e-5, 1e-6))
    throw std::runtime_error("custom region analytic value/VJP mismatch");
}
}  // namespace

void check_region_programs(const at::TensorOptions& opts) {
  for (int schedule = 0; schedule < 5; ++schedule) {
    Graph g; g.nodes = {{0}, {0}}; g.inputs = {0, 1}; g.outputs = {0, 1};
    for (auto& n : g.nodes) n.emission = "vector-control-v1";
    g.regions = {{1, true, false, "content", "clock-selector-v1"}}; g.compile();
    Model m;
    for (int i = 0; i < 2; ++i) {
      NodeWeights w{at::zeros({1}, opts), at::zeros({1, 1}, opts), at::zeros({1}, opts), at::ones({1}, opts)};
      w.full_kernel = std::make_shared<VectorFull>(); m.nodes.push_back(w);
    }
    m.input_scale = m.output_scale = {at::ones({}, opts), at::ones({}, opts)};
    auto gain = at::full({}, 2., opts); gain.set_requires_grad(true);
    auto kernel = std::make_shared<ClockSelector>(); m.regions = {{{{"gain", gain}}, kernel}};
    auto leaf = [&](double value) { auto t = at::full({1}, value, opts); t.set_requires_grad(true); return t; };
    auto x = leaf(1), y = leaf(3), z = leaf(0), a = leaf(2), unused = leaf(8);
    auto h = at::ones({}, opts); h.set_requires_grad(true);
    Continuation q; q.identity = g.identity; q.batch_size = 2;
    auto history = kernel->initial(m.regions[0], region_layout(g, 0), h); history.tensors["memory"] = h;
    q.history[{0, 0}] = history;
    std::vector<External> xs{{0, 0, 0, 1, x}, {0, 1, 0, 1, y}, {0, 0, 1, 4, z}, {0, 1, 1, 4, a},
                             {1, 0, 0, 1, unused}, {1, 0, 1, 4, unused}};
    Options options; options.workers = schedule ? 2 : 1; options.packed = schedule > 1;
    options.parallel_regions = options.compact_events = schedule == 4;
    Result result;
    if (schedule == 3) { Frontier engine(g, m, options); result = engine.run(q, xs, 6, 6); }
    else { Streaming engine(g, m, options); result = engine.run(q, xs, 6, 6); }
    const auto& final = result.continuation.history.at({0, 0});
    equal(final.tensors.at("memory"), 20);
    if (final.last_time != 4 || final.scalars.at("calls") != 2 || final.node_maps.at("seen").at(0) != 2)
      throw std::runtime_error("custom region metadata mismatch");
    if (result.outputs.size() != 2 || result.outputs[0].batch != 0 || result.outputs[0].port != 0)
      throw std::runtime_error("custom region empty selection/routing mismatch");
    equal(result.outputs[0].value, 7.5);
    auto grads = torch::autograd::grad({final.tensors.at("memory")}, {x, y, z, a, h, gain, unused}, {}, true, false, true);
    for (size_t i = 0; i < 6; ++i) equal(grads[i], std::vector<double>{2, 2, 1, 1, 4, 9}[i]);
    if (grads[6].defined()) throw std::runtime_error("custom region coupled independent sample gradients");
    grads = torch::autograd::grad({result.outputs[0].value.sum()}, {x, y, z, a, h, gain, unused}, {}, false, false, true);
    equal(grads[0], 1.5); equal(grads[1], 1); equal(grads[2], 3); equal(grads[4], 2); equal(grads[5], 1);
    if (grads[3].defined() || grads[6].defined()) throw std::runtime_error("custom region Full gained absent gradients");
    // Imported idle region histories validate once; an advance only checks touched histories.
    q.batch_size = 128;
    for (Index b = 1; b < 128; ++b) q.history[{b, 0}] = history;
    auto checked = kernel->validations.load();
    Streaming engine(g, m, options); StreamingCursor cursor(engine, q);
    if (kernel->validations != checked+128) throw std::runtime_error("region cursor import count mismatch");
    cursor.advance({xs[0], xs[1]}, 3, 3);
    if (kernel->validations != checked+129) throw std::runtime_error("region cursor rescanned idle histories");
    auto snapshot = cursor.snapshot(); snapshot.history.at({0, 0}).tensors.at("memory").fill_(99);
    equal(cursor.snapshot().history.at({0, 0}).tensors.at("memory"), 7);
    for (int invalid = 0; invalid < 7; ++invalid) {
      auto bad = m; bad.regions[0].kernel = std::make_shared<ClockSelector>(invalid);
      bool rejected = false;
      try { Streaming broken(g, bad, options); broken.run(q, {xs[0], xs[1], xs[4]}, 3, 3); }
      catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected) throw std::runtime_error("malformed custom region result was accepted");
    }
  }
}
