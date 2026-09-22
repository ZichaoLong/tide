#include "tide/fiber_attention.h"
#include "tide/operator_work.h"
#include "tide/stream.h"
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
Model model(const at::TensorOptions& opts) {
  NodeWeights w{at::zeros({2}, opts), at::eye(2, opts), at::zeros({2}, opts), at::ones({2}, opts)};
  w.extra = {{"fiber_qkv", at::ones({2, 6}, opts)}, {"fiber_qkv_bias", at::zeros({6}, opts)},
             {"fiber_out", at::eye(2, opts)}, {"fiber_out_bias", at::zeros({2}, opts)},
             {"fiber_decay", at::scalar_tensor(.01, opts)}, {"fiber_pool", at::ones({3}, opts)}};
  Model result; result.nodes = {w};
  result.input_scale = {at::ones({}, opts), at::ones({}, opts), at::ones({}, opts)};
  result.output_scale = {at::ones({}, opts)};
  return result;
}
}  // namespace

void check_fiber_policy(const at::TensorOptions& opts) {
  tide::Graph g; g.nodes = {{0}}; g.nodes[0].memory = "lh-fiber-attention-all-softmax-repeat-v1";
  g.inputs = {0, 0, 0}; g.outputs = {0}; g.regions = {{1}}; g.compile();
  // Analytic ragged two-event prefill: queries [2,1], initial cache [0,2].
  // Compare every resulting state with counting disabled, including after reset.
  for (Index cache : {0, 2}) {
    auto w = model(opts).nodes[0]; auto kernel = tide::make_fiber_attention_kernel(g.nodes[0], 3);
    auto old = kernel->initial(w);
    old.slots["key"] = old.slots["value"] = at::ones({cache, 1, 2}, opts);
    old.slots["log_bias"] = at::zeros({cache}, opts);
    std::vector<SourceInput> a{{0, {0, 0, 0, 0, 0, 0, at::ones({2}, opts)}, at::ones({}, opts)}};
    a.push_back(a[0]); a[1].slot = 1;
    std::vector<SourceInput> b{a[0]};
    ContentViews views{{at::ones({2}, opts), a, {}}, {at::ones({2}, opts), b, {}}};
    PackedSequence batch{at::ones({2, 2}, opts), {0, 2}, {{0, 0}}, {1, 3}, views};
    work::reset(false); auto expected = kernel->packed_sequence(w, {old}, batch);
    work::reset(true); auto actual = kernel->packed_sequence(w, {old}, batch);
    auto counters = work::metrics(); work::reset(false);
    if (counters.at("op/qkv_rows") != 3 || counters.at("op/out_rows") != 2 ||
        counters.at("op/qkv_flops") != 72 || counters.at("op/out_flops") != 16 ||
        counters.at("op/valid_score_elements") != 7+3*cache ||
        counters.at("op/executed_score_elements") != 9+3*cache)
      throw std::runtime_error("analytic ragged operator counts");
    for (size_t i = 0; i < actual.states.size(); ++i) {
      if (!at::equal(actual.states[i].value, expected.states[i].value))
        throw std::runtime_error("counting changed attention output");
      for (const auto& [key, value] : actual.states[i].slots)
        if (!at::equal(value, expected.states[i].slots.at(key))) throw std::runtime_error("counting changed cache");
    }
    work::reset(true);
    kernel->step(w, kernel->reset(actual.states.back()), views[1], 5);
    counters = work::metrics(); work::reset(false);
    if (counters.at("op/valid_score_elements") != 1 || counters.at("op/qkv_flops") != 24)
      throw std::runtime_error("reset cache counted incorrectly");
  }
  // Exercise native model construction directly, without Python pre-validation.
  tide::Streaming valid(g, model(opts), {});
  for (int invalid = 0; invalid < 7; ++invalid) {
    auto m = model(opts);
    auto& w = m.nodes[0];
    if (invalid == 0) w.extra.erase("fiber_pool");
    if (invalid == 1) w.extra["fiber_pool"] = at::ones({1, 3}, opts);
    if (invalid == 2) w.extra["fiber_pool"] = at::ones({2}, opts);
    if (invalid == 3) w.extra["fiber_pool"] = at::ones({3}, opts.dtype(opts.dtype().toScalarType() == at::kDouble ? at::kFloat : at::kDouble));
    if (invalid == 4) w.extra["fiber_pool"] = at::full({3}, std::numeric_limits<double>::quiet_NaN(), opts);
    if (invalid == 5) {
      // The cached kernel and parameter agree with each other, but not with the graph.
      w.kernel = tide::make_fiber_attention_kernel(g.nodes[0], 2);
      w.extra["fiber_pool"] = at::ones({2}, opts);
    }
    if (invalid == 6) {
      auto other = g.nodes[0]; other.memory = "lh-fiber-attention-linear-repeat-v1";
      w.kernel = tide::make_fiber_attention_kernel(other, 3);
    }
    bool rejected = false;
    try { tide::Streaming bad(g, m, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("native fiber pooling accepted incompatible parameter/policy/domain");
  }
}
