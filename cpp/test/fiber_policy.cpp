#include "tide/fiber_attention.h"
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
