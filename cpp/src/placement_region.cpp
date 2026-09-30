#include "placement_internal.h"
#include "tide/counters.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide::placement_detail {
namespace {
Index count(const History& history, const char* name, Index node) {
  const auto& values = history.node_maps.at(name);
  const auto it = values.find(node);
  return it == values.end() ? 0 : it->second;
}
class PlacedRegion final : public RegionKernel {
 public:
  PlacedRegion(Region spec, ResolvedPlacement placement)
      : spec_(std::move(spec)), placement_(std::move(placement)), base_(make_region_kernel(spec_)) {}
  History initial(const RegionWeights& w, const RegionLayout& layout, const Tensor& ref) const override {
    return base_->initial(w, layout, ref);
  }
  Selection step(const RegionWeights& w, const RegionInput& r) const override {
    if (r.candidates.empty()) throw std::invalid_argument("empty placed region input");
    std::vector<Tensor> values;
    std::vector<Index> selected, affected, slots;
    const bool lh = spec_.selector == "lh-count-affect-v1";
    const bool memory = spec_.selector == "tensor-history-v1";
    for (const auto& c : r.candidates) {
      values.push_back(c.descriptor);
      selected.push_back(r.layout.spec.count_priority ? count(r.history, "selected", c.node) : 0);
      if (lh) affected.push_back(count(r.history, "affected", c.node));
      if (memory) slots.push_back(r.layout.slot(c.node));
    }
    // A differentiable region-time frame remains an independent autograd root.
    // Do not batch unrelated frames and thereby connect their zero gradients.
    auto descriptors = at::stack(values).to(placement_.control);
    Tensor scores = descriptors;
    if (memory) {
      const auto& bias = w.extra.at("bias");
      const auto ids = at::tensor(slots, at::kLong).to(bias.device());
      auto adjustment = r.history.tensors.at("memory").to(placement_.control) *
                        bias.index_select(0, ids).to(placement_.control);
      scores = descriptors + adjustment;
    }
    if (!at::isfinite(scores).all().item<bool>()) throw std::invalid_argument("nonfinite region selector score");
    Tensor active;
    {
      at::NoGradGuard guard;
      const auto numeric = scores.to(placement_.selection);
      auto order = at::argsort(numeric, true, -1, true);
      // Stable least-to-most-significant passes preserve node-ID tie order.
      // Counters are int64 throughout, including device sorting above 2^53.
      if (lh) {
        auto counts = at::tensor(affected, at::kLong).to(placement_.selection);
        order = order.index_select(0, at::argsort(counts.index_select(0, order), true, -1, true));
      }
      if (r.layout.spec.count_priority) {
        auto counts = at::tensor(selected, at::kLong).to(placement_.selection);
        order = order.index_select(0, at::argsort(counts.index_select(0, order), true, -1, false));
      }
      if (spec_.selector == "positive-v1") {
        auto eligible = numeric.gt(0).to(at::kLong);
        order = order.index_select(0, at::argsort(eligible.index_select(0, order), true, -1, true));
      }
      auto take = at::arange(order.numel(), order.options()).lt(r.layout.spec.budget);
      if (spec_.selector == "positive-v1") take = take.logical_and(numeric.index_select(0, order).gt(0));
      active = at::zeros_like(take).scatter(0, order, take).to(at::kCPU).contiguous();
    }
    Selection result;
    result.history = r.history;
    result.history.last_time = r.time;
    const auto bits = active.accessor<bool, 1>();
    const auto controls = at::softmax(scores, 0).to(r.payload_options);
    for (size_t i = 0; i < r.candidates.size(); ++i) {
      const auto node = r.candidates[i].node;
      if (bits[i]) {
        result.active.insert(node);
        result.history.node_maps["selected"][node] = increment(count(r.history, "selected", node));
      }
      if (lh) result.history.node_maps["affected"][node] = increment(count(r.history, "affected", node));
      result.controls.emplace(node, controls[i]);
    }
    if (memory) {
      // Preserve the profile's payload-dtype recurrence and cast-before-add.
      auto total = descriptors.sum().to(r.payload_options.device(placement_.control));
      auto next = w.extra.at("alpha").to(placement_.control) * r.history.tensors.at("memory").to(placement_.control) + total;
      result.history.tensors["memory"] = next.to(r.payload_options);
    }
    return result;
  }
  void validate_weights(const RegionWeights& w, const RegionLayout& layout) const override { base_->validate_weights(w, layout); }
  void validate_history(const History& h, const RegionLayout& layout) const override { base_->validate_history(h, layout); }
  bool matches(const Region& spec) const { return spec.selector == spec_.selector; }
 private:
  Region spec_;
  ResolvedPlacement placement_;
  std::shared_ptr<const RegionKernel> base_;
};
}
std::shared_ptr<const RegionKernel> region_kernel(const Region& spec, const RegionWeights& w, const ResolvedPlacement& p) {
  const auto previous = dynamic_cast<const PlacedRegion*>(w.kernel.get());
  if (w.kernel && !(previous ? previous->matches(spec) : builtin_region(*w.kernel, spec)))
    throw std::invalid_argument("placement cannot replace a custom or mismatched Region kernel");
  return std::make_shared<PlacedRegion>(spec, p);
}
}  // namespace tide::placement_detail
