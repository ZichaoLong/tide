#include "tide/next.h"
#include "tide/kernel.h"
#include <stdexcept>

namespace tide {
namespace {
class AdoptNext final : public NextKernel {
 public:
  State step(const NodeWeights&, const NextInput& r) const override { return r.comparison; }
  bool comparison_identity() const override { return true; }
  void validate_weights(const NodeWeights&) const override {}
};
class ControlBlendNext final : public NextKernel {
 public:
  State step(const NodeWeights&, const NextInput& r) const override {
    auto result = r.comparison;
    const auto& a = r.old; const auto& b = r.comparison; const auto& c = r.control;
    if (a.slots.size() != b.slots.size()) throw std::invalid_argument("control-blend Next requires matching slot shapes");
    for (const auto& [name, value] : a.slots) {
      auto it = b.slots.find(name);
      if (it == b.slots.end() || it->second.sizes() != value.sizes())
        throw std::invalid_argument("control-blend Next requires matching slot shapes");
      result.slots[name] = (1-c)*value+c*it->second;
    }
    result.value = (1-c)*a.value+c*b.value;
    return result;
  }
  void validate_weights(const NodeWeights&) const override {}
};
void validate(const State& state, const NodeWeights& w, Index time) {
  if (state.last_time < -1 || state.last_time > time || state.observations < 0)
    throw std::invalid_argument("Next returned invalid state clocks");
  auto check = [&](const Tensor& value) {
    if (!value.defined() || value.device() != w.bias.device() || value.scalar_type() != w.bias.scalar_type())
      throw std::invalid_argument("Next returned incompatible state dtype/device");
    if (!at::isfinite(value).all().item<bool>()) throw std::invalid_argument("Next returned nonfinite state");
  };
  check(state.value);
  if (state.value.sizes() != w.bias.sizes()) throw std::invalid_argument("Next returned incompatible state shape");
  for (const auto& [name, value] : state.slots) check(value);
  w.kernel->validate_state(w, state);
}
}  // namespace
std::shared_ptr<const NextKernel> make_next_kernel(const Node& node) {
  if (node.next_state == "adopt-v1") return std::make_shared<AdoptNext>();
  if (node.next_state == "control-blend-v1") return std::make_shared<ControlBlendNext>();
  throw std::invalid_argument("unknown Next profile: " + node.next_state);
}
State evaluate_next(const Node& node, const NodeWeights& w, const NextInput& request) {
  auto result = w.next_kernel->step(w, request);
  if (!w.next_kernel->comparison_identity()) validate(result, w, request.time);
  if (node.clear && request.active) result = w.kernel->reset(result);
  return result;
}
}  // namespace tide
