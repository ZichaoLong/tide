#include "tide/operator_profile.h"
#include "tide/next.h"
#include "tide/kernel.h"
#include "tide/autograd.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide {
namespace {
class AdoptNext final : public NextKernel {
 public:
  State step(const NodeWeights&, const NextInput& r) const override { return r.comparison; }
  bool comparison_identity() const override { return true; }
  bool joint_batch() const override { return true; }
  std::vector<State> batch(const NodeWeights&, const std::vector<NextInput>& requests) const override {
    std::vector<State> result;
    for (const auto& request : requests) result.push_back(request.comparison);
    return result;
  }
  void validate_weights(const NodeWeights&) const override {}
};
class ControlBlendNext final : public NextKernel {
 public:
  State step(const NodeWeights&, const NextInput& r) const override {
    if (r.control.dim() != 0) throw std::invalid_argument("control-blend Next requires scalar control");
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
  op_profile::Scope profile(op_profile::Next);
  auto result = w.next_kernel->step(w, request);
  if (!w.next_kernel->comparison_identity()) validate(result, w, request.time);
  if (node.clear && request.active) result = w.kernel->reset(result);
  return result;
}
std::vector<State> evaluate_next_batch(const Node& node, const NodeWeights& w, const std::vector<NextInput>& requests) {
  if (!w.next_kernel->joint_batch()) {
    std::vector<State> result;
    for (const auto& request : requests) result.push_back(evaluate_next(node, w, request));
    return result;
  }
  op_profile::Scope profile(op_profile::Next);
  std::vector<State> result;
  {
    at::NoGradGuard guard;
    result = w.next_kernel->batch(w, requests);
    if (result.size() != requests.size()) throw std::invalid_argument("Next batch changed event count");
    for (size_t i = 0; i < result.size(); ++i)
      if (!w.next_kernel->comparison_identity()) validate(result[i], w, requests[i].time);
    if (node.clear) {
      std::vector<State> selected;
      for (size_t i = 0; i < requests.size(); ++i) if (requests[i].active) selected.push_back(result[i]);
      auto reset = w.kernel->reset_batch(selected);
      if (reset.size() != selected.size()) throw std::invalid_argument("state reset batch changed event count");
      size_t k = 0;
      for (size_t i = 0; i < requests.size(); ++i) if (requests[i].active) result[i] = std::move(reset[k++]);
    }
  }
  if (at::GradMode::is_enabled()) for (size_t i = 0; i < requests.size(); ++i)
    result[i] = semantic_state(result[i], evaluate_next(node, w, requests[i]));
  return result;
}
}  // namespace tide
