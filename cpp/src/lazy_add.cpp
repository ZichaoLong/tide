#include "tide/lazy_add.h"
#include "tide/counters.h"
#include <cstdint>
#include <stdexcept>

namespace tide {
namespace {
using Ticks = std::uint64_t;
Ticks gap(Index last, Index time) {
  if (last < -1 || time < last) throw std::invalid_argument("invalid Add tick clock");
  // Avoid signed overflow even for last=-1, time=INT64_MAX.
  return (static_cast<Ticks>(time) + 1) - static_cast<Ticks>(last + 1);
}
Tensor repeat(Tensor value, const Tensor& retention, Ticks ticks) {
  for (Ticks i = 0; i < ticks; ++i) value = value * retention;
  return value;
}
class AddRepeat final : public StateKernel {
 public:
  State initial(const NodeWeights& w) const override { return {at::zeros_like(w.bias)}; }
  State step(const NodeWeights& w, const State& old, const ContentView& h, Index time) const override {
    if (time < 0 || time <= old.last_time) throw std::invalid_argument("Add requires increasing tick times");
    auto observations = increment(old.observations);
    return {h.value + repeat(old.value, w.extra.at("add_retention"), gap(old.last_time, time)), time, observations};
  }
  std::vector<State> batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                           const std::vector<Index>& times, const ContentViews&) const override {
    if (old.empty() || h.dim() != 2 || h.size(0) != static_cast<Index>(old.size()) || times.size() != old.size())
      throw std::invalid_argument("invalid Add batch metadata");
    // Bucket actual samples by elapsed ticks. No padded observations or masked
    // excess decay, and no value-dependent shortcut for zero hidden/retention.
    std::map<Ticks, std::vector<Index>> groups;
    for (size_t i = 0; i < old.size(); ++i) {
      if (times[i] < 0 || times[i] <= old[i].last_time) throw std::invalid_argument("Add requires increasing tick times");
      increment(old[i].observations);
      groups[gap(old[i].last_time, times[i])].push_back(i);
    }
    std::vector<State> result(old.size());
    for (const auto& [ticks, ids] : groups) {
      std::vector<Tensor> previous, content;
      for (auto i : ids) { previous.push_back(old[i].value); content.push_back(h[i]); }
      auto values = at::stack(content) + repeat(at::stack(previous), w.extra.at("add_retention"), ticks);
      for (size_t row = 0; row < ids.size(); ++row) {
        auto i = ids[row];
        result[i] = {values[row].clone(), times[i], increment(old[i].observations)};
      }
    }
    return result;
  }
  bool joint_batch() const override { return true; }
  bool exact_sequence() const override { return true; }
  // The base sequence/packed_sequence loops preserve multiplication order and
  // report scalar_steps. This is not an associative time scan.
  void validate_weights(const NodeWeights& w) const override {
    auto it = w.extra.find("add_retention");
    if (it == w.extra.end() || !it->second.defined() || it->second.dim() != 0 || !it->second.device().is_cpu()
        || (it->second.scalar_type() != at::kFloat && it->second.scalar_type() != at::kDouble)
        || it->second.scalar_type() != w.bias.scalar_type() || it->second.device() != w.bias.device()
        || !at::isfinite(it->second).item<bool>())
      throw std::invalid_argument("Add requires finite payload-dtype scalar retention");
  }
  void validate_state(const NodeWeights&, const State& state) const override {
    if (!state.slots.empty()) throw std::invalid_argument("Add state has unexpected slots");
  }
};
}  // namespace
std::shared_ptr<const StateKernel> make_add_repeat_kernel() { return std::make_shared<AddRepeat>(); }
Tensor decode_add_repeat(const NodeWeights& w, const State& state, Index cut) {
  AddRepeat kernel; kernel.validate_weights(w); kernel.validate_state(w, state);
  if (cut < 0 || state.last_time < -1 || state.last_time >= cut || state.observations < 0)
    throw std::invalid_argument("invalid Add cut/state clock");
  if (!state.value.defined() || state.value.dim() != 1 || state.value.sizes() != w.bias.sizes()
      || state.value.scalar_type() != w.bias.scalar_type() || state.value.device() != w.bias.device()
      || !at::isfinite(state.value).all().item<bool>())
    throw std::invalid_argument("incompatible Add state value");
  return repeat(state.value, w.extra.at("add_retention"), gap(state.last_time, cut - 1));
}
}  // namespace tide
