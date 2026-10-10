#include "tide/lazy_add.h"
#include "tide/counters.h"
#include "tide/clocked_kernel.h"
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
  bool batched_autograd() const override { return true; }
  bool exact_sequence() const override { return true; }
  bool joint_sequence() const override { return true; }
  // Keep the time recurrence ordered, batching independent samples at each
  // depth. batch() buckets actual tick gaps and preserves every multiplication.
  PackedStates packed_sequence(const NodeWeights& w,const std::vector<State>& old,
                                const PackedSequence& p) const override {
    p.validate();
    if (old.size()!=p.owners.size()) throw std::invalid_argument("packed initial-state count mismatch");
    PackedStates result;result.states.resize(p.times.size());auto current=old;
    for (Index depth=0;;++depth) {
      std::vector<Index> owners,ids,times;std::vector<State> initial;
      std::vector<Tensor> content;ContentViews views;
      for (size_t i=0;i<old.size();++i) if (p.offsets[i]+depth<p.offsets[i+1]) {
        const auto j=p.offsets[i]+depth;owners.push_back(i);ids.push_back(j);
        initial.push_back(current[i]);content.push_back(p.contents[j]);
        views.push_back(p.views[j]);times.push_back(p.times[j]);
      }
      if (ids.empty()) break;
      auto states=batch(w,initial,at::stack(content),times,views);
      for (size_t i=0;i<ids.size();++i) current[owners[i]]=result.states[ids[i]]=std::move(states[i]);
      ++result.calls;result.max_length=depth+1;
      result.max_batch=std::max<Index>(result.max_batch,ids.size());
    }
    return result;
  }
  void validate_weights(const NodeWeights& w) const override {
    auto it = w.extra.find("add_retention");
    if (it == w.extra.end() || !it->second.defined() || it->second.dim() != 0
        || (it->second.scalar_type() != at::kFloat && it->second.scalar_type() != at::kDouble && it->second.scalar_type() != at::kHalf)
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
Tensor decode_add_repeat(const NodeWeights& w, const State& global, Index global_cut, std::optional<StateClock> policy) {
  const auto clock = policy.value_or(kernel_clock(w.kernel));
  const auto state = local_state(clock, global); const auto cut = clock.cut(global_cut);
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
