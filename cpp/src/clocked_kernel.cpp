#include "tide/clocked_kernel.h"
#include <stdexcept>

namespace tide {
State local_state(const StateClock& clock, const State& state) {
  auto result = state; result.last_time = clock.to_local(state.last_time); return result;
}
State global_state(const StateClock& clock, const State& state) {
  auto result = state; result.last_time = clock.to_global(state.last_time); return result;
}
namespace {
class ClockedKernel final : public StateKernel {
  std::shared_ptr<const StateKernel> program_;
  StateClock clock_;
  std::vector<State> local(const std::vector<State>& states) const {
    std::vector<State> result;
    for (const auto& state : states) result.push_back(local_state(clock_, state));
    return result;
  }
  std::vector<Index> times(const std::vector<Index>& ts) const {
    std::vector<Index> result;
    for (auto t : ts) result.push_back(event(t));
    return result;
  }
  std::vector<State> global(std::vector<State> states) const {
    for (auto& state : states) state.last_time = clock_.to_global(state.last_time);
    return states;
  }
  Index event(Index t) const {
    if (t < 0) throw std::invalid_argument("invalid state clock coordinate");
    return clock_.to_local(t);
  }
 public:
  ClockedKernel(std::shared_ptr<const StateKernel> program, StateClock clock) : program_(std::move(program)), clock_(clock) {}
  const StateClock& clock() const { return clock_; }
  State initial(const NodeWeights& w) const override { return global_state(clock_, program_->initial(w)); }
  State step(const NodeWeights& w, const State& old, const ContentView& h, Index t) const override {
    return global_state(clock_, program_->step(w, local_state(clock_, old), h, event(t)));
  }
  std::vector<State> batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                            const std::vector<Index>& ts, const ContentViews& views) const override {
    return global(program_->batch(w, local(old), h, times(ts), views));
  }
  std::vector<State> sequence(const NodeWeights& w, const State& old, const Tensor& h,
                               const std::vector<Index>& ts, const ContentViews& views) const override {
    return global(program_->sequence(w, local_state(clock_, old), h, times(ts), views));
  }
  PackedStates packed_sequence(const NodeWeights& w, const std::vector<State>& old, const PackedSequence& packed) const override {
    packed.validate();
    auto local_batch = packed; local_batch.times = times(packed.times);
    auto result = program_->packed_sequence(w, local(old), local_batch);
    result.states = global(std::move(result.states)); return result;
  }
  bool exact_sequence() const override { return program_->exact_sequence(); }
  bool joint_batch() const override { return program_->joint_batch(); }
  bool joint_sequence() const override { return program_->joint_sequence(); }
  State reset(const State& state) const override { return global_state(clock_, program_->reset(local_state(clock_, state))); }
  bool joint_reset_batch() const override { return program_->joint_reset_batch(); }
  std::vector<State> reset_batch(const std::vector<State>& states) const override {
    return global(program_->reset_batch(local(states)));
  }
  void validate_policy(const Node& node, Index slots) const override {
    if (node.state_clock != clock_) throw std::invalid_argument("shared state program does not match state clock");
    program_->validate_policy(node, slots);
  }
  void validate_weights(const NodeWeights& w) const override { program_->validate_weights(w); }
  void validate_state(const NodeWeights& w, const State& state) const override { program_->validate_state(w, local_state(clock_, state)); }
};
}
StateClock kernel_clock(const std::shared_ptr<const StateKernel>& program) {
  auto wrapped = dynamic_cast<const ClockedKernel*>(program.get());
  return wrapped ? wrapped->clock() : StateClock{};
}
std::shared_ptr<const StateKernel> with_state_clock(std::shared_ptr<const StateKernel> program, const StateClock& clock) {
  clock.validate();
  if (!program) throw std::invalid_argument("missing state program for clock wrapper");
  if (auto wrapped = dynamic_cast<const ClockedKernel*>(program.get())) {
    if (wrapped->clock() != clock) throw std::invalid_argument("shared state program does not match state clock");
    return program;
  }
  return clock == StateClock{} ? program : std::make_shared<ClockedKernel>(std::move(program), clock);
}
}  // namespace tide
