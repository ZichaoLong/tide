#pragma once
#include "tide/packed.h"

namespace tide {
// A C++ client can provide its own immutable program in NodeWeights::kernel.
// Tensor state is functional; no program may mutate persistent state or weights.
// Exact batch/sequence contracts also require deterministic, independent step
// semantics: training replays step to preserve autograd connectivity.
class StateKernel {
 public:
  virtual ~StateKernel() = default;
  virtual State initial(const NodeWeights&) const = 0;
  virtual State step(const NodeWeights&, const State&, const ContentView&, Index time) const = 0;
  virtual std::vector<State> batch(const NodeWeights&, const std::vector<State>&, const Tensor&,
                                  const std::vector<Index>&, const ContentViews&) const;
  virtual std::vector<State> sequence(const NodeWeights&, const State&, const Tensor&,
                                     const std::vector<Index>&, const ContentViews&) const;
  virtual PackedStates packed_sequence(const NodeWeights&, const std::vector<State>&,
                                       const PackedSequence&) const;
  virtual bool exact_sequence() const { return false; }
  virtual bool joint_batch() const { return false; }
  virtual bool joint_sequence() const { return false; }
  // A configured nondefault packed policy uses the scalar oracle on causal
  // fallback/replay. Executors report that path instead of silently ignoring it.
  virtual bool scalar_policy_fallback() const { return false; }
  virtual State reset(const State&) const;
  virtual bool joint_reset_batch() const { return false; }
  virtual std::vector<State> reset_batch(const std::vector<State>&) const;
  virtual void validate_policy(const Node&, Index input_slots) const {}
  virtual void validate_weights(const NodeWeights&) const = 0;
  virtual void validate_state(const NodeWeights&, const State&) const = 0;
};
std::shared_ptr<const StateKernel> make_state_kernel(const std::string& name);
void configure_model(const Graph&, Model&);
Tensor affine_scan(Tensor a, Tensor b, const Tensor& initial);
}  // namespace tide
