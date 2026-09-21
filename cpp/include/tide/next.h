#pragma once
#include "tide/types.h"

namespace tide {
// All inputs are read-only and borrowed during this call. Full is not an input.
struct NextInput {
  const State& old;
  const State& comparison;
  Index time;
  ContentView content;
  bool active;
  Tensor control;
};
class NextKernel {
 public:
  virtual ~NextKernel() = default;
  virtual State step(const NodeWeights&, const NextInput&) const = 0;
  virtual bool comparison_identity() const { return false; }
  virtual void validate_weights(const NodeWeights&) const = 0;
};
std::shared_ptr<const NextKernel> make_next_kernel(const Node&);
State evaluate_next(const Node&, const NodeWeights&, const NextInput&);
}  // namespace tide
