#pragma once
#include "tide/types.h"

namespace tide {
// The comparison pointer is borrowed only for this synchronous program call.
// Programs are immutable and must not mutate comparison, content or weights.
struct FullInput { const State* comparison; Index time; ContentView content; Tensor control; };
struct FullResult { Tensor value; std::vector<SlotValue> emitted; };
class FullKernel {
 public:
  virtual ~FullKernel() = default;
  virtual FullResult step(const NodeWeights&, const FullInput&, Index slots, const Options&) const = 0;
  virtual std::vector<FullResult> batch(const NodeWeights&, const std::vector<FullInput>&, Index slots, const Options&) const;
  virtual bool joint_batch() const { return false; }
  virtual void validate_weights(const NodeWeights&, Index slots) const = 0;
};
std::shared_ptr<const FullKernel> make_full_kernel(const Node&);
void evaluate_full(const Graph&, const Model&, std::vector<Event>&, const std::vector<size_t>&,
                   const Options&, bool packed);
}  // namespace tide
