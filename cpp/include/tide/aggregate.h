#pragma once
#include "tide/types.h"

namespace tide {
// Atom metadata is copied to allow graph-owned source-origin views; Tensor
// storage is shared read-only. Empty fibers never call Aggregate.
struct SourceInput { Index slot; Atom atom; Tensor scale; };
struct AggregateInput { Index time, slots; std::vector<SourceInput> sources; };
struct AggregateResult { Tensor value; std::vector<SlotValue> contributions; };
class AggregateKernel {
 public:
  virtual ~AggregateKernel() = default;
  virtual AggregateResult step(const NodeWeights&, const AggregateInput&) const = 0;
  virtual std::vector<AggregateResult> batch(const NodeWeights&, const std::vector<AggregateInput>&) const;
  virtual bool joint_batch() const { return false; }
  virtual void validate_weights(const NodeWeights&, Index slots) const = 0;
};
std::shared_ptr<const AggregateKernel> make_aggregate_kernel(const Node&);
void evaluate_aggregate(const Graph&, const Model&, std::vector<Event>&, const std::vector<size_t>&, bool packed);
}  // namespace tide
