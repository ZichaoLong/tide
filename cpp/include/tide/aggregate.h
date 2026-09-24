#pragma once
#include "tide/types.h"

namespace tide {
// Atom metadata is copied to allow graph-owned source-origin views; Tensor
// storage is shared read-only. Empty fibers never call Aggregate.
struct AggregateInput { Index time, slots; std::vector<SourceInput> sources; };
struct AggregateResult { Tensor value; std::vector<SlotValue> contributions; };
struct AggregateBatch {
  std::vector<AggregateResult> events;
  Tensor contents;
  std::shared_ptr<const SourceBatch> sources;
};
class AggregateKernel {
 public:
  virtual ~AggregateKernel() = default;
  virtual AggregateResult step(const NodeWeights&, const AggregateInput&) const = 0;
  virtual std::vector<AggregateResult> batch(const NodeWeights&, const std::vector<AggregateInput>&) const;
  virtual bool joint_batch() const { return false; }
  virtual bool joint_sources() const { return false; }
  virtual bool batched_autograd() const { return false; }
  virtual std::vector<AggregateResult> batch_grad(const NodeWeights&, const std::vector<AggregateInput>&) const;
  virtual AggregateBatch source_batch(const NodeWeights& w, const std::vector<AggregateInput>& r) const {
    return {batch(w, r), {}, {}};
  }
  virtual void validate_weights(const NodeWeights&, Index slots) const = 0;
};
std::shared_ptr<const AggregateKernel> make_aggregate_kernel(const Node&);
Tensor evaluate_aggregate(const Graph&, const Model&, std::vector<Event>&, const std::vector<size_t>&,
                          bool packed, bool packed_sources = false, const std::string& autograd = "replay");
void validate_aggregate_autograd(const Model&, const Options&);
}  // namespace tide
