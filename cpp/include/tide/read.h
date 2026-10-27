#pragma once
#include "tide/types.h"

namespace tide {
using TensorFields = std::map<std::string, Tensor>;
using FieldLayout = std::map<std::string, std::vector<Index>>;
// Only the mode-appropriate state is visible; content mode has nullptr.
// Pointers and content views are borrowed for the synchronous call.
struct ReadInput { const State* state; Index time; ContentView content; };
ReadInput read_input(const std::string& mode, const State& old, const State& proposal,
                     Index time, const ContentView&);
class ReadKernel {
 public:
  virtual ~ReadKernel() = default;
  virtual Tensor step(const NodeWeights&, const ReadInput&) const;
  // Core-5 Describe finite product, preserving independent field VJP roots.
  virtual bool structured() const { return false; }
  virtual FieldLayout field_layout(const NodeWeights&) const { return {}; }
  virtual TensorFields fields_step(const NodeWeights&, const ReadInput&) const;
  virtual std::vector<TensorFields> fields_batch(const NodeWeights&, const std::vector<ReadInput>&) const;
  virtual std::vector<Tensor> batch(const NodeWeights&, const std::vector<ReadInput>&) const;
  virtual bool joint_batch() const { return false; }
  virtual bool batched_autograd() const { return false; }
  virtual std::vector<Tensor> batch_grad(const NodeWeights&, const std::vector<ReadInput>&) const;
  virtual at::ScalarType descriptor_dtype(at::ScalarType payload) const { return payload; }
  virtual at::Device descriptor_device(at::Device payload) const { return payload; }
  virtual void validate_weights(const NodeWeights&) const = 0;
};
std::shared_ptr<const ReadKernel> make_read_kernel(const Node&);
void evaluate_read(const Graph&, const Model&, std::vector<Event>&, const std::vector<size_t>&, bool packed);
void evaluate_read_fields(const NodeWeights&, std::vector<Event>&, const std::vector<size_t>&,
                           const std::vector<ReadInput>&, bool packed);
}  // namespace tide
