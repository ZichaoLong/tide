#pragma once
#include "tide/types.h"

namespace tide {
// Only the mode-appropriate state is visible; content mode has nullptr.
// Pointers and content views are borrowed for the synchronous call.
struct ReadInput { const State* state; Index time; ContentView content; };
ReadInput read_input(const std::string& mode, const State& old, const State& proposal,
                     Index time, const ContentView&);
class ReadKernel {
 public:
  virtual ~ReadKernel() = default;
  virtual Tensor step(const NodeWeights&, const ReadInput&) const = 0;
  virtual std::vector<Tensor> batch(const NodeWeights&, const std::vector<ReadInput>&) const;
  virtual bool joint_batch() const { return false; }
  virtual void validate_weights(const NodeWeights&) const = 0;
};
std::shared_ptr<const ReadKernel> make_read_kernel(const Node&);
void evaluate_read(const Graph&, const Model&, std::vector<Event>&, const std::vector<size_t>&, bool packed);
}  // namespace tide
