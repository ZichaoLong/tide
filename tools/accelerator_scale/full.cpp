#include "placement.h"
#include <c10/core/DeviceGuard.h>

namespace accelerator_scale {
namespace {
class FullAdapter final : public FullKernel {
  std::shared_ptr<DeviceNode> node_;
 public:
  explicit FullAdapter(std::shared_ptr<DeviceNode> node) : node_(std::move(node)) {}
  bool joint_batch() const override { return node_->weights.full_kernel->joint_batch(); }
  bool batched_autograd() const override { return node_->weights.full_kernel->batched_autograd(); }
  FullResult step(const NodeWeights&, const FullInput& input, Index slots, const Options& options) const override {
    return execute({input}, slots, options, 0).at(0);
  }
  std::vector<FullResult> batch(const NodeWeights&, const std::vector<FullInput>& inputs,
                               Index slots, const Options& options) const override {
    return execute(inputs, slots, options, 1);
  }
  std::vector<FullResult> batch_grad(const NodeWeights&, const std::vector<FullInput>& inputs,
                                    Index slots, const Options& options) const override {
    return execute(inputs, slots, options, 2);
  }
  std::vector<FullResult> execute(const std::vector<FullInput>& inputs, Index slots,
                                 const Options& options, int mode) const {
    std::lock_guard<std::mutex> lock(*node_->mutex);
    const auto& w = node_->weights; c10::DeviceGuard guard(w.bias.device());
    Transfer in(w.bias.device());
    for (const auto& input : inputs) { in.add(*input.comparison); in.add(input.content); in.add(input.control); }
    in.execute();
    std::vector<State> states; std::vector<ContentStorage> storage; std::vector<FullInput> requests;
    states.reserve(inputs.size()); storage.reserve(inputs.size());
    for (const auto& input : inputs) { states.push_back(in.get(*input.comparison)); storage.emplace_back(input.content, in); }
    for (size_t i = 0; i < inputs.size(); ++i)
      requests.push_back({&states[i], inputs[i].time, storage[i].view(), in.get(inputs[i].control)});
    auto result = mode == 2 ? w.full_kernel->batch_grad(w, requests, slots, options)
                : mode == 1 ? w.full_kernel->batch(w, requests, slots, options)
                            : std::vector<FullResult>{w.full_kernel->step(w, requests.at(0), slots, options)};
    Transfer out(at::Device(at::kCPU));
    for (const auto& r : result) { out.add(r.value); for (const auto& s : r.emitted) out.add(s.value); }
    out.execute();
    for (auto& r : result) { r.value = out.get(r.value); for (auto& s : r.emitted) s.value = out.get(s.value); }
    return result;
  }
  void validate_weights(const NodeWeights&, Index slots) const override {
    node_->weights.full_kernel->validate_weights(node_->weights, slots);
  }
};
}
std::shared_ptr<const FullKernel> full_adapter(std::shared_ptr<DeviceNode> node) {
  return std::make_shared<FullAdapter>(std::move(node));
}
}  // namespace accelerator_scale
