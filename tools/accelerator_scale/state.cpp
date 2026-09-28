#include "placement.h"
#include <c10/core/DeviceGuard.h>

namespace accelerator_scale {
namespace {
class StateAdapter final : public StateKernel {
  std::shared_ptr<DeviceNode> node_;
 public:
  explicit StateAdapter(std::shared_ptr<DeviceNode> node) : node_(std::move(node)) {}
  State initial(const NodeWeights& host) const override { return node_->weights.kernel->initial(host); }
  bool joint_batch() const override { return node_->weights.kernel->joint_batch(); }
  State step(const NodeWeights& host, const State& old, const ContentView& value, Index time) const override {
    return execute({old}, value.value.unsqueeze(0), {time}, {value}, false).at(0);
  }
  std::vector<State> batch(const NodeWeights&, const std::vector<State>& old, const Tensor& values,
                           const std::vector<Index>& times, const ContentViews& views) const override {
    return execute(old, values, times, views, true);
  }
  std::vector<State> execute(const std::vector<State>& old, const Tensor& values,
                            const std::vector<Index>& times, const ContentViews& views, bool batched) const {
    std::lock_guard<std::mutex> lock(*node_->mutex);
    const auto& w = node_->weights; c10::DeviceGuard guard(w.bias.device());
    Transfer in(w.bias.device()); in.add(values);
    for (const auto& s : old) in.add(s);
    for (const auto& v : views) in.add(v);
    in.execute();
    std::vector<State> states; std::vector<ContentStorage> storage; ContentViews inputs;
    storage.reserve(views.size());
    for (const auto& s : old) states.push_back(in.get(s));
    for (const auto& v : views) storage.emplace_back(v, in);
    for (const auto& v : storage) inputs.push_back(v.view());
    auto result = batched ? w.kernel->batch(w, states, in.get(values), times, inputs)
                          : std::vector<State>{w.kernel->step(w, states.at(0), inputs.at(0), times.at(0))};
    Transfer out(at::Device(at::kCPU)); for (const auto& s : result) out.add(s); out.execute();
    for (auto& s : result) s = out.get(s);
    return result;
  }
  State reset(const State& value) const override { return node_->weights.kernel->reset(value); }
  bool joint_reset_batch() const override { return node_->weights.kernel->joint_reset_batch(); }
  std::vector<State> reset_batch(const std::vector<State>& states) const override {
    return node_->weights.kernel->reset_batch(states);
  }
  void validate_policy(const Node& node, Index slots) const override { node_->weights.kernel->validate_policy(node, slots); }
  void validate_weights(const NodeWeights&) const override { node_->weights.kernel->validate_weights(node_->weights); }
  void validate_state(const NodeWeights& host, const State& state) const override {
    node_->weights.kernel->validate_state(host, state);
  }
};
}
std::shared_ptr<const StateKernel> state_adapter(std::shared_ptr<DeviceNode> node) {
  return std::make_shared<StateAdapter>(std::move(node));
}
}  // namespace accelerator_scale
