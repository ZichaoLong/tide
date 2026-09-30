#include "placement_internal.h"
#include <stdexcept>

namespace tide::placement_detail {
namespace {
class PlacedRead final : public ReadKernel {
 public:
  PlacedRead(Node node, at::Device device, at::ScalarType dtype)
      : node_(std::move(node)), device_(device), dtype_(dtype) {}
  Tensor step(const NodeWeights& w, const ReadInput& r) const override {
    if (node_.identity) return at::zeros({}, r.content.value.options().device(device_).dtype(dtype_));
    return calculate(w, r.state ? r.state->value : r.content.value);
  }
  std::vector<Tensor> batch(const NodeWeights& w, const std::vector<ReadInput>& requests) const override {
    if (node_.identity) return ReadKernel::batch(w, requests);
    if (requests.empty()) return {};
    std::vector<Tensor> values;
    for (const auto& r : requests) values.push_back(r.state ? r.state->value : r.content.value);
    // Pack on the payload device, then one transfer per legal node-time batch.
    return calculate(w, at::stack(values)).unbind();
  }
  bool joint_batch() const override { return true; }
  at::ScalarType descriptor_dtype(at::ScalarType) const override { return dtype_; }
  at::Device descriptor_device(at::Device) const override { return device_; }
  void validate_weights(const NodeWeights& w) const override { make_read_kernel(node_)->validate_weights(w); }
  bool matches(const Node& n) const { return n.readout == node_.readout && n.identity == node_.identity; }
 private:
  Tensor calculate(const NodeWeights& w, const Tensor& value) const {
    auto x = value.to(value.options().device(device_).dtype(dtype_));
    if (node_.readout == "linear-v1") return (x * w.read.to(x.options())).sum(-1);
    return at::norm(x, 2, {-1}, false, dtype_);
  }
  Node node_;
  at::Device device_;
  at::ScalarType dtype_;
};
}
std::shared_ptr<const ReadKernel> read_kernel(const Node& node, const NodeWeights& w, const ResolvedPlacement& p) {
  const auto previous = dynamic_cast<const PlacedRead*>(w.read_kernel.get());
  if (w.read_kernel && !(previous ? previous->matches(node) : builtin_read(*w.read_kernel, node)))
    throw std::invalid_argument("placement cannot replace a custom or mismatched Read kernel");
  const auto base = make_read_kernel(node);
  auto dtype = base->descriptor_dtype(w.bias.scalar_type());
  if (p.scoring_dtype != "profile") {
    const auto explicit_dtype = p.scoring_dtype == "payload" ? w.bias.scalar_type() :
                                p.scoring_dtype == "float32" ? at::kFloat : at::kDouble;
    if (node.readout != "linear-v1" && dtype != explicit_dtype)
      throw std::invalid_argument("scoring precision conflicts with the named norm Read contract");
    dtype = explicit_dtype;
  }
  if (dtype == at::kDouble && (p.read.type() == c10::DeviceType::PrivateUse1 ||
      p.control.type() == c10::DeviceType::PrivateUse1 || p.selection.type() == c10::DeviceType::PrivateUse1))
    throw std::invalid_argument("NPU Read/control/selection cannot consume FP64 descriptors; select CPU or explicit FP32 Read");
  return std::make_shared<PlacedRead>(node, p.read, dtype);
}
}  // namespace tide::placement_detail
