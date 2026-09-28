#include "scoring.h"
#include <tide/read.h>
#include <stdexcept>

namespace accelerator_scale {
void Scoring::validate(at::Device device, bool resident) const {
  if ((read_device != "cpu" && read_device != "model")
      || (control_device != "cpu" && control_device != "model")
      || (dtype != at::kFloat && dtype != at::kDouble))
    throw std::invalid_argument("invalid Read/control device or Read dtype");
  if (!resident && (read_device != "cpu" || control_device != "cpu"))
    throw std::invalid_argument("model-device Read/controls require resident transport");
  if (!device.is_cpu() && dtype == at::kDouble
      && (read_device == "model" || control_device == "model"))
    throw std::invalid_argument("NPU Read/controls require --read-dtype float32; no FP64 fallback");
}
namespace {
class NormFP32 final : public tide::ReadKernel {
 public:
  tide::Tensor step(const tide::NodeWeights&, const tide::ReadInput& r) const override {
    return at::norm(r.state ? r.state->value : r.content.value, 2, {-1}, false, at::kFloat);
  }
  std::vector<tide::Tensor> batch(const tide::NodeWeights&, const std::vector<tide::ReadInput>& requests) const override {
    std::vector<tide::Tensor> values;
    for (const auto& r : requests) values.push_back(r.state ? r.state->value : r.content.value);
    return at::norm(at::stack(values), 2, {-1}, false, at::kFloat).unbind();
  }
  bool joint_batch() const override { return true; }
  void validate_weights(const tide::NodeWeights&) const override {}
};
}
void configure_scoring(pdg_scale::Fixture& f, const Scoring& scoring) {
  if (scoring.dtype == at::kDouble) return;
  for (size_t i = 0; i < f.graph.nodes.size(); ++i) {
    auto& node = f.graph.nodes[i];
    if (node.readout == "norm-fp64-v1") {
      // A distinct profile keeps precision visible in graph identity. The
      // standalone client supplies it through the public custom Read interface.
      node.readout = "scale-norm-fp32-v1";
      f.model.nodes[i].read_kernel = std::make_shared<NormFP32>();
    }
  }
}
}  // namespace accelerator_scale
