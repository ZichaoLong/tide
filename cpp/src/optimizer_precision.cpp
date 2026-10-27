// Explicit half owners, FP32 updates, and a static loss-scale boundary.
#include "tide/optimizer.h"
#include "tide/device.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <stdexcept>
namespace tide {
void NamedOptimizer::initialize_precision() {
  if (!std::isfinite(precision_.loss_scale) || precision_.loss_scale <= 0 ||
      (!precision_.fp32_masters && precision_.loss_scale != 1.))
    throw std::invalid_argument("invalid explicit optimizer loss scale");
  for (const auto& group : groups_) for (const auto& name : group.parameters) {
    const auto p = registry_.value(name);
    if (precision_.fp32_masters) {
      if (p.scalar_type() != at::kHalf || !p.requires_grad() || !p.is_leaf() || !supported_kernel_payload(p))
        throw std::invalid_argument("FP32 masters require supported trainable FP16 leaf parameters");
      masters_[name] = p.detach().to(at::kFloat).clone();
    } else if (p.scalar_type() == at::kHalf) {
      throw std::invalid_argument("FP16 optimizer requires explicit FP32 masters");
    }
  }
}

void NamedOptimizer::backward(const Tensor& loss, bool retain_graph) {
  if (!loss.defined() || loss.numel() != 1 || !loss.is_floating_point() ||
      (precision_.fp32_masters && loss.scalar_type() != at::kFloat))
    throw std::invalid_argument("master backward requires scalar FP32 loss");
  torch::autograd::backward({loss * precision_.loss_scale}, {}, retain_graph);
  scaled_backward_ = true;
}

void NamedOptimizer::master_step() {
  std::map<std::string, Tensor> gradients;
  for (const auto& group : groups_) for (const auto& name : group.parameters) {
    auto p = registry_.value(name), g = p.grad();
    if (!g.defined()) continue;  // No decay or optimizer state for absent roots.
    if (!scaled_backward_)
      throw std::invalid_argument("call optimizer.backward to declare gradient scaling");
    if (g.sizes() != p.sizes() || g.scalar_type() != p.scalar_type() || g.device() != p.device())
      throw std::invalid_argument("master gradient shape/device/dtype mismatch");
    auto unscaled = g.to(at::kFloat) / precision_.loss_scale;
    if (!at::isfinite(unscaled).all().item<bool>())
      throw std::invalid_argument("nonfinite master gradient; no update performed");
    gradients.emplace(name, unscaled);
  }
  // All gradients are checked before any state/master mutation. Representable
  // payloads are checked after updates and before any payload is published.
  std::map<std::string, Tensor> rounded;
  for (const auto& group : groups_) for (const auto& name : group.parameters) {
    const auto it = gradients.find(name);
    if (it == gradients.end()) continue;
    update(name, masters_.at(name), it->second, group);
    auto value = masters_.at(name).to(at::kHalf);
    if (!at::isfinite(value).all().item<bool>())
      throw std::invalid_argument("FP16 update overflow; restore checkpoint before continuing");
    rounded.emplace(name, value);
  }
  for (const auto& [name, value] : rounded) registry_.value(name).copy_(value);
  scaled_backward_ = false;
}
}
