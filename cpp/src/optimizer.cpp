#include "tide/optimizer.h"
#include <torch/csrc/autograd/grad_mode.h>
#include <cmath>
#include <set>
#include <stdexcept>

namespace tide {
namespace {
void finite_nonnegative(double value, const char* name) {
  if (!std::isfinite(value) || value < 0) throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
}

void validate_tensor(const Tensor& parameter, const Tensor& gradient, const std::string& name) {
  if (!parameter.device().is_cpu() || (parameter.scalar_type() != at::kFloat && parameter.scalar_type() != at::kDouble))
    throw std::invalid_argument("optimizer requires CPU FP32/FP64 parameter: " + name);
  if (gradient.device() != parameter.device() || gradient.scalar_type() != parameter.scalar_type()
      || gradient.sizes() != parameter.sizes())
    throw std::invalid_argument("gradient shape/device/dtype mismatch for parameter: " + name);
}

void validate_sgd(const OptimizerGroup& group) {
  finite_nonnegative(group.lr, "learning rate");
  finite_nonnegative(group.weight_decay, "weight decay");
  finite_nonnegative(group.momentum, "momentum");
  finite_nonnegative(group.dampening, "dampening");
  if (group.nesterov && (group.momentum <= 0 || group.dampening != 0))
    throw std::invalid_argument("Nesterov SGD requires momentum > 0 and dampening == 0");
}

void validate_adamw(const OptimizerGroup& group) {
  finite_nonnegative(group.lr, "learning rate");
  finite_nonnegative(group.weight_decay, "weight decay");
  finite_nonnegative(group.eps, "epsilon");
  if (!std::isfinite(group.beta1) || group.beta1 < 0 || group.beta1 >= 1
      || !std::isfinite(group.beta2) || group.beta2 < 0 || group.beta2 >= 1)
    throw std::invalid_argument("AdamW betas must be finite in [0, 1)");
}
}  // namespace

NamedOptimizer::NamedOptimizer(ParameterRegistry& registry, std::vector<OptimizerGroup> groups,
                               std::string class_name)
    : registry_(registry), groups_(std::move(groups)), class_name_(std::move(class_name)) {
  if (groups_.empty()) {
    OptimizerGroup group;
    for (const auto& owner : registry_.owners()) group.parameters.push_back(owner.canonical);
    groups_.push_back(std::move(group));
  }
  std::set<std::string> seen;
  for (auto& group : groups_) {
    std::vector<std::string> canonical;
    canonical.reserve(group.parameters.size());
    for (const auto& name : group.parameters) {
      const auto owner = registry_.canonical_name(name);
      if (!seen.insert(owner).second) throw std::invalid_argument("optimizer repeats a shared parameter: " + owner);
      canonical.push_back(owner);
    }
    group.parameters = std::move(canonical);
  }
}

void NamedOptimizer::step() {
  at::NoGradGuard guard;
  for (const auto& group : groups_) for (const auto& name : group.parameters) {
    const auto parameter = registry_.value(name);
    const auto gradient = parameter.grad();
    // None is structural absence.  In particular, decoupled AdamW decay must
    // not touch a parameter that did not participate in this objective.
    if (!gradient.defined()) continue;
    validate_tensor(parameter, gradient, name);
    update(name, parameter, gradient, group);
  }
}

void NamedOptimizer::zero_grad(bool set_to_none) {
  at::NoGradGuard guard;
  for (const auto& group : groups_) for (const auto& name : group.parameters) {
    auto parameter = registry_.value(name);
    if (!parameter.grad().defined()) continue;
    if (set_to_none) parameter.mutable_grad().reset();
    else parameter.grad().zero_();
  }
}

OptimizerLayout NamedOptimizer::layout() const {
  OptimizerLayout result{class_name_, {}};
  for (const auto& group : groups_) result.groups.push_back(group.parameters);
  return result;
}

SGD::SGD(ParameterRegistry& registry, std::vector<OptimizerGroup> groups)
    : NamedOptimizer(registry, std::move(groups), "torch.optim.sgd.SGD") {
  for (const auto& group : this->groups()) validate_sgd(group);
}

void SGD::update(const std::string& name, const Tensor& parameter, const Tensor& gradient,
                const OptimizerGroup& group) {
  Tensor direction = group.maximize ? -gradient : gradient;
  if (group.weight_decay != 0) direction = direction.add(parameter, group.weight_decay);
  if (group.momentum != 0) {
    auto& slot = state_[name].momentum_buffer;
    if (!slot.defined()) slot = direction.clone();
    else slot.mul_(group.momentum).add_(direction, 1 - group.dampening);
    direction = group.nesterov ? direction.add(slot, group.momentum) : slot;
  }
  parameter.add_(direction, -group.lr);
}

AdamW::AdamW(ParameterRegistry& registry, std::vector<OptimizerGroup> groups)
    : NamedOptimizer(registry, std::move(groups), "torch.optim.adamw.AdamW") {
  for (const auto& group : this->groups()) validate_adamw(group);
}

void AdamW::update(const std::string& name, const Tensor& parameter, const Tensor& gradient,
                   const OptimizerGroup& group) {
  auto& slot = state_[name];
  if (!slot.exp_avg.defined()) {
    slot.exp_avg = at::zeros_like(parameter);
    slot.exp_avg_sq = at::zeros_like(parameter);
    if (group.amsgrad) slot.max_exp_avg_sq = at::zeros_like(parameter);
  } else if (group.amsgrad && !slot.max_exp_avg_sq.defined()) {
    throw std::invalid_argument("AdamW AMSGrad state mismatch for parameter: " + name);
  }
  ++slot.step;
  parameter.mul_(1 - group.lr * group.weight_decay);
  const auto direction = group.maximize ? -gradient : gradient;
  slot.exp_avg.mul_(group.beta1).add_(direction, 1 - group.beta1);
  slot.exp_avg_sq.mul_(group.beta2).addcmul_(direction, direction, 1 - group.beta2);
  Tensor second = slot.exp_avg_sq;
  if (group.amsgrad) {
    slot.max_exp_avg_sq.copy_(at::maximum(slot.max_exp_avg_sq, slot.exp_avg_sq));
    second = slot.max_exp_avg_sq;
  }
  const auto correction1 = 1 - std::pow(group.beta1, static_cast<double>(slot.step));
  const auto correction2 = 1 - std::pow(group.beta2, static_cast<double>(slot.step));
  auto denominator = at::sqrt(second).div_(std::sqrt(correction2)).add_(group.eps);
  parameter.addcdiv_(slot.exp_avg, denominator, -group.lr / correction1);
}
}  // namespace tide
