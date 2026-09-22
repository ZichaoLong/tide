#include "checkpoint_internal.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace tide::checkpoint_detail {
void validate_value(const Tensor& value, const std::string& where) {
  if (!value.defined() || !value.device().is_cpu() || value.layout() != at::kStrided
      || (value.scalar_type() != at::kFloat && value.scalar_type() != at::kDouble))
    fail("unsupported tensor in " + where);
  if (!at::isfinite(value).all().item<bool>()) fail("nonfinite tensor in " + where);
}

void validate_group_options(const OptimizerGroup& group, const std::string& kind) {
  auto finite_nonnegative = [](double value, const char* name) {
    if (!std::isfinite(value) || value < 0) fail(std::string("invalid ") + name + " in optimizer group");
  };
  for (const auto value : {group.lr, group.weight_decay, group.momentum, group.dampening,
                          group.beta1, group.beta2, group.eps})
    if (!std::isfinite(value)) fail("nonfinite optimizer option");
  finite_nonnegative(group.lr, "learning rate"); finite_nonnegative(group.weight_decay, "weight decay");
  if (kind == "torch.optim.sgd.SGD") {
    finite_nonnegative(group.momentum, "momentum"); finite_nonnegative(group.dampening, "dampening");
    if (group.nesterov && (group.momentum <= 0 || group.dampening != 0)) fail("invalid Nesterov options");
  } else if (kind == "torch.optim.adamw.AdamW") {
    finite_nonnegative(group.eps, "epsilon");
    if (!std::isfinite(group.beta1) || group.beta1 < 0 || group.beta1 >= 1
        || !std::isfinite(group.beta2) || group.beta2 < 0 || group.beta2 >= 1)
      fail("invalid AdamW betas");
  } else fail("unsupported optimizer class");
}

bool same_group_names(const std::vector<OptimizerGroup>& a, const std::vector<OptimizerGroup>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) if (a[i].parameters != b[i].parameters) return false;
  return true;
}

void validate_state_tensor(const Tensor& saved, const Tensor& parameter, const std::string& where) {
  validate_value(saved, where);
  if (saved.scalar_type() != parameter.scalar_type() || saved.sizes() != parameter.sizes())
    fail("optimizer state shape/dtype mismatch in " + where);
}

std::map<std::string, Tensor> owner_values(const ParameterRegistry& registry) {
  std::map<std::string, Tensor> result;
  for (const auto& owner : registry.owners()) result.emplace(owner.canonical, owner.value);
  return result;
}

void validate_decoded(const Decoded& decoded, const ParameterRegistry& registry,
                      const NamedOptimizer* optimizer, const std::string& expected_identity) {
  if (!expected_identity.empty() && decoded.identity != expected_identity)
    fail("graph identity mismatch");
  std::vector<std::vector<std::string>> aliases;
  aliases.reserve(decoded.owners.size());
  std::set<std::string> owner_names;
  for (const auto& owner : decoded.owners) {
    if (owner.aliases.empty() || !std::is_sorted(owner.aliases.begin(), owner.aliases.end())
        || std::adjacent_find(owner.aliases.begin(), owner.aliases.end()) != owner.aliases.end())
      fail("invalid alias partition");
    for (const auto& name : owner.aliases) if (!owner_names.insert(name).second) fail("duplicate alias name");
    aliases.push_back(owner.aliases);
  }
  if (aliases != registry.alias_partitions()) fail("parameter alias topology mismatch");
  const auto values = owner_values(registry);
  if (values.size() != decoded.owners.size()) fail("parameter owner count mismatch");
  // Dense nonoverlapping destinations make copy_ predictable after preflight.
  // Distinct TensorImpls with overlapping storage are not declared aliases and
  // cannot have independent checkpoint values restored transactionally.
  std::map<uintptr_t, uintptr_t> storage_ranges;
  for (const auto& owner : decoded.owners) {
    const auto canonical = owner.aliases.front();
    const auto it = values.find(canonical);
    if (it == values.end()) fail("checkpoint has an unknown parameter owner");
    const auto& target = it->second;
    if (!target.device().is_cpu() || target.layout() != at::kStrided
        || !target.is_non_overlapping_and_dense())
      fail("unsupported destination layout/device: " + canonical);
    if (target.numel()) {
      const auto begin = reinterpret_cast<uintptr_t>(target.const_data_ptr());
      const auto end = begin + target.nbytes();
      const auto next = storage_ranges.lower_bound(begin);
      if ((next != storage_ranges.end() && next->first < end)
          || (next != storage_ranges.begin() && std::prev(next)->second > begin))
        fail("distinct parameter owners overlap storage");
      storage_ranges.emplace(begin, end);
    }
    validate_value(owner.value, "parameter " + canonical);
    if (owner.value.scalar_type() != it->second.scalar_type() || owner.value.sizes() != it->second.sizes())
      fail("parameter shape/dtype mismatch for " + canonical);
  }
  if (optimizer) {
    if (!decoded.has_optimizer) fail("checkpoint has no optimizer state");
    if (decoded.optimizer_class != optimizer->layout().class_name) fail("optimizer class mismatch");
    if (!same_group_names(decoded.groups, optimizer->groups())) fail("optimizer group ownership/order mismatch");
    for (const auto& group : optimizer->groups()) for (const auto& name : group.parameters)
      if (!values.count(name) || values.at(name).unsafeGetTensorImpl() != optimizer->registry().value(name).unsafeGetTensorImpl())
        fail("optimizer parameter is not owned by the supplied registry");
  }
  if (!decoded.has_optimizer) return;
  if (decoded.groups.empty()) fail("optimizer has no groups");
  for (const auto& group : decoded.groups) validate_group_options(group, decoded.optimizer_class);
  std::set<std::string> grouped;
  for (const auto& group : decoded.groups) {
    for (const auto& name : group.parameters) {
      if (!values.count(name)) fail("optimizer has an unknown parameter owner");
      if (!grouped.insert(name).second) fail("optimizer repeats a parameter owner");
    }
  }
  for (const auto& [name, state] : decoded.state) {
    if (!grouped.count(name) || state.step < 0) fail("optimizer state has an unowned parameter");
    const auto parameter = values.at(name);
    const auto group_it = std::find_if(decoded.groups.begin(), decoded.groups.end(), [&name](const auto& group) {
      return std::find(group.parameters.begin(), group.parameters.end(), name) != group.parameters.end();
    });
    if (decoded.optimizer_class == "torch.optim.sgd.SGD") {
      if (state.step != 0 || state.exp_avg.defined() || state.exp_avg_sq.defined() || state.max_exp_avg_sq.defined())
        fail("SGD state has Adam slots");
      if (group_it->momentum == 0 || !state.momentum_buffer.defined()) fail("invalid SGD momentum state");
      validate_state_tensor(state.momentum_buffer, parameter, "SGD " + name);
    } else {
      if (state.momentum_buffer.defined() || !state.exp_avg.defined() || !state.exp_avg_sq.defined())
        fail("AdamW state has invalid slots");
      validate_state_tensor(state.exp_avg, parameter, "AdamW " + name + ".exp_avg");
      validate_state_tensor(state.exp_avg_sq, parameter, "AdamW " + name + ".exp_avg_sq");
      if (group_it->amsgrad) {
        if (!state.max_exp_avg_sq.defined()) fail("AdamW AMSGrad state is incomplete");
        validate_state_tensor(state.max_exp_avg_sq, parameter, "AdamW " + name + ".max_exp_avg_sq");
      } else if (state.max_exp_avg_sq.defined()) fail("AdamW has unexpected AMSGrad state");
    }
  }
}

}  // namespace tide::checkpoint_detail
