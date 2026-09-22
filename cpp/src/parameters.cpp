#include "tide/parameters.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
namespace {
std::string qualified(const std::string& prefix, const std::string& name) {
  return prefix.empty() ? name : prefix + "." + name;
}

void add_if(ParameterRegistry& registry, const std::string& name, const Tensor& value,
            bool trainable_only) {
  if (!value.defined()) throw std::invalid_argument("cannot register an undefined parameter: " + name);
  if (!trainable_only || value.requires_grad()) registry.add(name, value);
}

template <typename Values>
void add_values(ParameterRegistry& registry, const std::string& prefix, const Values& values,
                bool trainable_only) {
  for (size_t index = 0; index < values.size(); ++index)
    add_if(registry, qualified(prefix, std::to_string(index)), values[index], trainable_only);
}
}  // namespace

void ParameterRegistry::add(const std::string& name, const Tensor& value) {
  if (name.empty()) throw std::invalid_argument("parameter name must not be empty");
  if (name.front() == '.' || name.back() == '.' || name.find("..") != std::string::npos)
    throw std::invalid_argument("parameter name has an empty path component");
  if (!value.defined()) throw std::invalid_argument("cannot register an undefined parameter: " + name);
  if (named_.count(name)) throw std::invalid_argument("duplicate parameter name: " + name);
  named_.emplace(name, value);
}

void ParameterRegistry::add_model(const Model& model, const std::string& prefix,
                                  bool trainable_only) {
  for (size_t node = 0; node < model.nodes.size(); ++node) {
    const auto base = qualified(prefix, "nodes." + std::to_string(node));
    const auto& weights = model.nodes[node];
    add_if(*this, base + ".decay", weights.decay, trainable_only);
    add_if(*this, base + ".weight", weights.weight, trainable_only);
    add_if(*this, base + ".bias", weights.bias, trainable_only);
    add_if(*this, base + ".read", weights.read, trainable_only);
    for (const auto& [name, value] : weights.extra)
      add_if(*this, base + ".extra." + name, value, trainable_only);
  }
  for (size_t region = 0; region < model.regions.size(); ++region) {
    // RegionWeights::extra is a native transport map; its keys already carry
    // the Python RegionProgram's local parameter names.
    const auto base = qualified(prefix, "regions." + std::to_string(region));
    for (const auto& [name, value] : model.regions[region].extra)
      add_if(*this, base + "." + name, value, trainable_only);
  }
  add_values(*this, qualified(prefix, "input_scale"), model.input_scale, trainable_only);
  add_values(*this, qualified(prefix, "agg_scale"), model.agg_scale, trainable_only);
  add_values(*this, qualified(prefix, "edge_scale"), model.edge_scale, trainable_only);
  add_values(*this, qualified(prefix, "output_scale"), model.output_scale, trainable_only);
}

bool ParameterRegistry::contains(const std::string& name) const { return named_.count(name) != 0; }

Tensor ParameterRegistry::value(const std::string& name) const {
  const auto it = named_.find(name);
  if (it == named_.end()) throw std::invalid_argument("unknown parameter: " + name);
  return it->second;
}

std::string ParameterRegistry::canonical_name(const std::string& name) const {
  const auto parameter = value(name);
  std::string canonical;
  const auto* identity = parameter.unsafeGetTensorImpl();
  for (const auto& [candidate, value] : named_)
    if (value.unsafeGetTensorImpl() == identity && (canonical.empty() || candidate < canonical)) canonical = candidate;
  if (canonical.empty()) throw std::logic_error("parameter registry lost a registered owner");
  return canonical;
}

std::vector<std::string> ParameterRegistry::names() const {
  std::vector<std::string> result;
  result.reserve(named_.size());
  for (const auto& [name, value] : named_) result.push_back(name);
  return result;
}

std::vector<ParameterOwner> ParameterRegistry::owners() const {
  std::map<const void*, ParameterOwner> grouped;
  for (const auto& [name, value] : named_) {
    const auto key = static_cast<const void*>(value.unsafeGetTensorImpl());
    auto& owner = grouped[key];
    owner.aliases.push_back(name);
    if (owner.canonical.empty() || name < owner.canonical) {
      owner.canonical = name;
      owner.value = value;
    }
  }
  std::vector<ParameterOwner> result;
  result.reserve(grouped.size());
  for (auto& [key, owner] : grouped) {
    std::sort(owner.aliases.begin(), owner.aliases.end());
    result.push_back(std::move(owner));
  }
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return a.canonical < b.canonical;
  });
  return result;
}

std::vector<std::vector<std::string>> ParameterRegistry::alias_partitions() const {
  std::vector<std::vector<std::string>> result;
  for (const auto& owner : owners()) result.push_back(owner.aliases);
  return result;
}

ParameterRegistry Model::parameters(bool trainable_only) const {
  ParameterRegistry result;
  result.add_model(*this, "", trainable_only);
  return result;
}
}  // namespace tide
