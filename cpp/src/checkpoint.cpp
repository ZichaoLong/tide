#include "checkpoint_internal.h"
#include <torch/csrc/autograd/grad_mode.h>

namespace tide {
using namespace checkpoint_detail;

void Checkpoint::save(const std::filesystem::path& path, const ParameterRegistry& registry,
                      const NamedOptimizer* optimizer, const std::string& identity) {
  Decoded record;
  record.identity = identity;
  for (const auto& owner : registry.owners()) record.owners.push_back({owner.aliases, owner.value});
  if (optimizer) {
    record.has_optimizer = true;
    record.optimizer_class = optimizer->layout().class_name;
    record.groups = optimizer->groups();
    record.state = optimizer->state();
  }
  validate_decoded(record, registry, optimizer, identity);
  publish_exclusive(path, encode(record));
}

void Checkpoint::load(const std::filesystem::path& path, ParameterRegistry& registry,
                      NamedOptimizer* optimizer, const std::string& expected_identity) {
  auto decoded = decode(read_all(path));
  validate_decoded(decoded, registry, optimizer, expected_identity);
  // All allocation, metadata, destination layout and optimizer checks precede
  // mutation. Owners retain their TensorImpl, requires_grad and grad buffers.
  const auto values = owner_values(registry);
  at::NoGradGuard guard;
  for (const auto& owner : decoded.owners) values.at(owner.aliases.front()).copy_(owner.value);
  if (optimizer) {
    optimizer->groups_.swap(decoded.groups);
    optimizer->state_.swap(decoded.state);
  }
}
}  // namespace tide
