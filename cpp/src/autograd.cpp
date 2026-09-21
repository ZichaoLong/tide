#include "tide/autograd.h"
#include <torch/csrc/autograd/custom_function.h>
#include <stdexcept>

namespace tide {
namespace {
class SemanticValue : public torch::autograd::Function<SemanticValue> {
 public:
  static Tensor forward(torch::autograd::AutogradContext*, Tensor reference, Tensor value) { return value.clone(); }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext*, torch::autograd::variable_list grad) {
    return {grad[0], Tensor()};
  }
};
}  // namespace
Tensor semantic_value(const Tensor& packed, const Tensor& reference) {
  if (packed.sizes() != reference.sizes() || packed.scalar_type() != reference.scalar_type()
      || packed.device() != reference.device()) throw std::invalid_argument("batching contract changed tensor metadata");
  if (!reference.requires_grad()) return packed.detach();
  return SemanticValue::apply(reference, packed.detach());
}
State semantic_state(const State& packed, const State& reference) {
  if (packed.last_time != reference.last_time || packed.observations != reference.observations
      || packed.slots.size() != reference.slots.size()) throw std::invalid_argument("state batching contract changed metadata");
  auto result = packed; result.value = semantic_value(packed.value, reference.value);
  for (auto& [name, tensor] : result.slots) {
    if (!reference.slots.count(name)) throw std::invalid_argument("state batching contract changed slots");
    tensor = semantic_value(tensor, reference.slots.at(name));
  }
  return result;
}
}  // namespace tide
