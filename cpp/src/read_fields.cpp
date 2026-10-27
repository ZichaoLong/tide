// Core-5 structured Describe: docs/structured-records.md. No flattening of
// independent public roots, no reference recomputation in representation decoding.
#include "tide/read.h"
#include "tide/autograd.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide {
Tensor ReadKernel::step(const NodeWeights&, const ReadInput&) const {
  throw std::invalid_argument("Read kernel has no scalar implementation");
}
TensorFields ReadKernel::fields_step(const NodeWeights&, const ReadInput&) const {
  throw std::invalid_argument("Read kernel has no structured implementation");
}
std::vector<TensorFields> ReadKernel::fields_batch(const NodeWeights& w,
                                                 const std::vector<ReadInput>& requests) const {
  std::vector<TensorFields> result;
  for (const auto& request : requests) result.push_back(fields_step(w, request));
  return result;
}
namespace {
void validate(const TensorFields& fields, const FieldLayout& layout, at::Device device, at::ScalarType dtype) {
  if (fields.size() != layout.size()) throw std::invalid_argument("structured Read field domain mismatch");
  for (const auto& [name, shape] : layout) {
    if (name.empty() || !fields.count(name)) throw std::invalid_argument("structured Read field domain mismatch");
    const auto& value = fields.at(name);
    if (!value.defined() || value.sizes() != at::IntArrayRef(shape) || value.scalar_type() != dtype || value.device() != device)
      throw std::invalid_argument("structured Read field metadata mismatch");
    if (!at::isfinite(value).all().item<bool>()) throw std::invalid_argument("nonfinite structured Read field");
  }
}
}
void evaluate_read_fields(const NodeWeights& w, std::vector<Event>& events,
                         const std::vector<size_t>& ids, const std::vector<ReadInput>& requests, bool packed) {
  const auto& kernel = *w.read_kernel;
  const auto layout = kernel.field_layout(w);
  std::vector<TensorFields> values;
  if (packed) { at::NoGradGuard guard; values = kernel.fields_batch(w, requests); }
  else for (const auto& request : requests) values.push_back(kernel.fields_step(w, request));
  if (values.size() != ids.size()) throw std::invalid_argument("structured Read batch changed event count");
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto payload = requests[i].content.value;
    const auto device = kernel.descriptor_device(payload.device());
    const auto dtype = kernel.descriptor_dtype(payload.scalar_type());
    if (dtype != payload.scalar_type() && dtype != at::kDouble && dtype != at::kFloat)
      throw std::invalid_argument("invalid Read precision policy");
    validate(values[i], layout, device, dtype);
    if (packed && at::GradMode::is_enabled()) {
      const auto semantic = kernel.fields_step(w, requests[i]);
      validate(semantic, layout, device, dtype);
      for (auto& [name, value] : values[i]) value = semantic_value(value, semantic.at(name));
    }
    events[ids[i]].descriptor = Tensor();
    events[ids[i]].descriptor_fields = std::move(values[i]);
  }
}
}  // namespace tide
