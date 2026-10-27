#include "tide/tensor_fields.h"
#include "tide/autograd.h"
#include <limits>
#include <set>
#include <stdexcept>

namespace tide {
TensorFields PackedTensorFields::decode() const {
  std::vector<Index> offsets(buffers.size(), 0);
  TensorFields result;
  for (const auto& field : layout) {
    if (field.name.empty() || result.count(field.name) || field.bank < 0 ||
        field.bank >= static_cast<Index>(buffers.size()) || field.offset != offsets[field.bank])
      throw std::invalid_argument("invalid packed field layout");
    const auto& buffer = buffers[field.bank];
    Index count = 1;
    for (auto n : field.shape) {
      if (n < 0 || (n && count > std::numeric_limits<Index>::max()/n))
        throw std::invalid_argument("invalid packed field shape");
      count *= n;
    }
    if (!buffer.defined() || buffer.dim() != 1 || buffer.scalar_type() != field.dtype ||
        field.offset > buffer.numel() || count > buffer.numel()-field.offset)
      throw std::invalid_argument("packed field dtype/shape mismatch");
    auto value = buffer.narrow(0, field.offset, count).reshape(field.shape);
    if (roots) {
      if (!roots->count(field.name)) throw std::invalid_argument("packed field root domain mismatch");
      value = semantic_value(value, roots->at(field.name));
    }
    result.emplace(field.name, value); offsets[field.bank] += count;
  }
  for (size_t i = 0; i < buffers.size(); ++i)
    if (buffers[i].dim() != 1 || offsets[i] != buffers[i].numel())
      throw std::invalid_argument("packed field buffer domain mismatch");
  if (roots && roots->size() != result.size()) throw std::invalid_argument("packed field root domain mismatch");
  return result;
}

PackedTensorFields pack_fields(const TensorFields& fields) {
  if (fields.empty()) throw std::invalid_argument("packing requires nonempty named fields");
  PackedTensorFields result;
  using Key = std::tuple<int, int, int>;
  std::map<Key, Index> banks;
  std::vector<std::vector<Tensor>> rows;
  std::vector<Index> offsets;
  for (const auto& [name, field] : fields) {
    if (name.empty() || !field.defined()) throw std::invalid_argument("invalid named tensor field");
    Key key{int(field.device().type()), field.device().index(), int(field.scalar_type())};
    if (!banks.count(key)) {
      banks.emplace(key, banks.size()); rows.emplace_back(); offsets.push_back(0);
    }
    const auto bank = banks.at(key);
    result.layout.push_back({name, field.sizes().vec(), field.scalar_type(), bank, offsets[bank]});
    rows[bank].push_back(field.detach().reshape({-1})); offsets[bank] += field.numel();
  }
  for (const auto& group : rows) result.buffers.push_back(at::cat(group));
  result.roots = fields;
  return result;
}
}  // namespace tide
