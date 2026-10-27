#pragma once
#include "tide/read.h"

namespace tide {
// Core-5 product representation; see docs/structured-records.md.
struct PackedField {
  std::string name;
  std::vector<Index> shape;
  at::ScalarType dtype;
  Index bank, offset;
};
struct PackedTensorFields {
  std::vector<PackedField> layout;
  std::vector<Tensor> buffers;
  // Actually computed field tensors, not a recursive reference reconstruction.
  std::optional<TensorFields> roots;
  TensorFields decode() const;
};
PackedTensorFields pack_fields(const TensorFields&);
}  // namespace tide
