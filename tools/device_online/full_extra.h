#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace tide::device_online {
// Borrowed forward banks, including their independently owned zero sentinels.
// Kinds/groups and SwiGLU's compact node mapping describe static topology only.
struct FullExtraTape {
  at::Tensor lh_kinds,lh_weights,lh_biases;
  std::vector<int64_t> lh_groups;
  at::Tensor swiglu_kinds,swiglu_mapping,gate,up,down;
  bool enabled() const {return lh_kinds.defined()||swiglu_kinds.defined();}
};
struct FullExtraVjp {at::Tensor lh_weights,lh_biases,gate,up,down;};
} // namespace tide::device_online
