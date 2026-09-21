#pragma once
#include "AccumulateLocal.h"
#include "tide/types.h"

namespace lh_oracle {
namespace AL = AccumulateLocal;
using tide::Index;
using Mode = AL::KVHidden::AttentionMode;
inline std::map<std::string, at::Tensor> cache(const AL::KVHidden& h, const at::TensorOptions& opts) {
  at::Tensor k, v;
  if (AL::KVHidden::attention_mode == Mode::LOOP || AL::KVHidden::attention_mode == Mode::PACKED) {
    k = h.keys.empty() ? at::empty({h.n_head, 0, h.D/h.n_head}, opts) : at::cat(h.keys, 1);
    v = h.values.empty() ? at::empty({h.n_head, 0, h.D/h.n_head}, opts) : at::cat(h.values, 1);
  } else { k = h.keys_cache.slice(1, 0, h.endidx); v = h.values_cache.slice(1, 0, h.endidx); }
  return {{"key", k.transpose(0, 1).clone()}, {"value", v.transpose(0, 1).clone()},
          {"log_bias", h.total_growth_rate.slice(0, 0, h.endidx).clone()}};
}
inline std::map<std::string, at::Tensor> cache(const AL::BatchPtrKVHidden& h, Index b, const at::TensorOptions& opts) {
  if (AL::KVHidden::attention_mode != Mode::CROSSBATCH) return cache(*h.hptrs[b], opts);
  const auto n = h.endindices[b];
  return {{"key", h.batch_keys_cache[b].slice(1, 0, n).transpose(0, 1).clone()},
          {"value", h.batch_values_cache[b].slice(1, 0, n).transpose(0, 1).clone()},
          {"log_bias", h.batch_total_growth_rate[b].slice(0, 0, n).clone()}};
}
}  // namespace lh_oracle
