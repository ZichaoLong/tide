#pragma once
#include "weighted_kv_full.h"
#include "weighted_kv_query.h"
#include "weighted_kv_sum.h"
namespace tide::device_online {
// Tapes borrowed by a new reverse program belong to the caller. Enumerating
// their storage before appending keeps the local charge about new allocation.
inline std::vector<at::Tensor> saved_tensors(const WeightedProjectionTape& t){
  return {t.input,t.rstd,t.owners,t.indices,t.matrix,t.bias,t.output};
}
inline std::vector<at::Tensor> saved_tensors(const WeightedKVWindowTape& t){
  return {t.keys,t.values,t.summaries,t.weights,t.decay,t.ages,t.mapping,t.lengths,t.present};
}
inline std::vector<at::Tensor> saved_tensors(const WeightedPoolTape& t){
  return {t.weights,t.values,t.meta,t.probabilities,t.coefficients,t.total,t.mean,t.output};
}
inline std::vector<at::Tensor> saved_tensors(const WeightedRotaryTape& t){return {t.output,t.cosine,t.sine,t.phase};}
template<class T> void append_saved_tensors(std::vector<at::Tensor>& out,const T& t){
  auto values=saved_tensors(t);out.insert(out.end(),values.begin(),values.end());
}
inline std::vector<at::Tensor> saved_tensors(const WeightedResidualTape& t){
  std::vector<at::Tensor> out{t.input,t.hidden,t.output,t.memory_scale,t.ffn_scale};
  for(const auto* p:{&t.out,&t.gate,&t.up,&t.down})append_saved_tensors(out,*p);return out;
}
inline std::vector<at::Tensor> saved_tensors(const WeightedKVQueryTape& t){
  std::vector<at::Tensor> out{t.scale,t.memory,t.total};append_saved_tensors(out,t.window);
  for(const auto* p:{&t.query,&t.key,&t.value})append_saved_tensors(out,*p);
  append_saved_tensors(out,t.rotary);append_saved_tensors(out,t.pool);return out;
}
inline std::vector<at::Tensor> saved_tensors(const WeightedKVSumTape& t){
  std::vector<at::Tensor> out{t.memory,t.total};append_saved_tensors(out,t.window);
  append_saved_tensors(out,t.pool);return out;
}
inline std::vector<at::Tensor> saved_tensors(const WeightedKVFullTape& t){
  std::vector<at::Tensor> out{t.queries,t.scale,t.output};append_saved_tensors(out,t.window);
  append_saved_tensors(out,t.query);append_saved_tensors(out,t.residual);
  for(const auto& x:t.rotary)append_saved_tensors(out,x);
  for(const auto& x:t.pool)append_saved_tensors(out,x);return out;
}
}
