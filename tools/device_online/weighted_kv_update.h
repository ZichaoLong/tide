#pragma once
#include "weighted_projection.h"
#include "weighted_raw_update.h"
namespace tide::device_online {
struct WeightedKVParameters {at::Tensor base,left,right,summary;};
struct WeightedKVUpdateTape {
  weighted::KVConfig config;
  WeightedRawUpdateTape raw;
  WeightedProjectionTape kv,summary;
  WeightedKVParameters saved_rows;
  at::Tensor owners,kv_indices,summary_indices;
  int64_t banks=0;
};
struct WeightedKVUpdateGradient {
  at::Tensor old_values,old_connected,content,content_connected;
  WeightedKVParameters parameters;
  at::Tensor kv_connected,summary_connected;
};
// KV §1.2–1.3: actual shared base + private adapter, independent summary,
// positive reception only, original-time window and complete raw state.
// Raw old/result storage is FP32/FP16. Content and parameter banks remain FP32;
// arithmetic, saved values and VJP are FP32. The final half raw store is guarded
// for representability and has the declared identity cast VJP.
WeightedKVUpdateTape append_weighted_kv_update(DeviceProgram&,const weighted::KVConfig&,
  const WeightedRawState&,const at::Tensor& content,const at::Tensor& time,
  const at::Tensor& owners,const at::Tensor& present,const WeightedKVParameters&,
  const at::Tensor& error,int64_t budget=256*1024*1024);
WeightedKVUpdateGradient append_weighted_kv_update_vjp(DeviceProgram&,const WeightedKVUpdateTape&,
  const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
