#pragma once
#include "weighted_kv_window.h"
#include "weighted_projection.h"
#include "weighted_rotary.h"
#include "weighted_pool.h"
namespace tide::device_online {
struct WeightedKVQueryParameters {at::Tensor query,key,value;};
struct WeightedKVQueryTape {
  WeightedKVWindowTape window;
  WeightedProjectionTape query,key,value;
  WeightedRotaryTape rotary;
  WeightedPoolTape pool;
  at::Tensor scale,memory,total; // scalar, [R,2,ds], [R,2].
};
struct WeightedKVQueryGradient {
  at::Tensor raw,raw_connected,input,input_connected;
  WeightedKVQueryParameters parameters;
  at::Tensor query_connected,key_connected,value_connected;
};
// KV §3.3/3.4: current normalized query, separate K/V projections and read RoPE.
// Raw storage may be FP32/FP16. Read input/parameters, numerical outputs and
// all adjoints remain explicitly FP32, independently of raw state storage.
WeightedKVQueryTape append_weighted_kv_query(DeviceProgram&,const weighted::KVConfig&,const WeightedRawState&,
  const at::Tensor& time,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& present,
  const WeightedKVQueryParameters&,const at::Tensor& error,int64_t budget=256*1024*1024);
WeightedKVQueryGradient append_weighted_kv_query_vjp(DeviceProgram&,const WeightedKVQueryTape&,
  const at::Tensor& memory_bar,const at::Tensor& total_bar,const at::Tensor& roots,
  const at::Tensor& error,int64_t budget=256*1024*1024);
}
