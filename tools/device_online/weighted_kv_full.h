#pragma once
#include "weighted_kv_window.h"
#include "weighted_rotary.h"
#include "weighted_pool.h"
#include "weighted_residual.h"
namespace tide::device_online {
struct WeightedKVFullParameters {at::Tensor query;WeightedResidualParameters residual;};
struct WeightedKVFullTape {
  WeightedKVWindowTape window;
  WeightedProjectionTape query;
  std::vector<WeightedRotaryTape> rotary;
  std::vector<WeightedPoolTape> pool;
  WeightedResidualTape residual;
  at::Tensor queries,scale,output;
};
struct WeightedKVFullGradient {
  at::Tensor raw,raw_connected,input,input_connected;
  WeightedKVFullParameters parameters;
  at::Tensor query_connected,out_connected,gate_connected,up_connected,down_connected;
};
// KV §4: selected Full reads the actual proposal snapshot. Returns G, not c*G.
// active is supplied by actual region selection; no activity inferred from u.
WeightedKVFullTape append_weighted_kv_full(DeviceProgram&,const weighted::KVConfig&,const WeightedRawState&,
  const at::Tensor& time,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& active,
  const WeightedKVFullParameters&,const at::Tensor& error,int64_t budget=256*1024*1024);
WeightedKVFullGradient append_weighted_kv_full_vjp(DeviceProgram&,const WeightedKVFullTape&,
  const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
