#pragma once
#include "weighted_kv_window.h"
#include "weighted_pool.h"
namespace tide::device_online {
struct WeightedKVSumTape {
  WeightedKVWindowTape window;
  WeightedPoolTape pool;
  at::Tensor memory,total; // [R,2,ds], [R,2]: short then long.
};
// KV §3.1/3.2/3.4, sum mode. Caller explicitly chooses old/new/bound initial raw.
WeightedKVSumTape append_weighted_kv_sum(DeviceProgram&,const weighted::KVConfig&,
  const WeightedRawState&,const at::Tensor& times,const at::Tensor& present,const at::Tensor& error,
  int64_t budget=256*1024*1024);
// roots[R,4]: M_short,M_long,W_short,W_long. The raw adapter value is not read.
WeightedKVWindowGradient append_weighted_kv_sum_vjp(DeviceProgram&,const WeightedKVSumTape&,
  const at::Tensor& memory_bar,const at::Tensor& total_bar,const at::Tensor& roots,
  const at::Tensor& error,int64_t budget=256*1024*1024);
}
