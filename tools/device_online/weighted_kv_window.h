#pragma once
#include "weighted_raw_state.h"
#include "device_program.h"
namespace tide::device_online {
// KV §1.3/3.1: original int64 time masking and two passive weight channels.
struct WeightedKVWindowTape {
  WeightedRawLayout layout;
  at::Tensor keys,values,summaries,weights,decay,ages,mapping,lengths,present;
};
struct WeightedKVWindowGradient {at::Tensor raw,connected;};
WeightedKVWindowTape append_weighted_kv_window(DeviceProgram&,const weighted::KVConfig&,
  const WeightedRawState&,const at::Tensor& times,const at::Tensor& present,
  const at::Tensor& error,int64_t budget=256*1024*1024);
// Independent roots[R,5]: keys,values,summaries,short_weights,long_weights.
// Output connected uses the raw layout's five fields; adapter value stays None.
WeightedKVWindowGradient append_weighted_kv_window_vjp(DeviceProgram&,const WeightedKVWindowTape&,
  const at::Tensor& keys,const at::Tensor& values,const at::Tensor& summaries,
  const at::Tensor& weights,const at::Tensor& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
