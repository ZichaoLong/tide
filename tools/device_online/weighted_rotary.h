#pragma once
#include "device_program.h"
namespace tide::device_online {
struct WeightedRotaryTape {at::Tensor output,cosine,sine,phase;int64_t pairs=0;};
// KV §2 relative Rot(-age/scale), original int64 differences. Frequencies fixed.
WeightedRotaryTape append_weighted_rotary(DeviceProgram&,const at::Tensor& keys,
  const at::Tensor& ages,double position_scale,const std::vector<double>& frequencies,
  const at::Tensor& error,int64_t budget=256*1024*1024);
at::Tensor append_weighted_rotary_vjp(DeviceProgram&,const WeightedRotaryTape&,
  const at::Tensor& bar,const at::Tensor& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
