#pragma once
// 0770593 allocation §3/S.1: ScalarHead and Query's two affine maps + SiLU.
#include "device_program.h"
namespace tide::device_online {
struct WeightedHeadParameters {at::Tensor matrix,hidden_bias,projection,bias;};
struct WeightedHeadTape {
  at::Tensor input,owners,indices,preactivation,hidden,output;
  WeightedHeadParameters rows;
  int64_t parameter_owners;
};
struct WeightedHeadGradient {
  at::Tensor input,row_connected,owner_connected;
  WeightedHeadParameters parameters;
};
// Banks [owners,H,I], [owners,H], [owners,O,H], [owners,O]. Row owner -1
// denotes absence; otherwise it selects a bank on device. Same owner shares
// parameters; every row retains its private input/activation and root identity.
WeightedHeadTape append_weighted_head(DeviceProgram&,const at::Tensor& input,
  const at::Tensor& owners,const WeightedHeadParameters&,const at::Tensor& error,
  int64_t scratch_budget=64*1024*1024);
WeightedHeadGradient append_weighted_head_vjp(DeviceProgram&,const WeightedHeadTape&,
  const at::Tensor& cotangent,const at::Tensor& roots,const at::Tensor& error,
  int64_t scratch_budget=64*1024*1024);
}
