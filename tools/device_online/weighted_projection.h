#pragma once
#include "device_program.h"
#include <optional>
namespace tide::device_online {
struct WeightedProjectionParameters {at::Tensor matrix,bias;}; // bias undefined means no bias role.
struct WeightedProjectionTape {
  at::Tensor input,rstd,owners,indices,matrix,bias,output;
  int64_t banks=0;bool normalized=false,has_bias=false;
};
struct WeightedProjectionGradient {
  at::Tensor input,row_connected,owner_connected;
  WeightedProjectionParameters parameters;
};
// Shared linear maps used by weighted memory/Describe/Full. Optional input RMS
// is the pinned N_d, with saved normalized-coordinate first-order contraction.
WeightedProjectionTape append_weighted_projection(DeviceProgram&,const at::Tensor& input,
  const at::Tensor& owners,const WeightedProjectionParameters&,std::optional<double> epsilon,
  const at::Tensor& error,int64_t budget=256*1024*1024);
WeightedProjectionGradient append_weighted_projection_vjp(DeviceProgram&,const WeightedProjectionTape&,
  const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
