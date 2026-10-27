#pragma once
#include "weighted_projection.h"
namespace tide::device_online {
// KV §4.2 and recurrent Full: x=u+alpha_mem Wout m;
// G=x+alpha_ff Wdown(SiLU(Wgate N(x))*Wup N(x)). No sending strength here.
struct WeightedResidualParameters {at::Tensor out,gate,up,down;};
struct WeightedResidualTape {
  WeightedProjectionTape out,gate,up,down;
  at::Tensor input,hidden,output,memory_scale,ffn_scale;
};
struct WeightedResidualGradient {
  at::Tensor input,memory,connected;
  WeightedResidualParameters parameters;
  at::Tensor out_connected,gate_connected,up_connected,down_connected;
};
WeightedResidualTape append_weighted_residual(DeviceProgram&,const at::Tensor& input,
  const at::Tensor& memory,const at::Tensor& owners,const WeightedResidualParameters&,
  double epsilon,double alpha_memory,double alpha_ffn,const at::Tensor& error,
  int64_t budget=256*1024*1024);
WeightedResidualGradient append_weighted_residual_vjp(DeviceProgram&,const WeightedResidualTape&,
  const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,
  int64_t budget=256*1024*1024);
}
