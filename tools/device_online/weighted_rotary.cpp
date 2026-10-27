#include "weighted_rotary.h"
#include "device_launch_tide_weighted_rotary_phase.h"
#include "device_launch_tide_weighted_rotary_reduce.h"
#include "device_launch_tide_weighted_rotary_unfold.h"
#include "device_launch_tide_weighted_rotary_apply.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void check(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid weighted rotary buffer");
}
void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("weighted rotary tensor budget exceeded");}
void shape(const at::Tensor& x){
  if(at::GradMode::is_enabled()||!x.defined()||x.dim()!=3||x.size(0)<1||x.size(1)<1||x.size(2)<1||x.device().type()!=resident_device_type)
    throw std::invalid_argument("weighted rotary requires explicit bounded FP32 accelerator rows");
  validate_kernel_device(x.device());check(x,x.device(),at::kFloat,x.sizes());
}
void apply(DeviceProgram& p,const WeightedRotaryTape& t,const at::Tensor& input,const at::Tensor& roots,
    const at::Tensor& output,const at::Tensor& error,bool reverse){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_rotary_apply)(1,stream,
    ptr(input),ptr(t.cosine),ptr(t.sine),ptr(roots),ptr(output),ptr(error),input.size(0),input.size(1),input.size(2),t.pairs,reverse),
    "weighted saved relative rotation");},{input,t.cosine,t.sine,roots,output,error});
}
}
WeightedRotaryTape append_weighted_rotary(DeviceProgram& p,const at::Tensor& input,const at::Tensor& ages,
    double scale,const std::vector<double>& frequencies,const at::Tensor& error,int64_t available){
  shape(input);const auto device=input.device();const auto rows=input.size(0),capacity=input.size(1),width=input.size(2),pairs=int64_t(frequencies.size());
  if(!std::isfinite(scale)||!std::isfinite(float(scale))||float(scale)<=0||pairs>width/2)
    throw std::invalid_argument("invalid weighted rotary positions/frequencies");
  for(auto f:frequencies)if(!std::isfinite(f)||!std::isfinite(float(f)))throw std::invalid_argument("unrepresentable weighted rotary frequency");
  check(ages,device,at::kLong,{rows,capacity});check(error,device,at::kInt,{1});
  budget(4.L*(rows*static_cast<long double>(capacity)*(width+5.L*std::max<int64_t>(1,pairs))+pairs),available);
  auto phase=at::zeros({rows,capacity,std::max<int64_t>(1,pairs)},input.options());
  WeightedRotaryTape t{at::zeros_like(input),at::zeros_like(phase),at::zeros_like(phase),phase,pairs};
  if(pairs){auto f=at::tensor(frequencies,at::kDouble).to(input.options());
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_rotary_phase)(1,stream,
      ptr(ages),ptr(f),ptr(phase),ptr(error),rows*capacity,pairs,float(scale)),"weighted exact-age relative phase");},{ages,f,phase,error});
    auto reduced=at::zeros_like(phase),quadrant=at::zeros_like(phase,phase.options().dtype(at::kInt));
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_rotary_reduce)(1,stream,
      ptr(phase),ptr(reduced),ptr(quadrant),ptr(error),phase.numel()),"weighted finite FP32 phase reduction");},{phase,reduced,quadrant,error});
    p.cos(reduced,t.cosine);p.sin(reduced,t.sine);
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_rotary_unfold)(1,stream,
      ptr(t.cosine),ptr(t.sine),ptr(quadrant),ptr(error),phase.numel()),"weighted saved full-phase trigonometric values");},{t.cosine,t.sine,quadrant,error});
  }
  apply(p,t,input,error,t.output,error,false);return t;
}
at::Tensor append_weighted_rotary_vjp(DeviceProgram& p,const WeightedRotaryTape& t,const at::Tensor& bar,
    const at::Tensor& roots,const at::Tensor& error,int64_t available){
  shape(t.output);const auto device=t.output.device();const auto rows=t.output.size(0),capacity=t.output.size(1),width=t.output.size(2);
  if(t.pairs<0||t.pairs>width/2)throw std::invalid_argument("invalid weighted rotary tape");
  check(t.cosine,device,at::kFloat,{rows,capacity,std::max<int64_t>(1,t.pairs)});check(t.sine,device,at::kFloat,t.cosine.sizes());
  check(bar,device,at::kFloat,t.output.sizes());check(roots,device,at::kBool,{rows});check(error,device,at::kInt,{1});budget(4.L*bar.numel(),available);
  auto out=at::zeros_like(bar);apply(p,t,bar,roots,out,error,true);return out;
}
}
