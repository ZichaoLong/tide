#include "weighted_convolution.h"
#include "device_launch_tide_weighted_conv_plan.h"
#include "device_launch_tide_weighted_conv_guard.h"
#include "device_launch_tide_weighted_conv_reverse.h"
#include <ATen/core/grad_mode.h>
#include <limits>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void check(const at::Tensor& t,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!t.defined()||t.device()!=device||t.scalar_type()!=type||t.sizes()!=shape||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid weighted fixed-time convolution buffer");
}
void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("weighted convolution tensor budget exceeded");}
void domain(const WeightedRawLayout& l,const WeightedRawState& s,const at::Tensor& error){
  validate_weighted_raw_buffers(l,s);
  if(!l.convolution()||s.floating.device().type()!=resident_device_type||s.floating.scalar_type()!=at::kFloat)
    throw std::invalid_argument("weighted convolution requires GDN/SSD raw schema and explicit FP32 accelerator");
  validate_kernel_device(s.floating.device());check(error,s.floating.device(),at::kInt,{1});
}
void guard(DeviceProgram& p,const WeightedConvTape& t,const at::Tensor& values,const at::Tensor& roots,
    const at::Tensor& clean,const at::Tensor& flags,const at::Tensor& error,int64_t mode){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_conv_guard)(1,stream,
    ptr(t.owners),ptr(t.rows.weight),ptr(t.rows.bias),ptr(values),ptr(roots),ptr(clean),ptr(flags),ptr(error),
    t.owners.numel(),t.rows.weight.size(1),t.output.size(1),mode),"weighted fixed-time convolution guards");},
    {t.owners,t.rows.weight,t.rows.bias,values,roots,clean,flags,error});
}
}
WeightedConvTape append_weighted_convolution(DeviceProgram& p,const WeightedRawLayout& l,const WeightedRawState& old,
    const at::Tensor& time,const at::Tensor& current,const at::Tensor& owners,const WeightedConvParameters& w,
    int64_t stride,const at::Tensor& error,int64_t available){
  domain(l,old,error);const auto device=old.floating.device();const auto rows=old.floating.size(0),features=l.field("conv_values").width;
  if(!w.weight.defined()||w.weight.dim()!=3||w.weight.size(0)<1||w.weight.size(1)<1||stride<1)
    throw std::invalid_argument("invalid weighted convolution bank/taps/stride");
  const auto banks=w.weight.size(0),size=w.weight.size(1);
  if(size-1>(std::numeric_limits<int64_t>::max()-1)/stride||(size-1)*stride!=l.horizon())
    throw std::invalid_argument("convolution taps/stride differ from raw schema horizon");
  check(w.weight,device,at::kFloat,{banks,size,features});check(w.bias,device,at::kFloat,{banks,features});
  check(time,device,at::kLong,{rows});check(current,device,at::kFloat,{rows,features});check(owners,device,at::kLong,{rows});
  budget(rows*(8.L*l.integer_width()+8.L*(size+2)+4.L*features*(3.L*size+2)),available);
  WeightedConvTape t{l,{at::zeros({rows,size,features},current.options()),at::zeros_like(current)},at::zeros_like(owners),
    at::zeros_like(owners),at::zeros_like(old.integer),at::zeros({rows,size},owners.options()),
    at::zeros({rows,size,features},current.options()),at::zeros_like(current),banks,stride};
  p.copy(t.owners,owners);p.copy(t.old_integer,old.integer);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_conv_plan)(1,stream,
    ptr(old.floating),ptr(t.old_integer),ptr(time),ptr(current),ptr(t.owners),ptr(t.indices),ptr(t.mapping),ptr(t.taps),ptr(error),
    rows,l.floating_width(),l.capacity(),l.field("conv_values").offset,features,size,stride,banks),"pack original logical-time convolution taps");},
    {old.floating,t.old_integer,time,current,t.owners,t.indices,t.mapping,t.taps,error});
  p.index_select(w.weight,0,t.indices,t.rows.weight);p.index_select(w.bias,0,t.indices,t.rows.bias);
  guard(p,t,error,error,error,error,error,0);auto products=at::zeros_like(t.taps);
  p.multiply(t.taps,t.rows.weight,products);p.sum(products,1,false,t.output);p.add(t.output,t.rows.bias);
  guard(p,t,t.output,error,error,error,error,1);return t;
}
WeightedConvGradient append_weighted_convolution_vjp(DeviceProgram& p,const WeightedConvTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  const auto& l=t.layout;const auto device=t.output.device();
  if(at::GradMode::is_enabled()||device.type()!=resident_device_type||!l.convolution()||t.banks<1
    ||!t.rows.weight.defined()||t.rows.weight.dim()!=3||t.rows.weight.size(1)<1||t.owners.dim()!=1||t.owners.numel()<1)
    throw std::invalid_argument("invalid weighted convolution tape");
  const auto rows=t.owners.numel(),size=t.rows.weight.size(1),features=l.field("conv_values").width;
  check(t.owners,device,at::kLong,{rows});check(t.old_integer,device,at::kLong,{rows,l.integer_width()});
  check(t.mapping,device,at::kLong,{rows,size});check(t.taps,device,at::kFloat,{rows,size,features});
  check(t.rows.weight,device,at::kFloat,{rows,size,features});check(t.rows.bias,device,at::kFloat,{rows,features});
  check(t.output,device,at::kFloat,{rows,features});check(gradient,device,at::kFloat,{rows,features});
  check(roots,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  budget(rows*(4.L*l.floating_width()+4.L*features*(2.L*size+2)+2)+t.banks*(4.L*(size+1)*features+1),available);
  auto dg=at::zeros_like(gradient),dt=at::zeros_like(t.taps),dw=at::zeros_like(t.taps);
  WeightedConvGradient g{{at::zeros({t.banks,size,features},gradient.options()),at::zeros({t.banks,features},gradient.options())},
    at::zeros_like(gradient),at::zeros({rows,l.floating_width()},gradient.options()),at::zeros_like(roots),at::zeros_like(roots),
    at::zeros({t.banks},roots.options())};
  guard(p,t,gradient,roots,dg,g.row_connected,error,2);
  p.multiply(dg.unsqueeze(1),t.rows.weight,dt);p.multiply(dg.unsqueeze(1),t.taps,dw);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_conv_reverse)(1,stream,
    ptr(t.owners),ptr(t.old_integer),ptr(t.mapping),ptr(dt),ptr(dw),ptr(dg),ptr(g.row_connected),ptr(g.raw_cache),ptr(g.current),ptr(g.cache_connected),
    ptr(g.parameters.weight),ptr(g.parameters.bias),ptr(g.owner_connected),ptr(error),rows,l.floating_width(),l.capacity(),l.field("conv_values").offset,
    features,size,t.banks),"convolution cache/current and shared parameter adjoints");},
    {t.owners,t.old_integer,t.mapping,dt,dw,dg,g.row_connected,g.raw_cache,g.current,g.cache_connected,g.parameters.weight,g.parameters.bias,g.owner_connected,error});
  return g;
}
}
