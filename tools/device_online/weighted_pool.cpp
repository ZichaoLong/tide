#include "weighted_pool.h"
#include "device_launch_tide_weighted_pool_prepare.h"
#include "device_launch_tide_weighted_pool_scale.h"
#include "device_launch_tide_weighted_pool_reverse_prepare.h"
#include "device_launch_tide_weighted_pool_reverse_finish.h"
#include "device_launch_tide_weighted_source_guard.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void check(const at::Tensor& t,at::Device d,at::ScalarType type,at::IntArrayRef shape){
  if(!t.defined()||t.device()!=d||t.scalar_type()!=type||t.sizes()!=shape||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid resident weighted pool buffer");
}
void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("resident weighted pool tensor budget exceeded");}
void validate(const at::Tensor& v){
  if(at::GradMode::is_enabled()||!v.defined()||v.dim()!=3||v.size(0)<1||v.size(1)<1||v.size(2)<1||v.device().type()!=resident_device_type)
    throw std::invalid_argument("resident weighted pool needs explicit bounded FP32 accelerator rows");
  validate_kernel_device(v.device());check(v,v.device(),at::kFloat,v.sizes());
}
void finite_output(DeviceProgram& p,const at::Tensor& output,const at::Tensor& error){
  auto errors=at::zeros({output.size(0)},error.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_source_guard)(1,stream,
    ptr(output),ptr(error),ptr(output),ptr(errors),ptr(error),output.size(0),output.size(1),0),"weighted pool result guard");},{output,errors,error});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(1,stream,
    ptr(errors),ptr(error),errors.numel()),"weighted pool result decision");},{errors,error});
}
}
WeightedPoolTape append_weighted_pool(DeviceProgram& p,const at::Tensor& w,const at::Tensor& v,
    const at::Tensor& logits,const at::Tensor& lengths,const at::Tensor& present,const at::Tensor& error,int64_t available){
  validate(v);const auto device=v.device();const auto rows=v.size(0),capacity=v.size(1),width=v.size(2);const bool attention=logits.defined();
  check(w,device,at::kFloat,{rows,capacity});if(attention)check(logits,device,at::kFloat,{rows,capacity});
  check(lengths,device,at::kLong,{rows});check(present,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  budget(rows*(4.L*(capacity*static_cast<long double>(width)+4.L*capacity+2.L*width+2)+28),available);
  WeightedPoolTape t{at::zeros_like(w),at::zeros_like(v),at::zeros({rows,3},lengths.options()),at::zeros_like(w),at::zeros_like(w),
    at::zeros({rows},w.options()),at::zeros({rows,width},v.options()),at::zeros({rows,width},v.options()),attention};
  auto scores=at::zeros_like(w),amplitude=at::zeros_like(t.total),l=attention?logits:error;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_pool_prepare)(1,stream,
    ptr(w),ptr(v),ptr(l),ptr(lengths),ptr(present),ptr(t.weights),ptr(t.values),ptr(scores),ptr(t.meta),ptr(error),rows,capacity,width,attention),
    "weighted pool complete ragged plan");},{w,v,l,lengths,present,t.weights,t.values,scores,t.meta,error});
  p.sum(t.weights,1,false,t.total);if(attention)p.softmax(scores,1,t.probabilities);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_pool_scale)(1,stream,
    ptr(t.weights),ptr(t.total),ptr(t.probabilities),ptr(t.meta),ptr(t.coefficients),ptr(amplitude),ptr(error),rows,capacity,attention),
    "weighted pool total attenuation");},{t.weights,t.total,t.probabilities,t.meta,t.coefficients,amplitude,error});
  p.batch_matmul(t.coefficients.unsqueeze(1),t.values,t.mean.unsqueeze(1));p.multiply(t.mean,amplitude.unsqueeze(1),t.output);
  finite_output(p,t.output,error);return t;
}
WeightedPoolGradient append_weighted_pool_vjp(DeviceProgram& p,const WeightedPoolTape& t,
    const at::Tensor& vector_bar,const at::Tensor& total_bar,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  validate(t.values);const auto device=t.values.device();const auto rows=t.values.size(0),capacity=t.values.size(1),width=t.values.size(2);
  for(const auto& x:{t.weights,t.probabilities,t.coefficients})check(x,device,at::kFloat,{rows,capacity});
  for(const auto& x:{t.mean,t.output,vector_bar})check(x,device,at::kFloat,{rows,width});
  check(t.meta,device,at::kLong,{rows,3});check(t.total,device,at::kFloat,{rows});check(total_bar,device,at::kFloat,{rows});
  check(roots,device,at::kBool,{rows,2});check(error,device,at::kInt,{1});
  budget(rows*(4.L*(2.L*capacity*width+4.L*capacity+width+2)+3),available);
  WeightedPoolGradient g{at::zeros_like(t.weights),at::zeros_like(t.values),at::zeros_like(t.weights),at::zeros({rows,3},roots.options())};
  auto clean_bar=at::zeros_like(vector_bar),stat=at::zeros_like(total_bar),centered=at::zeros_like(t.values),value_coeff=at::zeros_like(t.weights);
  auto dots=at::zeros_like(t.weights),mean_dot=at::zeros_like(total_bar);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_pool_reverse_prepare)(1,stream,
    ptr(t.meta),ptr(t.weights),ptr(t.values),ptr(t.probabilities),ptr(t.coefficients),ptr(t.total),ptr(t.output),ptr(vector_bar),ptr(total_bar),ptr(roots),
    ptr(clean_bar),ptr(stat),ptr(centered),ptr(value_coeff),ptr(g.connected),ptr(error),rows,capacity,width,t.attention),
    "weighted pool independent saved vector/statistic roots");},{t.meta,t.weights,t.values,t.probabilities,t.coefficients,t.total,t.output,vector_bar,total_bar,roots,clean_bar,stat,centered,value_coeff,g.connected,error});
  p.batch_matmul(value_coeff.unsqueeze(2),clean_bar.unsqueeze(1),g.values);
  p.batch_matmul(centered,clean_bar.unsqueeze(2),dots.unsqueeze(2));
  p.batch_matmul(t.mean.unsqueeze(1),clean_bar.unsqueeze(2),mean_dot.reshape({rows,1,1}));
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_pool_reverse_finish)(1,stream,
    ptr(t.meta),ptr(t.weights),ptr(t.probabilities),ptr(t.total),ptr(dots),ptr(mean_dot),ptr(stat),ptr(g.connected),ptr(g.weights),ptr(g.values),ptr(g.logits),ptr(error),
    rows,capacity,width,t.attention),"weighted pool saved map adjoints");},{t.meta,t.weights,t.probabilities,t.total,dots,mean_dot,stat,g.connected,g.weights,g.values,g.logits,error});
  if(!t.attention)g.logits=at::Tensor{};return g;
}
}
