#include "weighted_aggregate_internal.h"
#include "device_launch_tide_weighted_aggregate_reverse.h"
#include "device_launch_tide_weighted_aggregate_norm_vjp.h"
namespace tide::device_online {
WeightedAggregateGradient append_weighted_aggregate_vjp(DeviceProgram& p,const WeightedAggregateTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  using namespace aggregate_detail;Shape s(t.payload,t.alpha,t.epsilon);const auto device=t.payload.device();
  check(gradient,{s.rows,s.d+1},at::kFloat,device);check(roots,{s.rows},at::kBool,device);check(error,{1},at::kInt,device);
  check(t.lengths,{s.rows},at::kLong,device);check(t.values,{s.rows*s.width,s.d},at::kFloat,device);
  check(t.output,{s.rows,s.d+1},at::kFloat,device);check(t.normal,{s.rows*s.width,s.d},at::kFloat,device);
  check(t.rstd,{s.rows*s.width,1},at::kFloat,device);
  if(t.source.has_value()!=(t.alpha>0))throw std::invalid_argument("resident Aggregate tape/config mismatch");
  long double head=0;
  if(t.source){const auto h=t.source->hidden.size(1),banks=t.source->parameter_owners;
    const long double bank=2.L*h*s.d+h+s.d;
    head=4.L*(s.rows*s.width*(bank+h+s.d)+banks*bank)+s.rows*s.width*5.L+banks*5.L;}
  admit(s.rows*(s.width*(4.L*(2.L*s.d+1)+2)+4)+head,budget);
  auto options=gradient.options();WeightedAggregateGradient g;
  g.payload=at::zeros_like(t.payload);g.connected=at::zeros({s.rows,s.width},roots.options());
  auto df=at::zeros({s.rows*s.width,s.d},options),source_roots=at::zeros({s.rows*s.width},roots.options());
  auto errors=at::zeros({s.rows},error.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_aggregate_reverse)(1,stream,
    ptr(t.payload),ptr(t.lengths),ptr(t.values),ptr(t.output),ptr(gradient),ptr(roots),ptr(g.payload),ptr(df),
    ptr(g.connected),ptr(source_roots),ptr(errors),ptr(error),s.rows,s.width,s.d,t.alpha),"weighted Aggregate mean adjoint");},
    {t.payload,t.lengths,t.values,t.output,gradient,roots,g.payload,df,g.connected,source_roots,errors,error});status(p,errors,error);
  if(t.source){
    auto head=append_weighted_head_vjp(p,*t.source,df,source_roots,error,budget);
    g.parameters=WeightedSourceParameters{head.parameters.matrix,head.parameters.projection,head.parameters.hidden_bias};
    g.owner_connected=head.owner_connected;
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_aggregate_norm_vjp)(1,stream,
      ptr(t.normal),ptr(t.rstd),ptr(head.input),ptr(source_roots),ptr(g.payload),ptr(errors),ptr(error),
      s.rows,s.width,s.d),"weighted Aggregate RMS adjoint");},
      {t.normal,t.rstd,head.input,source_roots,g.payload,errors,error});status(p,errors,error);
  }
  return g;
}
}
