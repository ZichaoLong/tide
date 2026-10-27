#include "weighted_aggregate_internal.h"
#include "device_launch_tide_weighted_aggregate_prepare.h"
#include "device_launch_tide_weighted_aggregate_reduce.h"
namespace tide::device_online {
WeightedAggregateTape append_weighted_aggregate(DeviceProgram& p,const at::Tensor& payload,
    const at::Tensor& lengths,const at::Tensor& owners,const WeightedSourceParameters& weights,
    float alpha,float epsilon,const at::Tensor& error,int64_t budget){
  using namespace aggregate_detail;Shape s(payload,alpha,epsilon);const auto device=payload.device();
  check(lengths,{s.rows},at::kLong,device);check(owners,{s.rows,s.width},at::kLong,device);check(error,{1},at::kInt,device);
  long double head=0;int64_t banks=0,hidden=0;
  if(alpha>0){
    if(!weights.a.defined()||weights.a.dim()!=3||weights.a.size(0)<1||weights.a.size(1)<1)
      throw std::invalid_argument("missing resident source-function banks");
    banks=weights.a.size(0);hidden=weights.a.size(1);
    check(weights.a,{banks,hidden,s.d},at::kFloat,device);check(weights.b,{banks,s.d,hidden},at::kFloat,device);
    check(weights.embedding,{banks,hidden},at::kFloat,device);
    head=s.rows*s.width*(4.L*(3.L*s.d+4.L*hidden*s.d+3.L*hidden)+20)+4.L*banks*s.d;
  }
  admit(s.rows*(s.width*(4.L*(5.L*s.d+2)+8)+4.L*(s.d+1)+16)+head,budget);
  auto options=payload.options();WeightedAggregateTape t;t.alpha=alpha;t.epsilon=epsilon;
  t.payload=at::empty_like(payload);t.lengths=at::empty_like(lengths);p.copy(t.payload,payload);p.copy(t.lengths,lengths);
  t.clean=at::zeros({s.rows*s.width,s.d},options);t.owners=at::full({s.rows*s.width},-1,owners.options());
  t.normal=at::zeros_like(t.clean);t.rstd=at::zeros({s.rows*s.width,1},options);
  t.total=at::zeros({s.rows},options);t.values=at::zeros_like(t.clean);t.output=at::zeros({s.rows,s.d+1},options);
  auto errors=at::zeros({s.rows},error.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_aggregate_prepare)(1,stream,
    ptr(t.payload),ptr(t.lengths),ptr(owners),ptr(t.clean),ptr(t.owners),ptr(t.total),ptr(errors),ptr(error),
    s.rows,s.width,s.d,alpha),"weighted Aggregate complete fiber plan");},
    {t.payload,t.lengths,owners,t.clean,t.owners,t.total,errors,error});status(p,errors,error);
  at::Tensor transformed;
  if(alpha>0){
    p.rms_norm(t.clean,epsilon,t.normal,t.rstd);
    WeightedHeadParameters w{weights.a,weights.embedding,weights.b,at::zeros({banks,s.d},options)};
    t.source=append_weighted_head(p,t.normal,t.owners,w,error,budget);transformed=t.source->output;
  }else transformed=at::zeros_like(t.clean);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_aggregate_reduce)(1,stream,
    ptr(t.payload),ptr(t.lengths),ptr(transformed),ptr(t.total),ptr(t.values),ptr(t.output),ptr(errors),ptr(error),
    s.rows,s.width,s.d,alpha),"weighted Aggregate actual mass/content");},
    {t.payload,t.lengths,transformed,t.total,t.values,t.output,errors,error});status(p,errors,error);return t;
}
}
