#include "weighted_region_internal.h"
#include "device_launch_tide_weighted_region_reverse_finish.h"
#include "device_launch_tide_weighted_region_reverse_prepare.h"
namespace tide::device_online {
namespace {
long double head_reverse_bytes(const WeightedHeadTape& t){
  const auto r=t.input.size(0),i=t.input.size(1),h=t.hidden.size(1),o=t.output.size(1),banks=t.parameter_owners;
  const long double size=static_cast<long double>(h)*i+h+static_cast<long double>(o)*h+o;
  return 4.L*(r*(size+h+i)+banks*size)+r*5.L+banks*5.L;
}
}
WeightedRegionGradient append_weighted_region_vjp(DeviceProgram& p,const WeightedRegionTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  using namespace region_detail;Shape s(t.fields.received,t.config);const auto d=gradient.device();
  check(gradient,{s.rows,s.width},at::kFloat,t.fields.received.device());check(roots,{s.rows},at::kBool,d);
  check(error,{1},at::kInt,d);check(t.lengths,{s.rows},at::kLong,d);
  for(const auto& x:{t.fields.caps,t.fields.score})check(x,{s.rows,s.width},at::kFloat,d);
  check(t.fields.evidence,{s.rows,s.width,s.e},at::kFloat,d);
  for(const auto& x:{t.fields.current,t.fields.key})check(x,{s.rows,s.width,s.q},at::kFloat,d);
  check(t.statistics,{s.rows,3},at::kFloat,d);check(t.context_input,{s.rows,s.e+2},at::kFloat,d);
  check(t.query_input,{s.rows,s.q},at::kFloat,d);check(t.context_values,{s.rows,1},at::kFloat,d);
  check(t.query_values,{s.rows,s.q},at::kFloat,d);
  if(t.allocation.has_value()!=s.allocated||t.sending.has_value()==s.allocated
      ||t.context_head.has_value()!=s.context||t.query_head.has_value()!=s.attention)
    throw std::invalid_argument("resident region tape differs from function configuration");
  const long double own=s.rows*(4.L*s.width*(3+s.e+2.L*s.q)+4.L*(s.e+3.L*s.q+4)+12);
  admit(own+s.rows*((s.allocated?8.L:12.L)*s.width+26)
      +(s.context?head_reverse_bytes(*t.context_head):0)+(s.attention?head_reverse_bytes(*t.query_head):0),budget);
  at::Tensor vectors,scalars,connected;
  if(s.allocated){auto g=append_weighted_allocation_vjp(p,*t.allocation,gradient,roots,error,budget);
    vectors=g.vectors;scalars=g.scalars;connected=g.connected;}
  else {auto g=append_weighted_sending_vjp(p,*t.sending,gradient,roots,error,budget);
    vectors=g.vectors;scalars=g.scalars;connected=g.connected;}
  WeightedRegionGradient g;g.fields=zeros(s,gradient.options());
  g.connected=at::zeros({s.rows,6},roots.options());
  auto gc=at::zeros({s.rows,1},gradient.options()),gq=at::zeros({s.rows,s.q},gradient.options());
  auto cr=at::zeros_like(roots),qr=at::zeros_like(roots),errors=at::zeros({s.rows},error.options());
  auto f=t.fields,df=g.fields;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_region_reverse_finish)(1,stream,
    ptr(t.lengths),ptr(t.statistics),ptr(t.context_values),ptr(t.query_values),ptr(f.key),ptr(vectors),ptr(scalars),ptr(connected),
    ptr(df.received),ptr(df.caps),ptr(df.score),ptr(df.key),ptr(gc),ptr(gq),ptr(g.connected),ptr(cr),ptr(qr),ptr(errors),ptr(error),
    s.rows,s.width,s.q,s.kind,s.context,s.attention,s.received,s.k,s.fixed,s.alpha_local,s.alpha_query,s.inverse_sqrt),
    "weighted region declared map chain");},
    {t.lengths,t.statistics,t.context_values,t.query_values,f.key,vectors,scalars,connected,df.received,df.caps,
     df.score,df.key,gc,gq,g.connected,cr,qr,errors,error});status(p,errors,error);
  at::Tensor context_input_grad,query_input_grad;
  if(s.context){g.context_head=append_weighted_head_vjp(p,*t.context_head,gc,cr,error,budget);context_input_grad=g.context_head->input;}
  else context_input_grad=at::zeros_like(t.context_input);
  if(s.attention){g.query_head=append_weighted_head_vjp(p,*t.query_head,gq,qr,error,budget);query_input_grad=g.query_head->input;}
  else query_input_grad=at::zeros_like(t.query_input);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_region_reverse_prepare)(1,stream,
    ptr(t.lengths),ptr(f.received),ptr(f.caps),ptr(f.evidence),ptr(f.current),ptr(t.statistics),ptr(t.context_input),ptr(t.query_input),
    ptr(context_input_grad),ptr(query_input_grad),ptr(cr),ptr(qr),ptr(df.received),ptr(df.caps),ptr(df.evidence),ptr(df.current),
    ptr(errors),ptr(error),s.rows,s.width,s.e,s.q,s.received),"weighted region complete reduction adjoint");},
    {t.lengths,f.received,f.caps,f.evidence,f.current,t.statistics,t.context_input,t.query_input,context_input_grad,query_input_grad,
     cr,qr,df.received,df.caps,df.evidence,df.current,errors,error});status(p,errors,error);return g;
}
}
