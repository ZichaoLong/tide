#include "weighted_region_internal.h"
#include "device_launch_tide_weighted_region_prepare.h"
#include "device_launch_tide_weighted_region_finish.h"
#include "device_launch_tide_weighted_region_result.h"
namespace tide::device_online {
WeightedRegionTape append_weighted_region(DeviceProgram& p,const WeightedRegionFields& f,
    const at::Tensor& ids,const at::Tensor& times,const at::Tensor& lengths,const WeightedRegionParameters& weights,
    const weighted::RegionalConfig& c,const at::Tensor& error,int64_t budget){
  using namespace region_detail;Shape s(f.received,c);const auto d=f.received.device();
  for(const auto& x:{f.caps,f.score})check(x,{s.rows,s.width},at::kFloat,d);
  check(ids,{s.rows,s.width},at::kLong,d);check(times,{s.rows},at::kLong,d);check(lengths,{s.rows},at::kLong,d);
  check(error,{1},at::kInt,d);
  if(s.context){check(f.evidence,{s.rows,s.width,s.e},at::kFloat,d);head_shape(weights.context,s.rows,s.e+2,s.h,1,weights.context_owners,d);}
  if(s.attention){for(const auto& x:{f.current,f.key})check(x,{s.rows,s.width,s.q},at::kFloat,d);
    head_shape(weights.query,s.rows,s.q,s.q,s.q,weights.query_owners,d);}
  // Owned fields, immutable coordinates, all intermediate/result buffers,
  // downstream primitive tape and both actual head tapes. Vendor arena is separate.
  const long double own=s.rows*(s.width*(4.L*(5+s.e+2.L*s.q)+9)+4.L*(s.e+3.L*s.q+16)+65);
  admit(own+(s.context?head_bytes(s.rows,s.e+2,s.h,1):0)+(s.attention?head_bytes(s.rows,s.q,s.q,s.q):0)
    +s.rows*((s.allocated?16.L:20.L)*s.width+32),budget);
  WeightedRegionTape t;t.config=c;t.fields=zeros(s,f.received.options());
  p.copy(t.fields.received,f.received);p.copy(t.fields.caps,f.caps);p.copy(t.fields.score,f.score);
  if(s.context)p.copy(t.fields.evidence,f.evidence);
  if(s.attention){p.copy(t.fields.current,f.current);p.copy(t.fields.key,f.key);}
  t.ids=at::empty_like(ids);t.times=at::empty_like(times);t.lengths=at::empty_like(lengths);
  p.copy(t.ids,ids);p.copy(t.times,times);p.copy(t.lengths,lengths);
  auto opts=f.received.options();t.context_input=at::zeros({s.rows,s.e+2},opts);t.query_input=at::zeros({s.rows,s.q},opts);
  t.statistics=at::zeros({s.rows,3},opts);t.context_rows=at::full_like(lengths,-1);t.query_rows=at::full_like(lengths,-1);
  auto errors=at::zeros({s.rows},error.options()),absent=at::full_like(lengths,-1);
  auto context_owners=s.context?weights.context_owners:absent,query_owners=s.attention?weights.query_owners:absent;
  auto copied=t.fields;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_region_prepare)(1,stream,
    ptr(copied.received),ptr(copied.caps),ptr(copied.score),ptr(copied.evidence),ptr(copied.current),ptr(t.ids),ptr(t.times),ptr(t.lengths),
    ptr(context_owners),ptr(query_owners),ptr(t.context_input),ptr(t.query_input),ptr(t.statistics),ptr(t.context_rows),ptr(t.query_rows),
    ptr(errors),ptr(error),s.rows,s.width,s.e,s.q,s.context,s.attention,s.received),"weighted region complete reductions");},
    {copied.received,copied.caps,copied.score,copied.evidence,copied.current,t.ids,t.times,t.lengths,context_owners,query_owners,
     t.context_input,t.query_input,t.statistics,t.context_rows,t.query_rows,errors,error});status(p,errors,error);
  if(s.context){t.context_head=append_weighted_head(p,t.context_input,t.context_rows,weights.context,error,budget);t.context_values=t.context_head->output;}
  else t.context_values=at::zeros({s.rows,1},opts);
  if(s.attention){t.query_head=append_weighted_head(p,t.query_input,t.query_rows,weights.query,error,budget);t.query_values=t.query_head->output;}
  else t.query_values=at::zeros({s.rows,s.q},opts);
  t.scores=at::zeros_like(f.score);t.bounds=at::zeros({s.rows,4},opts);t.parameters=at::zeros({s.rows,2},opts);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_region_finish)(1,stream,
    ptr(copied.score),ptr(copied.key),ptr(t.lengths),ptr(t.statistics),ptr(t.context_values),ptr(t.query_values),ptr(t.scores),
    ptr(t.bounds),ptr(t.parameters),ptr(errors),ptr(error),s.rows,s.width,s.q,s.kind,s.context,s.attention,s.k,s.fixed,s.tau,
    s.alpha_local,s.alpha_query,s.inverse_sqrt),"weighted region scores/control bounds");},
    {copied.score,copied.key,t.lengths,t.statistics,t.context_values,t.query_values,t.scores,t.bounds,t.parameters,errors,error});status(p,errors,error);
  if(s.allocated){
    auto rule=*c.regularizer=="quad"?weighted_allocation::Quad:*c.regularizer=="bern"?weighted_allocation::Bernoulli:weighted_allocation::Entropy;
    t.allocation=append_weighted_allocation(p,t.scores,copied.caps,t.bounds,t.lengths,error,rule,c.ent_alpha.value_or(1),
      s.context&&s.kind==weighted_region::Threshold,budget);t.controls=t.allocation->controls;
  }else{
    auto kind=s.kind==weighted_region::ZeroTotal?weighted_sending::SortedFill:c.hard()?weighted_sending::Hard:
      s.kind==weighted_region::Direct?weighted_sending::Direct:s.kind==weighted_region::Unit?weighted_sending::Unit:
      s.kind==weighted_region::Pooled?weighted_sending::Pooled:weighted_sending::Proportional;
    auto base=c.allocation_base=="received"?weighted_sending::ReceivedBase:c.allocation_base=="uniform"?weighted_sending::UniformBase:weighted_sending::CapBase;
    t.sending=append_weighted_sending(p,t.scores,copied.caps,copied.received,t.parameters,t.ids,t.lengths,error,
      kind,base,kind==weighted_sending::Hard||kind==weighted_sending::Proportional?s.k:0,budget);t.controls=t.sending->controls;
  }
  t.active=at::zeros({s.rows,s.width},opts.dtype(at::kBool));t.history_times=at::full_like(times,-1);
  t.history_present=at::zeros({s.rows},opts.dtype(at::kBool));
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_region_result)(1,stream,
    ptr(t.lengths),ptr(t.times),ptr(t.controls),ptr(t.active),ptr(t.history_times),ptr(t.history_present),ptr(error),s.rows,s.width),
    "weighted region actual selection/history");},{t.lengths,t.times,t.controls,t.active,t.history_times,t.history_present,error});return t;
}
}
