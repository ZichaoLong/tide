#include "fiber_reverse.h"
#include "fiber_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_fiber_reverse_links.h"
#include "aclrtlaunch_tide_fiber_reverse_plan.h"
#include "aclrtlaunch_tide_fiber_reverse_pack.h"
#include "aclrtlaunch_tide_fiber_reverse_fold.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid fiber reverse tape");
}
}
FiberReverse prepare_fiber_reverse(CannProgram& p,const ReverseTape& t,const ReverseLinks& links,const FiberAttentionTape& g,
    const CacheCotangents& roots,const at::Tensor& error,int64_t budget) {
  const auto& a=g.cache;const auto device=a.key.device();
  const auto payload=a.key.scalar_type();
  const int64_t events=t.state.metadata.size(0),rows=a.metadata.size(0),nodes=t.graph->nodes.size(),ps=a.nodes.size();
  const int64_t owners=a.samples*ps,fibers=t.fiber_values.size(0),w=a.width,k=a.capacity;int64_t buckets=1;
  while(buckets<2.L*events){if(buckets>std::numeric_limits<int64_t>::max()/2)throw std::invalid_argument("fiber reverse hash overflow");buckets*=2;}
  const long double own=64.L*events+8.L*buckets+16.L*fibers+8.L*(nodes+1)+32.L*owners+4096;
  if(ps<1||a.heads<1||a.kv_heads!=a.heads||w<1||w%a.heads||k<1||a.samples!=t.state.samples||w!=t.full.width||own>budget/2.L)
    throw std::invalid_argument("fiber reverse geometry/tensor budget exceeded");
  tensor(a.mapping,device,at::kLong,{nodes});tensor(a.config,device,at::kLong,{ps,2});
  tensor(a.metadata,device,at::kLong,{rows,5});tensor(a.values,device,at::kFloat,{rows,2*w+1});tensor(a.count,device,at::kLong,{1});
  tensor(a.qkv,device,payload,{ps+1,w,3*w});tensor(a.projection,device,payload,{ps+1,w,w});
  tensor(g.qkv_bias,device,payload,{ps+1,3*w});tensor(g.pool_kinds,device,at::kLong,{ps});tensor(g.pool_lengths,device,at::kLong,{ps});
  if(g.pool_weights.dim()!=2||g.pool_weights.size(1)<1)throw std::invalid_argument("invalid fiber pool reverse bank");
  tensor(g.pool_weights,device,at::kFloat,{ps+1,g.pool_weights.size(1)});
  FiberReverse out;static_cast<CacheCotangents&>(out.cache)=append_fiber_cache_seed(p,g,roots,nullptr,error,budget/2);
  out.cache.lengths=at::empty_like(a.lengths);out.ranges=at::empty({events,6},a.lengths.options());
  out.previous=at::empty({events},a.lengths.options());out.tails=at::empty({owners},a.lengths.options());
  out.tokens=at::empty({fibers},a.lengths.options());out.ticks=at::empty({events},a.lengths.options());
  out.node_offsets=at::tensor(fiber_parameter_offsets(*t.graph,w),at::kLong).to(device);
  auto hash=at::empty({buckets},a.lengths.options()),scratch=at::empty_like(out.tokens);
  const int64_t inputs=t.graph->inputs.size(),sources=inputs+t.graph->edges.size();
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_reverse_links)(1,stream,
    ptr(t.state.metadata),ptr(t.state.count),ptr(a.mapping),ptr(a.config),ptr(a.metadata),ptr(a.count),ptr(a.lengths),ptr(t.state.clock_policy),
    ptr(links.consumer_head),ptr(links.consumer_next),ptr(t.fiber_meta),ptr(t.sources),ptr(hash),ptr(out.ranges),ptr(out.previous),ptr(out.tails),
    ptr(out.cache.lengths),ptr(out.tokens),ptr(scratch),ptr(out.ticks),ptr(error),events,rows,k,nodes,ps,a.samples,buckets,fibers,inputs,sources,t.state.max_repeat_ticks),
    "associate actual fiber cache/events and stably order logical source slots");},
    {t.state.metadata,t.state.count,a.mapping,a.config,a.metadata,a.count,a.lengths,t.state.clock_policy,links.consumer_head,links.consumer_next,
     t.fiber_meta,t.sources,hash,out.ranges,out.previous,out.tails,out.cache.lengths,out.tokens,scratch,out.ticks,error});
  return out;
}
void append_fiber_reverse(CannProgram& p,const ReverseTape& t,const ReverseLinks& links,const FiberAttentionTape& g,const FiberReverse& reverse,
    const at::Tensor& stage,const StateVjp& state,const at::Tensor& messages,const at::Tensor& message_on,
    const at::Tensor& scale_partials,const at::Tensor& parameters,const at::Tensor& parameter_on,
    const at::Tensor& error,int64_t chunk,int64_t budget) {
  const auto& a=g.cache;const int64_t w=a.width,h=a.heads,d=w/h,k=a.capacity,ps=a.nodes.size(),owners=a.samples*ps;
  const int64_t c=std::min(chunk,owners),domain=g.pool_weights.size(1),s=std::min(k,domain),inputs=t.graph->inputs.size();
  const long double own=4.L*c*(s*w+4.L*k*w+4.L*k+4.L*w*w+4.L*w+domain)+512.L*c+8.L*c*s+8.L*t.graph->nodes.size()+4096;
  if(c<1||budget<2||own>budget/2.L)throw std::invalid_argument("fiber reverse batch tensor budget exceeded");
  const auto payload=a.key.options();const int64_t fp16=a.key.scalar_type()==at::kHalf;
  auto f=payload.dtype(at::kFloat),l=a.lengths.options(),b=f.dtype(at::kBool);
  auto plan=at::empty({c,10},l),flags=at::empty({c,4},b),branch=at::empty_like(error);
  FiberVjpInput in;
  in.rows=at::empty({c,s,w},payload);in.slots=at::empty({c,s},l);in.counts=at::empty({c},l);
  in.key=at::empty({c,h,k,d},payload);in.value=at::empty_like(in.key);in.bias=at::empty({c,k},payload);
  in.lengths=at::empty({c},l);in.old_lengths=at::empty_like(in.lengths);in.ticks=at::empty_like(in.lengths);
  in.qkv=at::empty({c,w,3*w},payload);in.qkv_bias=at::empty({c,3*w},payload);in.projection=at::empty({c,w,w},payload);
  in.pool_kinds=at::empty({c},l);in.pool_lengths=at::empty_like(in.pool_kinds);in.pool_weights=at::empty({c,domain},f);
  in.cotangent=at::empty({c,w},f);in.connected=at::empty({c},b);
  in.key_root=at::empty(in.key.sizes(),f);in.value_root=at::empty(in.value.sizes(),f);in.bias_root=at::empty(in.bias.sizes(),f);
  in.key_on=at::empty({c},b);in.value_on=at::empty_like(in.key_on);in.bias_on=at::empty_like(in.key_on);in.max_repeat_ticks=t.state.max_repeat_ticks;
  auto source_counts=at::tensor(t.graph->source_counts,at::kLong).to(a.key.device());
  auto table=at::empty({c*6,5},l),tiles=at::empty({c*6+1},l),table_count=at::empty({2},l);
  const auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_reverse_plan)(1,stream,
    ptr(t.state.metadata),ptr(a.config),ptr(reverse.ranges),ptr(reverse.previous),ptr(reverse.tails),ptr(stage),ptr(state.proposal_connected),
    ptr(reverse.cache.key_connected),ptr(reverse.cache.value_connected),ptr(reverse.cache.bias_connected),ptr(plan),ptr(flags),ptr(branch),ptr(error),owners,ps,c),
    "pack independent fiber cache owners for reverse progression");},
    {t.state.metadata,a.config,reverse.ranges,reverse.previous,reverse.tails,stage,state.proposal_connected,reverse.cache.key_connected,
     reverse.cache.value_connected,reverse.cache.bias_connected,plan,flags,branch,error});
  p.branch(branch,{done,body});p.mark(body);
  for(int64_t phase:{0,1})p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_reverse_pack)(phase?32:1,stream,
    ptr(plan),ptr(flags),ptr(t.state.metadata),ptr(a.config),ptr(reverse.tokens),ptr(t.fiber_meta),ptr(t.sources),ptr(links.messages),ptr(links.scales),ptr(t.fiber_values),
    ptr(a.values),ptr(reverse.ticks),ptr(a.qkv),ptr(g.qkv_bias),ptr(a.projection),ptr(g.pool_kinds),ptr(g.pool_lengths),ptr(g.pool_weights),
    ptr(state.proposal),ptr(reverse.cache.key),ptr(reverse.cache.value),ptr(reverse.cache.bias),ptr(in.rows),ptr(in.slots),ptr(in.counts),
    ptr(in.key),ptr(in.value),ptr(in.bias),ptr(in.lengths),ptr(in.old_lengths),ptr(in.ticks),ptr(in.qkv),ptr(in.qkv_bias),ptr(in.projection),
    ptr(in.pool_kinds),ptr(in.pool_lengths),ptr(in.pool_weights),ptr(in.cotangent),ptr(in.connected),ptr(in.key_root),ptr(in.value_root),ptr(in.bias_root),
    ptr(in.key_on),ptr(in.value_on),ptr(in.bias_on),ptr(error),c,s,w,h,k,domain,inputs,phase,fp16),"pack actual source and KV fiber adjoint operands");},
    {plan,flags,t.state.metadata,a.config,reverse.tokens,t.fiber_meta,t.sources,links.messages,links.scales,t.fiber_values,a.values,reverse.ticks,
     a.qkv,g.qkv_bias,a.projection,g.pool_kinds,g.pool_lengths,g.pool_weights,state.proposal,reverse.cache.key,reverse.cache.value,reverse.cache.bias,
     in.rows,in.slots,in.counts,in.key,in.value,in.bias,in.lengths,in.old_lengths,in.ticks,in.qkv,in.qkv_bias,in.projection,in.pool_kinds,in.pool_lengths,
     in.pool_weights,in.cotangent,in.connected,in.key_root,in.value_root,in.bias_root,in.key_on,in.value_on,in.bias_on,error});
  auto local=append_fiber_vjp(p,in,error,std::min(chunk,c*s),std::min<int64_t>(64,k),budget/2);
  for(int64_t phase:{0,1})p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_reverse_fold)(phase?32:1,stream,
    ptr(plan),ptr(flags),ptr(t.state.metadata),ptr(a.config),ptr(reverse.tokens),ptr(links.messages),ptr(links.scales),ptr(t.fiber_values),ptr(reverse.node_offsets),ptr(source_counts),
    ptr(local.rows),ptr(local.rows_connected),ptr(local.key),ptr(local.value),ptr(local.bias),ptr(local.cache_connected),ptr(local.qkv),ptr(local.qkv_bias),
    ptr(local.projection),ptr(local.projection_bias),ptr(local.decay),ptr(local.pool),ptr(local.parameter_connected),ptr(messages),ptr(message_on),ptr(scale_partials),
    ptr(reverse.cache.key),ptr(reverse.cache.value),ptr(reverse.cache.bias),ptr(reverse.cache.key_connected),ptr(reverse.cache.value_connected),ptr(reverse.cache.bias_connected),
    ptr(parameters),ptr(parameter_on),ptr(table),ptr(tiles),ptr(table_count),ptr(error),c,s,w,h,k,domain,phase),"fold source/cache and used fiber parameter adjoints");},
    {plan,flags,t.state.metadata,a.config,reverse.tokens,links.messages,links.scales,t.fiber_values,reverse.node_offsets,source_counts,local.rows,local.rows_connected,
     local.key,local.value,local.bias,local.cache_connected,local.qkv,local.qkv_bias,local.projection,local.projection_bias,local.decay,local.pool,local.parameter_connected,
     messages,message_on,scale_partials,reverse.cache.key,reverse.cache.value,reverse.cache.bias,reverse.cache.key_connected,reverse.cache.value_connected,
     reverse.cache.bias_connected,parameters,parameter_on,table,tiles,table_count,error});
  p.branch(branch,{head});p.mark(done);
}
} // namespace tide::device_online
