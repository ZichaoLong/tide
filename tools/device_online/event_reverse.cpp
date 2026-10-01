#include "event_reverse.h"
#include "attention_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_event_reverse_links.h"
#include "aclrtlaunch_tide_event_reverse_plan.h"
#include "aclrtlaunch_tide_event_reverse_pack.h"
#include "aclrtlaunch_tide_event_reverse_fold.h"
#include "aclrtlaunch_tide_event_reverse_reduce.h"
#include "aclrtlaunch_tide_cache_merge.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType dtype,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=dtype||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid event attention reverse buffer");
}
CacheCotangents complete(const EventAttentionTape& t,CacheCotangents x) {
  const auto device=t.key.device();const int64_t owners=t.samples*t.nodes.size();
  for(auto pair:{std::make_pair(&x.key,&x.key_connected),std::make_pair(&x.value,&x.value_connected)}) {
    auto& data=*pair.first;auto& flag=*pair.second;
    if(data.defined()!=flag.defined())throw std::invalid_argument("incomplete cache cotangent pair");
    if(data.defined()){tensor(data,device,at::kFloat,t.key.sizes());tensor(flag,device,at::kBool,{owners});}
    else {data=t.key;flag=at::zeros({owners},t.key.options().dtype(at::kBool));}
  }
  return x;
}
CacheCotangents merge(CannProgram& p,const EventAttentionTape& t,const CacheCotangents& local,
    const CacheGradient* extra,const at::Tensor& error,int64_t budget) {
  const auto device=t.key.device();const int64_t owners=t.samples*t.nodes.size(),kv=t.width/t.heads*t.kv_heads;
  if(budget<1||8.L*t.key.numel()+8.L*owners+1024>budget)throw std::invalid_argument("cache adjoint boundary budget exceeded");
  tensor(t.key,device,at::kFloat,{owners,t.capacity,t.kv_heads,t.width/t.heads});
  tensor(t.value,device,at::kFloat,t.key.sizes());tensor(t.lengths,device,at::kLong,{owners});
  auto a=complete(t,local),b=extra?complete(t,*extra):a;const auto lengths=extra?extra->lengths:t.lengths;
  tensor(lengths,device,at::kLong,{owners});
  CacheCotangents out{at::empty_like(t.key),at::empty_like(t.value),at::empty_like(a.key_connected),at::empty_like(a.value_connected)};
  for(auto x:{out.key,out.value,out.key_connected,out.value_connected})p.zero(x);
  for(int64_t phase:{0,1})p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_cache_merge)(phase?32:1,stream,
    ptr(t.lengths),ptr(lengths),ptr(a.key),ptr(a.value),ptr(a.key_connected),ptr(a.value_connected),
    ptr(b.key),ptr(b.value),ptr(b.key_connected),ptr(b.value_connected),ptr(out.key),ptr(out.value),ptr(out.key_connected),ptr(out.value_connected),ptr(error),
    owners,t.capacity,kv,int64_t(extra!=nullptr),phase),"merge connected cache boundary adjoints");},
    {t.lengths,lengths,a.key,a.value,a.key_connected,a.value_connected,b.key,b.value,b.key_connected,b.value_connected,
     out.key,out.value,out.key_connected,out.value_connected,error});
  return out;
}
}
CacheCotangents append_cache_seed(CannProgram& p,const EventAttentionTape& t,const CacheCotangents& local,
    const at::Tensor& error,int64_t budget) {return merge(p,t,local,nullptr,error,budget);}
CacheCotangents append_cache_bridge(CannProgram& p,const EventAttentionTape& t,const CacheCotangents& local,
    const CacheGradient& extra,const at::Tensor& error,int64_t budget) {
  return merge(p,t,local,&extra,error,budget);
}
EventReverse prepare_event_reverse(CannProgram& p,const ReverseTape& t,const EventAttentionTape& a,
    const CacheCotangents& roots,const at::Tensor& error,int64_t budget) {
  if(roots.bias.defined()||roots.bias_connected.defined())throw std::invalid_argument("event attention has no log-bias cache roots");
  const auto device=t.state.metadata.device();const int64_t events=t.state.metadata.size(0),rows=a.metadata.size(0),nodes=t.graph->nodes.size();
  const int64_t ps=a.nodes.size(),owners=a.samples*ps;int64_t buckets=1;
  while(buckets<2.L*events){if(buckets>std::numeric_limits<int64_t>::max()/2)throw std::invalid_argument("KV reverse hash overflow");buckets*=2;}
  if(a.heads<1||a.kv_heads<1||a.width<1||a.width%a.heads||a.heads%a.kv_heads||a.capacity<1||ps<1||a.samples!=t.state.samples
      ||a.width!=t.full.width||48.L*events+8.L*buckets+32.L*owners+8.L*(nodes+1)+8.L*a.key.numel()+4096>budget)
    throw std::invalid_argument("event reverse geometry/tensor budget exceeded");
  const auto kv=a.width/a.heads*a.kv_heads;
  tensor(a.mapping,device,at::kLong,{nodes});tensor(a.windows,device,at::kLong,{ps});tensor(a.config,device,at::kLong,{ps,2});
  tensor(a.metadata,device,at::kLong,{rows,5});tensor(a.values,device,at::kFloat,{rows,2*kv});tensor(a.count,device,at::kLong,{1});
  tensor(a.qkv,device,at::kFloat,{ps+1,a.width,a.width+2*kv});tensor(a.projection,device,at::kFloat,{ps+1,a.width,a.width});
  EventReverse out;static_cast<CacheCotangents&>(out.cache)=merge(p,a,roots,nullptr,error,budget);
  out.cache.lengths=at::empty_like(a.lengths);out.previous=at::empty({events},a.lengths.options());
  out.tails=at::empty({owners},a.lengths.options());out.ranges=at::empty({events,4},a.lengths.options());
  out.node_offsets=at::tensor(event_parameter_offsets(*t.graph,a.width),at::kLong).to(device);auto hash=at::empty({buckets},a.lengths.options());
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_reverse_links)(1,stream,
    ptr(t.state.metadata),ptr(t.state.count),ptr(a.mapping),ptr(a.config),ptr(a.windows),ptr(a.metadata),ptr(a.count),ptr(a.lengths),
    ptr(hash),ptr(out.ranges),ptr(out.previous),ptr(out.tails),ptr(out.cache.lengths),ptr(error),events,rows,a.capacity,nodes,ps,a.samples,buckets),
    "associate actual attention cache/event journals");},{t.state.metadata,t.state.count,a.mapping,a.config,a.windows,a.metadata,a.count,a.lengths,
      hash,out.ranges,out.previous,out.tails,out.cache.lengths,error});
  return out;
}
void append_event_reverse(CannProgram& p,const ReverseTape& t,const EventAttentionTape& a,const EventReverse& reverse,
    const at::Tensor& range,StateVjp& state,const at::Tensor& parameters,const at::Tensor& parameter_on,
    const at::Tensor& error,int64_t chunk,int64_t budget) {
  const int64_t w=a.width,h=a.heads,kh=a.kv_heads,d=w/h,kv=kh*d,k=a.capacity,ps=a.nodes.size(),owners=a.samples*ps,nodes=t.graph->nodes.size();
  const int64_t c=std::min(chunk,owners),cols=w+2*kv;
  const long double own=4.L*c*(4.L*w*cols+4.L*w*w+8.L*w+4.L*kh*k*d)+128.L*c;
  if(c<1||budget<1||own>budget/2.L)throw std::invalid_argument("event reverse batch tensor budget exceeded");
  auto f=a.values.options(),l=a.lengths.options(),b=f.dtype(at::kBool);
  auto plan=at::empty({c,8},l),flags=at::empty({c,6},b),lengths=at::empty({c},l),branch=at::empty_like(error);
  auto x=at::empty({c,1,w},f),weights=at::empty({c,w,cols},f),wo=at::empty({c,w,w},f);
  auto u=at::empty_like(x),projected=at::empty({c,1,cols},f),query=at::empty({c,h,d},f),on=at::empty({c},b);
  auto key=at::empty({c,kh,k,d},f),value=at::empty_like(key),bias=at::zeros({c,k},f);
  auto wot=at::empty_like(wo),cot=at::empty_like(query),dprojected=at::empty_like(projected);
  auto wt=at::empty({c,cols,w},f),dx=at::empty_like(x),xt=at::empty({c,w,1},f);
  auto dw=at::empty_like(weights),dwo=at::empty_like(wo),ot=at::empty_like(xt);
  const auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_reverse_plan)(1,stream,
    ptr(t.state.metadata),ptr(a.config),ptr(reverse.ranges),ptr(reverse.previous),ptr(reverse.tails),ptr(range),ptr(state.proposal_connected),
    ptr(reverse.cache.key_connected),ptr(reverse.cache.value_connected),ptr(plan),ptr(flags),ptr(lengths),ptr(branch),ptr(error),owners,ps,c),
    "pack independent attention reverse owners");},{t.state.metadata,a.config,reverse.ranges,reverse.previous,reverse.tails,range,state.proposal_connected,
      reverse.cache.key_connected,reverse.cache.value_connected,plan,flags,lengths,branch,error});
  p.branch(branch,{done,body});p.mark(body);
  auto pack=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_reverse_pack)(32,stream,
      ptr(plan),ptr(flags),ptr(t.state.metadata),ptr(t.state.values),ptr(state.proposal),ptr(a.qkv),ptr(a.projection),ptr(a.values),
      ptr(x),ptr(weights),ptr(wo),ptr(u),ptr(projected),ptr(query),ptr(key),ptr(value),ptr(on),ptr(error),c,w,h,kh,k,mode),
      "pack event attention adjoint operands");},{plan,flags,t.state.metadata,t.state.values,state.proposal,a.qkv,a.projection,a.values,
        x,weights,wo,u,projected,query,key,value,on,error});
  };
  pack(0);p.batch_matmul(x,weights,projected);pack(1);
  p.permute(wo,{0,2,1},wot);p.batch_matmul(u,wot,cot.reshape({c,1,w}));
  auto local=append_attention_vjp(p,{query,key,value,bias,lengths,cot,on},error,1./std::sqrt(double(d)),std::min<int64_t>(64,k),budget/2);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_reverse_fold)(32,stream,
    ptr(plan),ptr(flags),ptr(t.state.metadata),ptr(a.config),ptr(local.query),ptr(local.key),ptr(local.value),
    ptr(reverse.cache.key),ptr(reverse.cache.value),ptr(dprojected),ptr(error),c,w,h,kh,k),"reverse cache adoption window and clear");},
    {plan,flags,t.state.metadata,a.config,local.query,local.key,local.value,reverse.cache.key,reverse.cache.value,dprojected,error});
  p.permute(weights,{0,2,1},wt);p.batch_matmul(dprojected,wt,dx);
  p.permute(x,{0,2,1},xt);p.batch_matmul(xt,dprojected,dw);
  p.permute(local.output.reshape({c,1,w}),{0,2,1},ot);p.batch_matmul(ot,u,dwo);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_reverse_reduce)(32,stream,
    ptr(plan),ptr(flags),ptr(t.state.metadata),ptr(reverse.node_offsets),ptr(dx),ptr(dw),ptr(dwo),ptr(state.content),ptr(state.content_connected),
    ptr(parameters),ptr(parameter_on),ptr(error),c,w,kv,nodes),"reduce event projection and content adjoints");},
    {plan,flags,t.state.metadata,reverse.node_offsets,dx,dw,dwo,state.content,state.content_connected,parameters,parameter_on,error});
  p.branch(branch,{head});p.mark(done);
}
} // namespace tide::device_online
