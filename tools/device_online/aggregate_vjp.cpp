#include "aggregate_vjp.h"
#include "packed_aggregate.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_aggregate_vjp_plan.h"
#include "aclrtlaunch_tide_aggregate_vjp_payload.h"
#include "aclrtlaunch_tide_aggregate_vjp_reduce.h"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <ATen/core/grad_mode.h>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid Aggregate VJP buffer");
}
}
void append_aggregate_vjp(CannProgram& p,const ReverseTape& t,const ReverseLinks& links,
    const at::Tensor& stage_count,const at::Tensor& stage_range,const at::Tensor& gradient,const at::Tensor& connected,
    const at::Tensor& messages,const at::Tensor& scale_partials,AggregateVjp& out,const at::Tensor& error,int64_t rows,int64_t budget) {
  const auto& a=t.aggregate;if(!a.kinds.defined())return;
  if(at::GradMode::is_enabled()||!t.graph||!t.state.metadata.defined()||t.state.metadata.dim()!=2
      ||!t.fiber_values.defined()||t.fiber_values.dim()!=2||!t.sources.defined()||t.sources.dim()!=2
      ||!links.messages.defined()||links.messages.dim()!=2)
    throw std::invalid_argument("Aggregate VJP requires actual no-grad device tape");
  const auto device=t.fiber_values.device();const int64_t capacity=t.state.metadata.size(0),nodes=t.graph->nodes.size(),slots=a.slots,d=t.full.width;
  const long double row_bytes=32.L*slots*d+96.L*slots+16.L*d+256.L;
  if(device.type()!=c10::DeviceType::PrivateUse1||capacity<1||nodes<1||d<1||slots<1||rows<1||budget<row_bytes+256)
    throw std::invalid_argument("one Aggregate VJP row exceeds tensor budget");
  const int64_t chunk=std::min<int64_t>({capacity,rows,int64_t((budget-256)/row_bytes)});
  tensor(a.kinds,device,at::kLong,{nodes});tensor(a.lengths,device,at::kLong,{nodes});tensor(a.weights,device,at::kFloat,{nodes,slots});
  tensor(out.values,device,at::kFloat,{nodes,slots});tensor(out.connected,device,at::kBool,{nodes,slots});tensor(out.chunks,device,at::kLong,{1});
  tensor(t.state.metadata,device,at::kLong,{capacity,13});tensor(gradient,device,at::kFloat,{capacity,d});
  tensor(connected,device,at::kBool,{capacity});tensor(stage_count,device,at::kLong,{1});tensor(stage_range,device,at::kLong,{2});
  tensor(error,device,at::kInt,{1});tensor(scale_partials,device,at::kFloat,{t.fiber_values.size(0),d});
  tensor(messages,device,at::kFloat,{links.messages.size(0),d});
  tensor(t.fiber_values,device,at::kFloat,{t.fiber_values.size(0),d});
  tensor(t.sources,device,at::kLong,{t.sources.size(0),2});tensor(t.source_scales,device,at::kFloat,{t.sources.size(0)});
  tensor(links.messages,device,at::kLong,{links.messages.size(0),4});
  tensor(links.consumer_head,device,at::kLong,{capacity});
  tensor(links.consumer_next,device,at::kLong,{t.fiber_values.size(0)});
  auto longs=a.kinds.options(),floats=a.weights.options();
  auto ids=at::empty({chunk},longs),row_nodes=at::empty_like(ids),owners=at::empty_like(ids);
  auto message_ids=at::empty({chunk,slots},longs),cursor=at::empty({1},longs),owner_count=at::empty_like(cursor),branch=at::empty_like(error);
  auto logits=at::empty({chunk,slots},floats),prob=at::empty_like(logits),mass=at::empty_like(logits),slope=at::empty_like(logits);
  auto dot=at::empty_like(logits),product=at::empty_like(logits),dc=at::empty_like(logits);
  auto total=at::empty({chunk,1},floats),center=at::empty_like(total),negative=at::empty_like(total);
  auto minus=at::full({1},-1.f,floats),packed=at::empty({chunk,1,d},floats);
  auto weighted=at::empty({chunk,slots,d},floats),vectors=at::empty_like(weighted);
  std::set<int64_t> groups;for(const auto& node:t.graph->nodes)if(auto kind=aggregate_kind(node.aggregation))groups.insert(kind);
  const auto values=out.values,flags=out.connected,chunks=out.chunks;
  auto payload=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_aggregate_vjp_payload)(32,stream,
      ptr(ids),ptr(message_ids),ptr(links.messages),ptr(t.source_scales),ptr(t.fiber_values),ptr(gradient),ptr(packed),ptr(weighted),
      ptr(prob),ptr(messages),ptr(scale_partials),ptr(error),chunk,slots,d,mode),"packed Aggregate VJP payload");},
      {ids,message_ids,links.messages,t.source_scales,t.fiber_values,gradient,packed,weighted,prob,messages,scale_partials,error});
  };
  for(auto kind:groups) {
    p.zero(cursor);auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_aggregate_vjp_plan)(1,stream,
      ptr(t.state.metadata),ptr(stage_count),ptr(stage_range),ptr(connected),ptr(links.consumer_head),ptr(links.consumer_next),ptr(links.messages),
      ptr(t.sources),ptr(a.kinds),ptr(a.lengths),ptr(a.weights),ptr(cursor),ptr(ids),ptr(row_nodes),ptr(message_ids),ptr(owners),ptr(owner_count),
      ptr(logits),ptr(prob),ptr(flags),ptr(branch),ptr(chunks),ptr(error),capacity,t.fiber_values.size(0),t.sources.size(0),nodes,slots,chunk,kind),
      "pack connected Aggregate VJP domains");},
      {t.state.metadata,stage_count,stage_range,connected,links.consumer_head,links.consumer_next,links.messages,t.sources,
       a.kinds,a.lengths,a.weights,cursor,ids,row_nodes,message_ids,owners,owner_count,logits,prob,flags,branch,chunks,error});
    p.branch(branch,{done,body});p.mark(body);payload(0);
    if(kind==2){p.softplus(logits,mass);p.sum(mass,1,true,total);p.divide(mass,total,prob);}
    else if(kind>2)p.softmax(logits,1,prob);
    payload(1);
    if(kind>=2) {
      p.multiply(weighted,packed,vectors);p.sum(vectors,2,false,dot);
      p.multiply(dot,prob,product);p.sum(product,1,true,center);p.multiply(center,minus,negative);p.add(dot,negative);
      if(kind==2){p.divide(dot,total,product);p.sigmoid(logits,slope);p.multiply(product,slope,dc);}
      else p.multiply(dot,prob,dc);
      p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_aggregate_vjp_reduce)(32,stream,
        ptr(row_nodes),ptr(owners),ptr(owner_count),ptr(dc),ptr(values),ptr(error),chunk,slots),"ordered Aggregate coefficient partials");},
        {row_nodes,owners,owner_count,dc,values,error});
    }
    p.branch(branch,{head});p.mark(done);
  }
}
} // namespace tide::device_online
