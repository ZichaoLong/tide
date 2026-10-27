#include "weighted_kv_query_internal.h"
#include "device_launch_tide_weighted_query_plan.h"
namespace tide::device_online {
using namespace kv_query_detail;
WeightedKVQueryTape append_weighted_kv_query(DeviceProgram& p,const weighted::KVConfig& c,const WeightedRawState& raw,
    const at::Tensor& time,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& present,
    const WeightedKVQueryParameters& w,const at::Tensor& error,int64_t available){
  c.validate();if(c.memory_read!="qry"||!raw.floating.defined()||raw.floating.dim()!=2||!raw.integer.defined()||raw.integer.dim()!=2
    ||raw.integer.size(1)<=4||!w.query.defined()||w.query.dim()!=3||w.query.size(0)<1)
    throw std::invalid_argument("KV qry needs explicit query configuration and raw cache");
  const auto device=raw.floating.device();const auto rows=raw.floating.size(0),capacity=raw.integer.size(1)-4,d=c.width,ds=c.summary_width,banks=w.query.size(0);
  check(input,device,at::kFloat,{rows,d});check(owners,device,at::kLong,{rows});
  for(const auto& x:{w.query,w.key,w.value})check(x,device,at::kFloat,{banks,ds,d});
  budget(rows*(256.L*capacity*(static_cast<long double>(d)*ds+d+ds+1)+256)
    +(raw.floating.scalar_type()==at::kHalf?4.L*raw.floating.numel():0),available);
  auto window=append_weighted_kv_window(p,c,raw,time,present,error,available);
  auto clean=at::zeros_like(input),query_ids=at::zeros_like(owners),entry_ids=at::zeros({rows*capacity},owners.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_query_plan)(1,stream,
    ptr(window.weights),ptr(window.lengths),ptr(window.present),ptr(owners),ptr(input),ptr(clean),ptr(query_ids),ptr(entry_ids),ptr(error),rows,capacity,d,banks),
    "KV qry actual parameter/input use");},{window.weights,window.lengths,window.present,owners,input,clean,query_ids,entry_ids,error});
  auto query=append_weighted_projection(p,clean,query_ids,{w.query,{}},c.epsilon,error,available);
  auto key=append_weighted_projection(p,window.keys.reshape({rows*capacity,d}),entry_ids,{w.key,{}},std::nullopt,error,available);
  auto value=append_weighted_projection(p,window.values.reshape({rows*capacity,d}),entry_ids,{w.value,{}},std::nullopt,error,available);
  auto rotary=append_weighted_rotary(p,key.output.reshape({rows,capacity,ds}),window.ages,c.position_scale,c.read_frequencies,error,available);
  auto dots=at::zeros({rows,capacity},input.options()),logits=at::zeros_like(dots),scale=at::full({1},1/std::sqrt(ds),input.options());
  p.batch_matmul(rotary.output,query.output.unsqueeze(2),dots.unsqueeze(2));p.multiply(dots,scale,logits);
  std::vector<int64_t> duplicated;for(int64_t r=0;r<rows;++r){duplicated.push_back(r);duplicated.push_back(r);}
  auto ids=at::tensor(duplicated,at::kLong).to(device);
  auto pool=append_weighted_pool(p,window.weights.reshape({2*rows,capacity}),repeat(p,value.output.reshape({rows,capacity,ds}),ids),
    repeat(p,logits,ids),repeat(p,window.lengths,ids),repeat(p,window.present,ids),error,available);
  return {window,query,key,value,rotary,pool,scale,pool.output.reshape({rows,2,ds}),pool.total.reshape({rows,2})};
}
}
