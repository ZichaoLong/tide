#include "weighted_kv_full_internal.h"
#include "device_launch_tide_weighted_full_plan.h"
#include "device_launch_tide_weighted_full_bias.h"
#include "device_launch_tide_weighted_query_plan.h"
namespace tide::device_online {
using namespace kv_full_detail;
WeightedKVFullTape append_weighted_kv_full(DeviceProgram& p,const weighted::KVConfig& c,const WeightedRawState& raw,
    const at::Tensor& time,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& active,
    const WeightedKVFullParameters& w,const at::Tensor& error,int64_t available){
  c.validate();if(!input.defined()||input.dim()!=2||input.size(0)<1||!raw.integer.defined()||raw.integer.dim()!=2||raw.integer.size(1)<=4
    ||!w.query.defined()||w.query.dim()!=3||w.query.size(0)<1)
    throw std::invalid_argument("KV Full requires explicit rows and raw snapshot");
  const auto device=input.device();const auto rows=input.size(0),d=c.width,heads=c.heads,dh=d/heads,capacity=raw.integer.size(1)-4,banks=w.query.size(0);
  const auto initial_bytes=p.retained_tensor_bytes();
  check(input,device,at::kFloat,{rows,d});check(owners,device,at::kLong,{rows});check(active,device,at::kBool,{rows});check(w.query,device,at::kFloat,{banks,d,d});
  check(w.residual.out,device,at::kFloat,{banks,d,d});check(w.residual.gate,device,at::kFloat,{banks,c.ffn_width,d});
  check(w.residual.up,device,at::kFloat,{banks,c.ffn_width,d});check(w.residual.down,device,at::kFloat,{banks,d,c.ffn_width});
  check(error,device,at::kInt,{1});budget(rows*(256.L*(capacity*(d+c.summary_width+heads+1.L)+static_cast<long double>(d)*d+3.L*d*c.ffn_width+d)+256)
    +(raw.floating.scalar_type()==at::kHalf?4.L*raw.floating.numel():0),available);
  for(auto slope:c.distance_slopes)if(!std::isfinite(float(slope)))throw std::invalid_argument("unrepresentable Full distance slope");
  auto window=append_weighted_kv_window(p,c,raw,time,active,error,available);
  auto full_owners=at::zeros_like(owners),qowners=at::zeros_like(owners),entry_owners=at::zeros({rows*capacity},owners.options()),clean=at::zeros_like(input);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_full_plan)(1,stream,
    ptr(active),ptr(owners),ptr(full_owners),ptr(error),rows,banks),"KV Full actual selected parameter owners");},{active,owners,full_owners,error});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_query_plan)(1,stream,
    ptr(window.weights),ptr(window.lengths),ptr(window.present),ptr(full_owners),ptr(input),ptr(clean),ptr(qowners),ptr(entry_owners),ptr(error),rows,capacity,d,banks),
    "KV Full positive-memory query use");},{window.weights,window.lengths,window.present,full_owners,input,clean,qowners,entry_owners,error});
  auto query=append_weighted_projection(p,clean,qowners,{w.query,{}},c.epsilon,error,available);
  auto keys=split(p,window.keys,heads),values=split(p,window.values,heads),queries=split(p,query.output,heads);
  auto weights=at::zeros({rows,1,capacity},input.options()),ids=at::ones({1},owners.options());p.index_select(window.weights,1,ids,weights);
  auto scale=at::full({1},1/std::sqrt(dh),input.options()),reads=at::zeros({heads,rows,1,dh},input.options());
  std::vector<WeightedRotaryTape> rotary;std::vector<WeightedPoolTape> pool;
  for(int64_t h=0;h<heads;++h){const auto frequencies=c.full_frequencies.empty()?std::vector<double>{}:c.full_frequencies[h];
    const float slope=c.distance_slopes.empty()?0:float(c.distance_slopes[h]);
    auto rot=append_weighted_rotary(p,keys[h],window.ages,c.position_scale,frequencies,error,available);
    auto dots=at::zeros({rows,capacity,1},input.options()),logits=at::zeros_like(dots),bias=at::zeros_like(dots);
    auto q=at::zeros({rows,dh,1},input.options());p.permute(queries[h],{0,2,1},q);p.batch_matmul(rot.output,q,dots);p.multiply(dots,scale,logits);
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_full_bias)(1,stream,
      ptr(window.ages),ptr(bias),ptr(error),rows*capacity,float(c.position_scale),slope),"KV Full original-age distance bias");},{window.ages,bias,error});p.add(logits,bias);
    auto read=append_weighted_pool(p,weights.reshape({rows,capacity}),values[h],logits.reshape({rows,capacity}),window.lengths,window.present,error,available);
    p.copy(reads[h].reshape({rows,dh}),read.output);rotary.push_back(rot);pool.push_back(read);
  }
  auto memory=merge(p,reads).reshape({rows,d});
  auto residual=append_weighted_residual(p,input,memory,full_owners,w.residual,c.epsilon,c.alpha_att,c.alpha_ff,error,available);
  budget(p.retained_tensor_bytes()-initial_bytes,available);
  return {window,query,rotary,pool,residual,queries,scale,residual.output};
}
}
