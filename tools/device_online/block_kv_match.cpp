#include "block_kv_match.h"
#include "block_kv_bank_internal.h"
#include "device_launch_tide_block_kv_match_phase.h"
#include "device_launch_tide_block_kv_match.h"
#include <cmath>
namespace tide::device_online {
using namespace block_bank_detail;
std::vector<at::Tensor> block_kv_match_tensors(const BlockKVMatchTape& t){return {t.query,t.keys,t.ages,t.visible,t.phase,t.cosine,t.sine,t.scale,t.logits};}
BlockKVMatchTape append_block_kv_match(DeviceProgram& p,const at::Tensor& query,const at::Tensor& keys,
    const at::Tensor& ages,const at::Tensor& visible,double position_scale,const std::vector<double>& frequencies,
    double slope,const at::Tensor& error,int64_t budget){
  if(query.dim()!=3||keys.dim()!=3||query.size(0)!=keys.size(0)||query.size(2)!=keys.size(2)||!(position_scale>0))
    throw std::invalid_argument("KV bank match shape/position scale");
  const auto o=query.size(0),q=query.size(1),k=keys.size(1),d=query.size(2),pairs=int64_t(frequencies.size());
  if(pairs*2>d||!std::isfinite(float(position_scale))||float(position_scale)<=0||!std::isfinite(float(slope)))throw std::invalid_argument("KV bank match position configuration");
  for(auto f:frequencies)if(!std::isfinite(float(f)))throw std::invalid_argument("KV bank match frequency");
  check(query,error,at::kFloat,{o,q,d});check(keys,error,at::kFloat,{o,k,d});check(ages,error,at::kLong,{o,q,k});check(visible,error,at::kBool,{o,q,k});
  const WeightedStorageIncrement storage(p,{query,keys,ages,visible,error});
  const auto phase_pairs=std::max<int64_t>(1,pairs);
  admit(4.L*o*q*k*(3*phase_pairs+2+int64_t(!pairs))+(pairs?0:4.L*o*k*d)+4.L*phase_pairs+4,budget);
  BlockKVMatchTape t;t.query=query;t.keys=keys;t.ages=ages;t.visible=visible;t.pairs=pairs;
  t.logits=at::zeros({o,q,k},query.options());t.scale=at::full({1},1/std::sqrt(d),query.options());
  t.phase=at::zeros({o,q,k,std::max<int64_t>(1,pairs)},query.options());t.cosine=at::zeros_like(t.phase);t.sine=at::zeros_like(t.phase);
  auto bias=at::zeros_like(t.logits),freq=at::tensor(frequencies,at::kFloat).to(error.device());if(!pairs)freq=at::zeros({1},query.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_match_phase)(1,stream,ptr(ages),ptr(visible),ptr(freq),
    ptr(t.phase),ptr(bias),ptr(error),o*q*k,pairs,float(position_scale),float(slope)),"KV bank original-time relative phase");},{ages,visible,freq,t.phase,bias,error});
  if(!pairs){auto dots=at::zeros_like(t.logits);p.batch_matmul(query,transpose(p,keys),dots);p.multiply(dots,t.scale,t.logits);}
  else{p.cos(t.phase,t.cosine);p.sin(t.phase,t.sine);
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_match)(1,stream,ptr(query),ptr(keys),ptr(visible),ptr(t.cosine),ptr(t.sine),
      ptr(t.logits),ptr(t.logits),ptr(t.logits),ptr(t.logits),ptr(error),o,q,k,d,pairs,0,float(1/std::sqrt(d))),"KV bank position match without expanded key tensors");},
      {query,keys,visible,t.cosine,t.sine,t.logits,error});}
  p.add(t.logits,bias);admit(storage.bytes(),budget);return t;
}
BlockKVMatchGradient append_block_kv_match_vjp(DeviceProgram& p,const BlockKVMatchTape& t,const at::Tensor& bar,const at::Tensor& error,int64_t budget){
  const auto o=t.query.size(0),q=t.query.size(1),k=t.keys.size(1),d=t.keys.size(2);check(bar,error,at::kFloat,{o,q,k});
  auto borrowed=block_kv_match_tensors(t);borrowed.insert(borrowed.end(),{bar,error});const WeightedStorageIncrement storage(p,std::move(borrowed));
  admit(8.L*o*((q+k)*d+q*k),budget);BlockKVMatchGradient out{at::zeros_like(t.query),at::zeros_like(t.keys)};
  if(!t.pairs){auto scaled=at::zeros_like(bar);p.multiply(bar,t.scale,scaled);p.batch_matmul(scaled,t.keys,out.query);p.batch_matmul(transpose(p,scaled),t.query,out.keys);}
  else p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_match)(1,stream,ptr(t.query),ptr(t.keys),ptr(t.visible),ptr(t.cosine),ptr(t.sine),
    ptr(t.logits),ptr(bar),ptr(out.query),ptr(out.keys),ptr(error),o,q,k,d,t.pairs,1,float(1/std::sqrt(d))),"KV bank relative-position match transpose");},
    {t.query,t.keys,t.visible,t.cosine,t.sine,t.logits,bar,out.query,out.keys,error});
  admit(storage.bytes(),budget);return out;
}
}
