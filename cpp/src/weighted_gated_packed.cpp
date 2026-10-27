// GDN/SSD §2/§4 and causal-convolution §1–3: input-only coefficients.
#include "weighted_packed.h"
#include "tide/weighted_convolution.h"

namespace tide::weighted::packed {
using detail::weight;
namespace {
std::pair<Tensor,std::vector<std::map<std::string,Tensor>>> convolve(const NodeWeights& w,
    const std::vector<State>& old,const Writes& plan,const Tensor& raw,const RecurrentConfig& c){
  detail::require(at::isfinite(raw).all().item<bool>(),"invalid convolution query value");
  auto filters=weight(w,"kernel.conv.weight");
  auto base=weight(w,"kernel.conv.bias")+filters[0]*raw;
  auto taps=c.conv_stride*at::arange(1,c.conv_size,plan.times.options());
  std::vector<Tensor> rows;std::vector<std::map<std::string,Tensor>> caches;
  for(size_t i=0;i<old.size();++i){auto a=plan.offsets[i],b=plan.offsets[i+1];if(a==b)continue;
    validate_cache({old[i].slots.at("conv_times"),old[i].slots.at("conv_values")},c.conv_width(),(c.conv_size-1)*c.conv_stride);
    detail::require((old[i].slots.at("conv_times")<plan.times[a]).all().item<bool>(),"convolution writes require increasing logical time");
    auto now=plan.times.slice(0,a,b);
    auto times=at::cat({old[i].slots.at("conv_times"),now});
    auto values=at::cat({old[i].slots.at("conv_values"),plan.receive.slice(0,a,b).unsqueeze(-1)*raw.slice(0,a,b)});
    if(c.conv_size>1){
      auto queries=now.unsqueeze(-1)-taps,indices=at::searchsorted(times,queries);
      auto safe=indices.clamp_max(times.numel()-1);
      auto present=(indices<times.numel())&(times.index({safe})==queries)&(queries>=0);
      auto history=values.index({safe})*present.unsqueeze(-1);
      rows.push_back(base.slice(0,a,b)+(filters.slice(0,1)*history).sum(-2));
    }else rows.push_back(base.slice(0,a,b));
    for(Index j=a;j<b;++j){
      auto visible=(times>=plan.times[j]-(c.conv_size-1)*c.conv_stride)&(times<=plan.times[j]);
      caches.push_back({{"conv_times",times.index({visible})},{"conv_values",values.index({visible})}});
    }
  }
  return {at::cat(rows),std::move(caches)};
}
}
Coefficients gated_coefficients(const NodeWeights& w,const std::vector<State>& old,const Writes& plan,
    const Tensor& x,const Tensor& gap,const RecurrentConfig& c){
  auto raw=project(x,weight(w,"kernel.projection"));Tensor dt;
  if(c.memory_family=="ssd"){
    auto parts=raw.split_with_sizes({c.heads*c.value_width,c.conv_width(),c.heads},-1);
    raw=parts[1];dt=at::softplus(parts[2]+weight(w,"kernel.delta_bias"));
  }
  auto [conv,caches]=convolve(w,old,plan,raw,c);auto projected=at::silu(conv);
  auto rates=c.matrix_rates;if(rates.empty())rates.resize(c.heads,0.);
  auto decay=at::exp(-gap.unsqueeze(-1)*at::tensor(rates,x.options()));
  if(c.memory_family=="gated_delta"){
    auto parts=projected.reshape({-1,c.heads,2*c.key_width+c.value_width}).split_with_sizes({c.key_width,c.key_width,c.value_width},-1);
    auto key=parts[1]/at::sqrt(c.qk_epsilon+parts[1].square().sum(-1,true)),value=parts[2];
    dt=at::softplus(project(x,weight(w,"kernel.delta"))+weight(w,"kernel.delta_bias"));
    auto gain=plan.receive.unsqueeze(-1)*at::sigmoid(project(x,weight(w,"kernel.write"))+weight(w,"kernel.write_bias"));
    auto retain=at::exp(-plan.receive.unsqueeze(-1)*weight(w,"kernel.log_rate").exp()*dt);
    auto a=at::eye(c.key_width,x.options())-gain.unsqueeze(-1).unsqueeze(-1)*key.unsqueeze(-1)*key.unsqueeze(-2);
    a=a*(decay*retain).unsqueeze(-1).unsqueeze(-1);
    auto b=gain.unsqueeze(-1).unsqueeze(-1)*value.unsqueeze(-1)*key.unsqueeze(-2);
    return {a,b,true,std::move(caches)};
  }
  auto parts=projected.split_with_sizes({c.heads*c.value_width,c.groups*c.key_width,c.groups*c.key_width},-1);
  auto value=parts[0].reshape({-1,c.heads,c.value_width});
  auto key=parts[1].reshape({-1,c.groups,c.key_width}).repeat_interleave(c.heads/c.groups,1);
  auto step=plan.receive.unsqueeze(-1)*dt;
  auto a=(decay*at::exp(-weight(w,"kernel.log_rate").exp()*step)).unsqueeze(-1).unsqueeze(-1);
  auto b=step.unsqueeze(-1).unsqueeze(-1)*value.unsqueeze(-1)*key.unsqueeze(-2);
  return {a,b,false,std::move(caches)};
}
}
