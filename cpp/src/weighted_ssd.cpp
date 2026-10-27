// SSD §2–3: grouped B/C are not normalized; discretization uses d, not ZOH.
#include "weighted_recurrent_detail.h"

namespace tide::weighted::recurrent {
namespace {
struct Features {Tensor raw,gate,value,b,query,delta;};
Features features(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  const auto inner=c.heads*c.value_width,bc=c.groups*c.key_width;
  auto all=at::matmul(weight(w,"kernel.projection"),normalize(u,c.epsilon));
  auto gate=all.slice(0,0,inner),raw=all.slice(0,inner,inner+c.conv_width());
  auto delta=at::softplus(all.slice(0,inner+c.conv_width())+weight(w,"kernel.delta_bias"));
  auto features=at::silu(convolve(cache(s),time,raw,weight(w,"kernel.conv.weight"),weight(w,"kernel.conv.bias"),c.conv_stride));
  auto value=features.slice(0,0,inner).reshape({c.heads,c.value_width});
  auto b=features.slice(0,inner,inner+bc).reshape({c.groups,c.key_width}).repeat_interleave(c.heads/c.groups,0);
  auto q=features.slice(0,inner+bc).reshape({c.groups,c.key_width}).repeat_interleave(c.heads/c.groups,0);
  return {raw,gate,value,b,q,delta};
}
}
std::pair<Tensor,Tensor> ssd_update(const NodeWeights& w,const State& s,Index time,const Tensor& u,const Tensor& receive,const Tensor& H,const RecurrentConfig& c){
  auto f=features(w,s,time,u,c);auto step=receive*f.delta;
  auto g=at::exp(-weight(w,"kernel.log_rate").exp()*step);
  return {g.reshape({-1,1,1})*H+step.reshape({-1,1,1})*f.value.unsqueeze(-1)*f.b.unsqueeze(-2),f.raw};
}
Tensor ssd_read(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  auto f=features(w,s,time,u,c);return at::matmul(effective(s,time,c)[0],f.query.unsqueeze(-1)).squeeze(-1);
}
Tensor ssd_mix(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  auto f=features(w,s,time,u,c);
  auto read=at::matmul(effective(s,time,c)[0],f.query.unsqueeze(-1)).squeeze(-1);
  auto y=(read+weight(w,"full_program.content.direct").unsqueeze(-1)*f.value).flatten()*at::silu(f.gate);
  return at::matmul(weight(w,"full_program.content.out"),scaled_rms(y,weight(w,"full_program.content.rho"),c.output_epsilon));
}
void validate_ssd(const NodeWeights& w,const RecurrentConfig& c){
  shape(w,"kernel.projection",{c.heads*c.value_width+c.conv_width()+c.heads,c.width});
  shape(w,"kernel.delta_bias",{c.heads});shape(w,"kernel.log_rate",{c.heads});
}
}
