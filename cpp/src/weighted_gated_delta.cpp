// gated-delta §2–3: head order (Q,K,V), epsilon-L2 and post-RMS output gate.
#include "weighted_recurrent_detail.h"

namespace tide::weighted::recurrent {
namespace {
struct Features {Tensor raw,q,k,value;};
Features features(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  auto raw=at::matmul(weight(w,"kernel.projection"),normalize(u,c.epsilon));
  auto rows=at::silu(convolve(cache(s),time,raw,weight(w,"kernel.conv.weight"),weight(w,"kernel.conv.bias"),c.conv_stride)).reshape({c.heads,-1});
  auto q=rows.slice(1,0,c.key_width),k=rows.slice(1,c.key_width,2*c.key_width),v=rows.slice(1,2*c.key_width);
  q=q/at::sqrt(c.qk_epsilon+q.square().sum(-1,true))/std::sqrt(c.key_width);
  k=k/at::sqrt(c.qk_epsilon+k.square().sum(-1,true));return {raw,q,k,v};
}
}
std::pair<Tensor,Tensor> gdn_update(const NodeWeights& w,const State& s,Index time,const Tensor& u,const Tensor& receive,const Tensor& H,const RecurrentConfig& c){
  auto f=features(w,s,time,u,c);auto x=normalize(u,c.epsilon);
  auto delta=at::softplus(at::matmul(weight(w,"kernel.delta"),x)+weight(w,"kernel.delta_bias"));
  auto g=at::exp(-receive*weight(w,"kernel.log_rate").exp()*delta);
  auto beta=receive*at::sigmoid(at::matmul(weight(w,"kernel.write"),x)+weight(w,"kernel.write_bias"));
  auto old=g.reshape({-1,1,1})*H;
  auto error=f.value-at::matmul(old,f.k.unsqueeze(-1)).squeeze(-1);
  return {old+beta.reshape({-1,1,1})*error.unsqueeze(-1)*f.k.unsqueeze(-2),f.raw};
}
Tensor gdn_read(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  auto f=features(w,s,time,u,c);return at::matmul(effective(s,time,c)[0],f.q.unsqueeze(-1)).squeeze(-1);
}
Tensor gdn_mix(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  auto read=gdn_read(w,s,time,u,c);
  auto rms=weight(w,"full_program.content.rho")*read/at::sqrt(c.output_epsilon+read.square().mean(-1,true));
  auto gate=at::silu(at::matmul(weight(w,"full_program.content.output_gate"),normalize(u,c.epsilon)));
  return at::matmul(weight(w,"full_program.content.out"),rms.flatten()*gate);
}
void validate_gdn(const NodeWeights& w,const RecurrentConfig& c){
  shape(w,"kernel.projection",{c.conv_width(),c.width});
  for(const std::string name:{"delta","write"})shape(w,"kernel."+name,{c.heads,c.width});
  for(const std::string name:{"delta_bias","write_bias","log_rate"})shape(w,"kernel."+name,{c.heads});
}
}
