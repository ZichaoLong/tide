// affine-delta §2–3. Independent LibTorch local functions.
#include "weighted_recurrent_detail.h"

namespace tide::weighted::recurrent {
Tensor basic_update(const NodeWeights& w,const Tensor& u,const Tensor& receive,const Tensor& H,const RecurrentConfig& c){
  auto x=normalize(u,c.epsilon),key=at::matmul(weight(w,"kernel.key"),x);
  key=key/key.norm().clamp_min(1);auto value=at::matmul(weight(w,"kernel.value"),x);
  auto g=at::exp(-receive*at::softplus(at::dot(weight(w,"kernel.retain"),x)+weight(w,"kernel.retain_bias")));
  auto old=g*H;
  if(c.memory_family=="affine")return old+receive*value.unsqueeze(-1)*key;
  auto beta=receive*at::sigmoid(at::dot(weight(w,"kernel.write"),x)+weight(w,"kernel.write_bias"));
  return old+beta*(value-at::matmul(old,key)).unsqueeze(-1)*key;
}
Tensor basic_read(const NodeWeights& w,const Tensor& u,const Tensor& H,const RecurrentConfig& c){
  auto query=at::matmul(weight(w,"kernel.read_query"),normalize(u,c.epsilon));
  return at::matmul(weight(w,"kernel.read_out"),at::matmul(H,query).unsqueeze(-1)).squeeze(-1);
}
Tensor basic_mix(const NodeWeights& w,const Tensor& u,const Tensor& H,const RecurrentConfig& c){
  auto query=at::matmul(weight(w,"full_program.content.query"),normalize(u,c.epsilon));
  return at::matmul(weight(w,"full_program.content.out"),at::matmul(H,query).unsqueeze(-1)).squeeze(-1).sum(0);
}
void validate_basic(const NodeWeights& w,const RecurrentConfig& c){
  shape(w,"kernel.key",{c.key_width,c.width});shape(w,"kernel.value",{c.value_width,c.width});
  shape(w,"kernel.retain",{c.width});shape(w,"kernel.retain_bias",{});
  if(c.memory_family=="delta"){shape(w,"kernel.write",{c.width});shape(w,"kernel.write_bias",{});}
  if(c.memory_read=="state"){
    shape(w,"kernel.read_query",{c.key_width,c.width});shape(w,"kernel.read_out",{2,c.summary_width,c.value_width});
  }
}
}
