#include "block_kv_pool.h"
#include "weighted_storage_budget.h"
#include "device_launch_tide_block_kv_pool_prepare.h"
#include "device_launch_tide_block_kv_pool_scale.h"
#include "device_launch_tide_block_kv_pool_roots.h"
#include "device_launch_tide_block_kv_pool_finish.h"
#include "device_launch_tide_block_kv_pool_center.h"
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,const at::Tensor& e,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=e.device()||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid shared KV pool tensor");
}
void admit(long double bytes,int64_t budget){if(budget<1||bytes>budget)throw std::invalid_argument("shared KV pool tensor budget exceeded");}
at::Tensor transpose(DeviceProgram& p,const at::Tensor& x){auto y=at::zeros({x.size(0),x.size(2),x.size(1)},x.options());p.permute(x,{0,2,1},y);return y;}
}
std::vector<at::Tensor> block_kv_pool_tensors(const BlockKVPoolTape& t){
  return {t.weights,t.values,t.visible,t.present,t.meta,t.probabilities,t.coefficients,t.total,t.mean,t.output,t.centered,t.anchor};
}
BlockKVPoolTape append_block_kv_pool(DeviceProgram& p,const at::Tensor& weights,const at::Tensor& values,
    const at::Tensor& logits,const at::Tensor& visible,const at::Tensor& present,const at::Tensor& error,int64_t budget){
  if(!weights.defined()||weights.dim()!=3||!values.defined()||values.dim()!=3)throw std::invalid_argument("shared KV pool needs O,Q,K and O,K,D");
  const auto o=weights.size(0),q=weights.size(1),k=weights.size(2),d=values.size(2),rows=o*q;const bool attention=logits.defined();
  if(o<1||q<1||k<1||d<1)throw std::invalid_argument("empty shared KV pool extent");
  check(weights,error,at::kFloat,{o,q,k});check(values,error,at::kFloat,{o,k,d});check(visible,error,at::kBool,{o,q,k});
  check(present,error,at::kBool,{o,q});check(error,error,at::kInt,{1});if(attention)check(logits,error,at::kFloat,{o,q,k});
  const WeightedStorageIncrement storage(p,{weights,values,logits,visible,present,error});
  admit(rows*(16.L*k+8.L*d+24)+8.L*o*k*d+4.L*o*d,budget);
  BlockKVPoolTape t;t.values=at::zeros_like(values);t.centered=at::zeros_like(values);t.anchor=at::zeros({o,1,d},values.options());
  t.visible=visible;t.present=present;t.attention=attention;
  t.weights=at::zeros_like(weights);t.meta=at::zeros({o,q,2},weights.options().dtype(at::kLong));
  auto scores=at::zeros_like(weights);t.probabilities=at::zeros_like(weights);t.coefficients=at::zeros_like(weights);
  t.total=at::zeros({o,q},weights.options());t.mean=at::zeros({o,q,d},weights.options());t.output=at::zeros_like(t.mean);
  const auto input_logits=attention?logits:scores;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_pool_prepare)(1,stream,
    ptr(weights),ptr(input_logits),ptr(visible),ptr(present),ptr(t.weights),ptr(scores),ptr(t.meta),ptr(error),rows,k,attention),
    "shared KV pool support and weighted logits");},{weights,input_logits,visible,present,t.weights,scores,t.meta,error});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_pool_center)(1,stream,
    ptr(values),ptr(t.weights),ptr(visible),ptr(present),ptr(t.values),ptr(t.centered),ptr(t.anchor),ptr(error),o,q,k,d,attention),
    "shared KV values and stable reverse anchor");},{values,t.weights,visible,present,t.values,t.centered,t.anchor,error});
  p.sum(t.weights,2,false,t.total);if(attention)p.softmax(scores.reshape({rows,k}),1,t.probabilities.reshape({rows,k}));
  auto amplitude=at::zeros({o,q,1},weights.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_pool_scale)(1,stream,
    ptr(t.weights),ptr(t.total),ptr(t.probabilities),ptr(t.meta),ptr(t.coefficients),ptr(amplitude),ptr(error),rows,k,attention),
    "shared KV pool original amplitude");},{t.weights,t.total,t.probabilities,t.meta,t.coefficients,amplitude,error});
  p.batch_matmul(t.coefficients,t.values,t.mean);p.multiply(t.mean,amplitude,t.output);admit(storage.bytes(),budget);return t;
}
BlockKVPoolGradient append_block_kv_pool_vjp(DeviceProgram& p,const BlockKVPoolTape& t,const at::Tensor& vector_bar,
    const at::Tensor& total_bar,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  const auto o=t.weights.size(0),q=t.weights.size(1),k=t.weights.size(2),d=t.values.size(2),rows=o*q;
  check(vector_bar,error,at::kFloat,{o,q,d});check(total_bar,error,at::kFloat,{o,q});check(roots,error,at::kBool,{o,q,2});
  auto borrowed=block_kv_pool_tensors(t);borrowed.insert(borrowed.end(),{vector_bar,total_bar,roots,error});
  const WeightedStorageIncrement storage(p,std::move(borrowed));
  admit(rows*(16.L*k+16.L*d+19)+8.L*o*k*d+4.L*o*d,budget);
  auto clean=at::zeros_like(vector_bar),scaled=at::zeros_like(vector_bar),stat=at::zeros_like(total_bar);
  auto connected=at::zeros({o,q,3},roots.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_pool_roots)(1,stream,
    ptr(t.meta),ptr(t.total),ptr(vector_bar),ptr(total_bar),ptr(roots),ptr(clean),ptr(scaled),ptr(stat),ptr(connected),ptr(error),rows,d,t.attention),
    "shared KV pool independent output and total roots");},{t.meta,t.total,vector_bar,total_bar,roots,clean,scaled,stat,connected,error});
  auto value_bar=at::zeros_like(t.values);p.batch_matmul(transpose(p,t.coefficients),scaled,value_bar);
  auto dots=at::zeros_like(t.weights);p.batch_matmul(clean,transpose(p,t.centered),dots);
  auto products=at::zeros_like(t.mean),mean_dot=at::zeros_like(t.total);p.multiply(clean,t.mean,products);p.sum(products,2,false,mean_dot);
  auto center_mean=at::zeros_like(t.mean),center_dot=at::zeros_like(t.total),anchor_dot=at::zeros_like(t.total);
  p.batch_matmul(t.coefficients,t.centered,center_mean);p.multiply(clean,center_mean,products);p.sum(products,2,false,center_dot);
  p.batch_matmul(clean,transpose(p,t.anchor),anchor_dot.unsqueeze(2));
  auto weight_bar=at::zeros_like(t.weights),logit_bar=at::zeros_like(t.weights);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_pool_finish)(1,stream,
    ptr(t.weights),ptr(t.probabilities),ptr(t.total),ptr(dots),ptr(mean_dot),ptr(center_dot),ptr(anchor_dot),ptr(stat),ptr(t.visible),ptr(connected),
    ptr(weight_bar),ptr(logit_bar),ptr(error),rows,k,t.attention),"shared KV pool scalar transpose");},
    {t.weights,t.probabilities,t.total,dots,mean_dot,center_dot,anchor_dot,stat,t.visible,connected,weight_bar,logit_bar,error});
  admit(storage.bytes(),budget);return {weight_bar,value_bar,logit_bar,connected};
}
}
