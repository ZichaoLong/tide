#include "attention_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_attention_reverse_plan.h"
#include "aclrtlaunch_tide_attention_reverse_pack.h"
#include "aclrtlaunch_tide_attention_reverse_weights.h"
#include "aclrtlaunch_tide_attention_reverse_reduce.h"
#include "aclrtlaunch_tide_attention_softmax.h"
#include "aclrtlaunch_tide_attention_merge.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <limits>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid packed attention VJP buffer");
}
}
AttentionVjp append_attention_vjp(CannProgram& p,const AttentionVjpInput& in,const at::Tensor& error,
    double scale,int64_t tile,int64_t budget) {
  if(at::GradMode::is_enabled()||in.query.dim()!=3||in.key.dim()!=4)
    throw std::invalid_argument("attention VJP requires explicit no-grad packed queries/KV");
  const auto device=in.query.device();
  const auto payload=in.query.scalar_type();const bool half=payload==at::kHalf;
  if(payload!=at::kFloat&&!half)throw std::invalid_argument("attention VJP requires FP32/FP16 forward payloads");
  const int64_t q=in.query.size(0),h=in.query.size(1),d=in.query.size(2),kh=in.key.size(1),k=in.key.size(2);
  if(device.type()!=c10::DeviceType::PrivateUse1||q<1||h<1||d<1||kh<1||h%kh||k<1
      ||k>std::numeric_limits<int64_t>::max()-256||tile<1||tile>256||!std::isfinite(scale)||scale<=0)
    throw std::invalid_argument("invalid packed attention VJP geometry");
  tensor(in.query,device,payload,{q,h,d});tensor(in.key,device,payload,{q,kh,k,d});
  tensor(in.value,device,payload,{q,kh,k,d});tensor(in.bias,device,at::kFloat,{q,k});
  tensor(in.cotangent,device,at::kFloat,{q,h,d});tensor(in.lengths,device,at::kLong,{q});
  tensor(in.connected,device,at::kBool,{q});tensor(error,device,at::kInt,{1});
  // Outputs plus all retained numerical/metadata scratch; original inputs and
  // the separately bounded CANN operator workspace are outside this reservation.
  const long double bytes=4.L*(2.L*in.key.numel()+in.bias.numel()+12.L*in.query.numel()
      +8.L*q*h*tile*d+10.L*q*h*tile+q*static_cast<long double>(tile)+8.L*q*h)+32.L*q+1024
      +2.L*half*(in.query.numel()+q*h*static_cast<long double>(tile)*(d+1));
  if(budget<1||bytes>budget)throw std::invalid_argument("attention VJP tensor budget exceeded");
  auto f=in.query.options().dtype(at::kFloat),l=in.lengths.options();
  AttentionVjp out{at::empty(in.query.sizes(),f),at::empty(in.key.sizes(),f),at::empty(in.value.sizes(),f),at::empty_like(in.bias),
    at::empty_like(in.connected),at::empty({1},l)};
  for(const auto& x:{out.query,out.key,out.value,out.bias,out.connected,out.tiles})p.zero(x);
  auto cursor=at::empty({2},l),valid=at::empty({q},l),branch=at::zeros_like(error);
  auto query=at::empty(in.query.sizes(),f),u=at::empty_like(query);
  auto key=at::empty({q,h,tile,d},f),value=at::empty_like(key);
  auto kt=at::empty({q,h,d,tile},f),vt=at::empty_like(kt),bias=at::empty({q,1,1,tile},f);
  auto scores=at::empty({q,h,1,tile},f),scaled=at::empty_like(scores),weights=at::empty_like(scores);
  auto dp=at::empty_like(scores),ds=at::empty_like(scores),ds_scaled=at::empty_like(scores);
  auto transposed=at::empty({q,h,tile,1},f),key_partial=at::empty_like(key),value_partial=at::empty_like(value);
  auto partial=at::empty_like(query),sum=at::empty_like(query),product=at::empty_like(query);
  auto correction=at::empty({q,h},f),normalization=at::empty({q*h,8},f),factor=at::full({},scale,f);
  auto forward_query=half?at::empty_like(in.query):query;
  auto forward_key=half?at::empty(kt.sizes(),in.query.options()):kt;
  auto forward_scores=half?at::empty(scores.sizes(),in.query.options()):scores;
  p.zero(sum);p.zero(normalization);
  auto plan=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_reverse_plan)(1,stream,
      ptr(in.lengths),ptr(in.connected),ptr(cursor),ptr(valid),ptr(branch),ptr(out.connected),ptr(out.tiles),ptr(error),q,k,tile,mode),
      "plan actual reverse attention tile");},{in.lengths,in.connected,cursor,valid,branch,out.connected,out.tiles,error});
  };
  auto pack=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_reverse_pack)(32,stream,
      ptr(in.query),ptr(in.key),ptr(in.value),ptr(in.cotangent),ptr(in.bias),ptr(out.connected),ptr(cursor),ptr(valid),
      ptr(query),ptr(u),ptr(key),ptr(value),ptr(bias),ptr(error),q,h,kh,d,k,tile,mode,int64_t(half)),"pack reverse attention and sanitize absent rows");},
      {in.query,in.key,in.value,in.cotangent,in.bias,out.connected,cursor,valid,query,u,key,value,bias,error});
  };
  auto logits=[&] {
    pack(1);p.permute(key,{0,1,3,2},kt);
    if(half)p.cast(kt,forward_key);
    p.batch_matmul(forward_query.reshape({q*h,1,d}),forward_key.reshape({q*h,d,tile}),forward_scores.reshape({q*h,1,tile}));
    if(half)p.cast(forward_scores,scores);
    p.multiply(scores,factor,scaled);p.add(scaled,bias);
  };
  auto merge=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_merge)(32,stream,
      ptr(partial),ptr(normalization),ptr(sum),ptr(error),q*h,d,mode),"merge reverse attention normalization");},
      {partial,normalization,sum,error});
  };
  plan(0);pack(0);
  if(half)p.cast(query,forward_query);
  const auto first=p.label(),body=p.label(),done=p.label();p.mark(first);plan(1);p.branch(branch,{done,body});p.mark(body);
  logits();
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_softmax)(32,stream,
    ptr(scaled),ptr(valid),ptr(normalization),ptr(weights),ptr(error),q,h,tile),"global reverse attention denominator");},
    {scaled,valid,normalization,weights,error});
  p.batch_matmul(weights.reshape({q*h,1,tile}),value.reshape({q*h,tile,d}),partial.reshape({q*h,1,d}));
  merge(0);p.branch(branch,{first});p.mark(done);merge(1);
  p.multiply(u,sum,product);p.sum(product,2,false,correction);
  // The softmax correction uses the complete attention output. Computing a
  // separate softmax/Jacobian per physical key tile changes the model.
  plan(2);const auto reverse=p.label(),reverse_body=p.label(),reverse_done=p.label();
  p.mark(reverse);plan(1);p.branch(branch,{reverse_done,reverse_body});p.mark(reverse_body);logits();
  p.permute(value,{0,1,3,2},vt);p.batch_matmul(u.reshape({q*h,1,d}),vt.reshape({q*h,d,tile}),dp.reshape({q*h,1,tile}));
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_reverse_weights)(32,stream,
    ptr(scaled),ptr(valid),ptr(normalization),ptr(dp),ptr(correction),ptr(weights),ptr(ds),ptr(error),q,h,tile),
    "globally normalized attention adjoint");},{scaled,valid,normalization,dp,correction,weights,ds,error});
  p.multiply(ds,factor,ds_scaled);
  p.batch_matmul(ds_scaled.reshape({q*h,1,tile}),key.reshape({q*h,tile,d}),partial.reshape({q*h,1,d}));p.add(out.query,partial);
  p.permute(ds_scaled,{0,1,3,2},transposed);
  p.batch_matmul(transposed.reshape({q*h,tile,1}),query.reshape({q*h,1,d}),key_partial.reshape({q*h,tile,d}));
  p.permute(weights,{0,1,3,2},transposed);
  p.batch_matmul(transposed.reshape({q*h,tile,1}),u.reshape({q*h,1,d}),value_partial.reshape({q*h,tile,d}));
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_reverse_reduce)(32,stream,
    ptr(key_partial),ptr(value_partial),ptr(ds),ptr(valid),ptr(cursor),ptr(out.key),ptr(out.value),ptr(out.bias),ptr(error),q,h,kh,d,k,tile),
    "reduce GQA key/value and log-bias adjoints");},{key_partial,value_partial,ds,valid,cursor,out.key,out.value,out.bias,error});
  p.branch(branch,{reverse});p.mark(reverse_done);
  // The downstream projection sees a rounded payload result; softmax's
  // Jacobian above must use the complete unrounded FP32 weighted sum.
  if(half){p.cast(sum,forward_query);p.cast(forward_query,sum);}
  out.output=sum;return out;
}
} // namespace tide::device_online
