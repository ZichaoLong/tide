#include "fiber_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_fiber_vjp_plan.h"
#include "aclrtlaunch_tide_fiber_vjp_pool.h"
#include "aclrtlaunch_tide_fiber_vjp_pack.h"
#include "aclrtlaunch_tide_fiber_vjp_state.h"
#include "aclrtlaunch_tide_fiber_vjp_queries.h"
#include "aclrtlaunch_tide_fiber_vjp_sources.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid same-fiber adjoint buffer");
}
}
FiberVjp append_fiber_vjp(CannProgram& p,const FiberVjpInput& in,const at::Tensor& error,
    int64_t chunk,int64_t tile,int64_t budget) {
  if(at::GradMode::is_enabled()||in.rows.dim()!=3||in.key.dim()!=4||in.pool_weights.dim()!=2)
    throw std::invalid_argument("fiber VJP requires explicit no-grad packed sources/KV");
  const auto device=in.rows.device();const int64_t b=in.rows.size(0),s=in.rows.size(1),w=in.rows.size(2);
  const int64_t h=in.key.size(1),k=in.key.size(2),domain=in.pool_weights.size(1),c=chunk;
  if(device.type()!=c10::DeviceType::PrivateUse1||b<1||s<1||w<1||h<1||w%h||k<1||s>k
      ||domain<1||c<1||tile<1||tile>256||in.max_repeat_ticks<1||budget<1)
    throw std::invalid_argument("invalid fiber VJP geometry/limits");
  const auto d=w/h;
  const long double own=4.L*(4.L*b*s*w+8.L*b*w*w+16.L*b*w+4.L*b*k*w+4.L*b*k+8.L*b*domain+4.L*b*s
      +12.L*c*w*w+32.L*c*w+4.L*c*k*w+4.L*c*k)+64.L*(b+c)+4096;
  if(own>budget/2.L)throw std::invalid_argument("fiber VJP tensor budget exceeded");
  tensor(in.rows,device,at::kFloat,{b,s,w});tensor(in.slots,device,at::kLong,{b,s});
  for(const auto& x:{in.counts,in.lengths,in.old_lengths,in.ticks,in.pool_kinds,in.pool_lengths})tensor(x,device,at::kLong,{b});
  for(const auto& x:{in.key,in.value,in.key_root,in.value_root})tensor(x,device,at::kFloat,{b,h,k,d});
  for(const auto& x:{in.bias,in.bias_root})tensor(x,device,at::kFloat,{b,k});
  for(const auto& x:{in.connected,in.key_on,in.value_on,in.bias_on})tensor(x,device,at::kBool,{b});
  tensor(in.cotangent,device,at::kFloat,{b,w});tensor(in.qkv,device,at::kFloat,{b,w,3*w});
  tensor(in.qkv_bias,device,at::kFloat,{b,3*w});tensor(in.projection,device,at::kFloat,{b,w,w});
  tensor(in.pool_weights,device,at::kFloat,{b,domain});tensor(error,device,at::kInt,{1});
  auto f=in.rows.options(),l=in.counts.options(),flags=in.connected.options();
  FiberVjp out{at::empty_like(in.rows),at::empty_like(in.connected),at::empty_like(in.key),at::empty_like(in.value),at::empty_like(in.bias),
    at::empty({b,3},flags),at::empty_like(in.qkv),at::empty_like(in.qkv_bias),at::empty_like(in.projection),
    at::empty_like(in.cotangent),at::empty({b},f),at::empty_like(in.pool_weights),at::empty({b,6},flags),at::empty({1},l)};
  for(auto x:{out.rows,out.rows_connected,out.key,out.value,out.bias,out.cache_connected,out.qkv,out.qkv_bias,
      out.projection,out.projection_bias,out.decay,out.pool,out.parameter_connected,out.chunks})p.zero(x);
  auto cursor=at::empty({1},l),plan=at::empty({c,3},l),branch=at::empty_like(error);
  auto schedule=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_vjp_plan)(1,stream,
      ptr(in.counts),ptr(in.slots),ptr(in.lengths),ptr(in.old_lengths),ptr(in.ticks),ptr(in.pool_kinds),ptr(in.pool_lengths),
      ptr(in.connected),ptr(in.key_on),ptr(in.value_on),ptr(in.bias_on),ptr(out.rows_connected),ptr(out.cache_connected),ptr(out.parameter_connected),
      ptr(cursor),ptr(plan),ptr(branch),ptr(out.chunks),ptr(error),b,s,k,domain,c,in.max_repeat_ticks,mode),"pack connected fiber VJP source/query rows");},
      {in.counts,in.slots,in.lengths,in.old_lengths,in.ticks,in.pool_kinds,in.pool_lengths,in.connected,in.key_on,in.value_on,in.bias_on,
       out.rows_connected,out.cache_connected,out.parameter_connected,cursor,plan,branch,out.chunks,error});
  };
  auto safe_projection=at::empty_like(in.projection),wt=at::empty_like(in.projection),pooled_root=at::empty_like(in.cotangent);
  auto state=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_vjp_state)(32,stream,
      ptr(in.cotangent),ptr(in.connected),ptr(in.projection),ptr(in.key_root),ptr(in.value_root),ptr(in.bias_root),
      ptr(in.key_on),ptr(in.value_on),ptr(in.bias_on),ptr(in.lengths),ptr(in.old_lengths),ptr(in.ticks),ptr(out.parameter_connected),
      ptr(out.projection_bias),ptr(safe_projection),ptr(out.key),ptr(out.value),ptr(out.bias),ptr(out.decay),ptr(error),b,w,h,k,mode),
      "seed proposed-cache roots or return old-cache/decay adjoints");},
      {in.cotangent,in.connected,in.projection,in.key_root,in.value_root,in.bias_root,in.key_on,in.value_on,in.bias_on,in.lengths,in.old_lengths,
       in.ticks,out.parameter_connected,out.projection_bias,safe_projection,out.key,out.value,out.bias,out.decay,error});
  };
  schedule(0);state(0);p.permute(safe_projection,{0,2,1},wt);
  p.batch_matmul(out.projection_bias.reshape({b,1,w}),wt,pooled_root.reshape({b,1,w}));
  auto logits=at::empty_like(in.pool_weights),prob=at::empty_like(logits),coeff=at::empty({b,s},f);
  auto pool_partial=at::empty_like(logits),pooled=at::empty({b,w},f),source_gradient=at::empty_like(in.rows);
  for(auto x:{pool_partial,pooled,source_gradient})p.zero(x);
  auto pool=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_vjp_pool)(1,stream,
      ptr(in.counts),ptr(in.slots),ptr(in.pool_kinds),ptr(in.pool_lengths),ptr(in.pool_weights),ptr(in.connected),
      ptr(logits),ptr(prob),ptr(coeff),ptr(pool_partial),ptr(out.pool),ptr(error),b,s,domain,mode),"fiber pooling coefficients/Jacobian");},
      {in.counts,in.slots,in.pool_kinds,in.pool_lengths,in.pool_weights,in.connected,logits,prob,coeff,pool_partial,out.pool,error});
  };
  pool(0);p.softmax(logits,1,prob);pool(1);
  auto x=at::empty({c,1,w},f),qw=at::empty({c,w,w},f),qb=at::empty({c,1,w},f),projected=at::empty_like(x);
  auto keys=at::empty({c,h,k,d},f),values=at::empty_like(keys),biases=at::empty({c,k},f),lens=at::empty({c},l),on=at::empty({c},flags);
  auto cot=at::empty({c,h,d},f),query_root=at::empty({c,w},f),product=at::empty_like(query_root),dp=at::empty({c},f);
  auto qkv=at::empty({c,w,3*w},f),qkvt=at::empty({c,3*w,w},f),bar=at::empty({c,1,3*w},f),dx=at::empty_like(x),dw=at::empty_like(qkv);
  auto pack=[&](int64_t mode) {
    const auto weight=mode==2?qkv:qw;
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_vjp_pack)(32,stream,
      ptr(plan),ptr(in.rows),ptr(in.qkv),ptr(in.qkv_bias),ptr(in.projection),ptr(in.cotangent),ptr(in.connected),ptr(in.key_on),ptr(in.value_on),
      ptr(in.key),ptr(in.value),ptr(in.bias),ptr(in.lengths),ptr(in.old_lengths),ptr(coeff),ptr(pooled_root),ptr(source_gradient),ptr(out.key),ptr(out.value),
      ptr(x),ptr(weight),ptr(qb),ptr(projected),ptr(projected),ptr(keys),ptr(values),ptr(biases),ptr(lens),ptr(cot),ptr(query_root),ptr(on),ptr(bar),ptr(error),
      b,s,w,h,k,c,mode),"pack complete-fiber attention and projection operands");},
      {plan,in.rows,in.qkv,in.qkv_bias,in.projection,in.cotangent,in.connected,in.key_on,in.value_on,in.key,in.value,in.bias,in.lengths,
       in.old_lengths,coeff,pooled_root,source_gradient,out.key,out.value,x,weight,qb,projected,keys,values,biases,lens,cot,query_root,on,bar,error});
  };
  const auto head=p.label(),body=p.label(),done=p.label();p.mark(head);schedule(1);p.branch(branch,{done,body});p.mark(body);
  pack(0);p.batch_matmul(x,qw,projected);p.add(projected,qb);pack(1);
  auto a=append_attention_vjp(p,{projected.reshape({c,h,d}),keys,values,biases,lens,cot,on},error,1./std::sqrt(double(d)),tile,budget/2);
  p.multiply(a.output.reshape({c,w}),query_root,product);p.sum(product,1,false,dp);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_vjp_queries)(32,stream,
    ptr(plan),ptr(coeff),ptr(a.output),ptr(a.query),ptr(a.key),ptr(a.value),ptr(a.bias),ptr(dp),ptr(pooled),ptr(source_gradient),
    ptr(out.key),ptr(out.value),ptr(out.bias),ptr(pool_partial),ptr(error),b,s,w,h,k,domain,c),"reduce all query contributions to complete-fiber KV and pooling");},
    {plan,coeff,a.output,a.query,a.key,a.value,a.bias,dp,pooled,source_gradient,out.key,out.value,out.bias,pool_partial,error});
  p.branch(branch,{head});p.mark(done);
  pool(2);p.batch_matmul(pooled.reshape({b,w,1}),out.projection_bias.reshape({b,1,w}),out.projection);
  schedule(3);const auto next=p.label(),work=p.label(),end=p.label();p.mark(next);schedule(2);p.branch(branch,{end,work});p.mark(work);
  pack(2);p.permute(qkv,{0,2,1},qkvt);p.batch_matmul(bar,qkvt,dx);p.batch_matmul(x.reshape({c,w,1}),bar,dw);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_vjp_sources)(32,stream,
    ptr(plan),ptr(dx),ptr(dw),ptr(bar),ptr(out.rows),ptr(out.qkv),ptr(out.qkv_bias),ptr(error),b,s,w,c),"reduce stable source/projection adjoints");},
    {plan,dx,dw,bar,out.rows,out.qkv,out.qkv_bias,error});
  p.branch(branch,{next});p.mark(end);state(1);return out;
}
} // namespace tide::device_online
