#include "tiled_attention.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_attention_tile.h"
#include "aclrtlaunch_tide_attention_softmax.h"
#include "aclrtlaunch_tide_attention_merge.h"

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
at::Tensor append_tiled_attention(CannProgram& p,const at::Tensor& events,const at::Tensor& tokens,
    const at::Tensor& ids,const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,
    const at::Tensor& bias,const at::Tensor& error,const at::Tensor& work,TiledAttentionSpec s) {
  const auto c=query.size(0),h=s.heads,d=query.size(1)/h,k=s.keys;
  if(k<1||k>256||query.size(1)%h||h%s.kv_heads)throw std::invalid_argument("invalid tiled attention geometry");
  const auto opts=query.options(),longs=ids.options();
  auto cursor=at::zeros({1},longs),zero=at::zeros_like(cursor),go=at::zeros_like(error);
  auto indices=at::empty({c,h,k},longs),valid=at::empty({c},longs),additive=at::empty({c,1,1,k},opts);
  auto keys=at::empty({c,h,k,d},opts),values=at::empty_like(keys),kt=at::empty({c,h,d,k},opts);
  auto scores=at::empty({c,h,1,k},opts),scaled=at::empty_like(scores),weights=at::empty_like(scores);
  auto partial=at::empty({c,h,1,d},opts),sum=at::zeros_like(partial),empty=at::zeros_like(partial);
  // Vector DMA owns each full stat row; no adjacent scalar GM cache-line writes.
  auto normalization=at::zeros({c*h,8},opts),empty_norm=at::zeros_like(normalization);
  auto factor=at::full({},s.scale,opts);
  const auto index_key=key.reshape({-1,d}),index_value=value.reshape({-1,d});
  auto begin=p.label(),body=p.label(),done=p.label();
  p.copy(cursor,zero);p.copy(sum,empty);p.copy(normalization,empty_norm);p.mark(begin);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_tile)(1,stream,
    ptr(events),ptr(tokens),ptr(ids),ptr(bias),ptr(cursor),ptr(indices),ptr(valid),ptr(additive),ptr(go),ptr(work),ptr(error),
    c,h,s.kv_heads,s.capacity,s.owners,k,int64_t(s.fiber),s.event_rows,int64_t(s.fiber_bias_rows)),"plan actual attention key tile");},
    {events,tokens,ids,bias,cursor,indices,valid,additive,go,work,error});
  p.branch(go,{done,body});p.mark(body);
  p.index_select(index_key,0,indices.reshape({-1}),keys.reshape({-1,d}));
  p.index_select(index_value,0,indices.reshape({-1}),values.reshape({-1,d}));p.permute(keys,{0,1,3,2},kt);
  p.batch_matmul(query.reshape({c*h,1,d}),kt.reshape({c*h,d,k}),scores.reshape({c*h,1,k}));
  p.multiply(scores,factor,scaled);p.add(scaled,additive);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_softmax)(32,stream,
    ptr(scaled),ptr(valid),ptr(normalization),ptr(weights),ptr(error),c,h,k),"update global attention denominator");},
    {scaled,valid,normalization,weights,error});
  p.batch_matmul(weights.reshape({c*h,1,k}),values.reshape({c*h,k,d}),partial.reshape({c*h,1,d}));
  auto merge=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_attention_merge)(32,stream,
      ptr(partial),ptr(normalization),ptr(sum),ptr(error),c*h,d,mode),"merge globally normalized attention tiles");},
      {partial,normalization,sum,error});
  };
  merge(0);p.branch(go,{begin});p.mark(done);merge(1);
  return sum.reshape({c,h*d});
}
} // namespace tide::device_online
