#include "block_kv_memory.h"
#include "block_kv_bank_internal.h"
#include "device_launch_tide_block_kv_read_roots.h"
#include "device_launch_tide_block_kv_read_weight_vjp.h"
namespace tide::device_online {
using namespace block_bank_detail;
BlockKVReadRoots append_block_kv_read_roots(DeviceProgram& p,const BlockKVReadPlan& t,const at::Tensor& flags,int64_t kind,const at::Tensor& error){
  const auto& b=*t.snapshot.bank;const auto l=t.snapshot.length,o=b.batch*b.nodes,rows=o*l,k=b.entries;
  BlockKVReadRoots r{at::zeros({rows,5},t.present.options()),at::zeros({rows},t.present.options()),at::zeros({o*k},t.present.options()),at::zeros({o*k},t.present.options())};
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_read_roots)(1,stream,
    ptr(flags),ptr(t.weights),ptr(t.present),ptr(r.snapshot),ptr(r.query),ptr(r.keys),ptr(r.values),ptr(error),b.batch,l,b.nodes,k,flags.size(0),kind),
    "KV bank independent structural Read/Full dependencies");},{flags,t.weights,t.present,r.snapshot,r.query,r.keys,r.values,error});return r;
}
at::Tensor append_block_kv_weight_vjp(DeviceProgram& p,const BlockKVReadPlan& t,const at::Tensor& bars,const at::Tensor& error){
  const auto o=t.weights.size(1),q=t.weights.size(2),k=t.weights.size(3);auto out=at::zeros({o,k},bars.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_read_weight_vjp)(1,stream,
    ptr(bars),ptr(t.decay),ptr(t.visible),ptr(out),ptr(error),o,q,k),"KV bank actual original-age weight transpose");},{bars,t.decay,t.visible,out,error});return out;
}
BlockKVMemoryGradient append_block_kv_memory_vjp(DeviceProgram& p,const BlockKVMemoryTape& t,const at::Tensor& memory_bar,
    const at::Tensor& total_bar,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  const auto& s=t.plan.snapshot;const auto& b=*s.bank;const auto o=b.batch*b.nodes,q=s.length,k=b.entries,rows=o*q,d=b.layout.field("keys").width,ds=t.memory.size(2);
  check(memory_bar,error,at::kFloat,{rows,2,ds});check(total_bar,error,at::kFloat,{rows,2});check(roots,error,at::kBool,{rows,4});
  auto borrowed=block_kv_memory_tensors(t);borrowed.insert(borrowed.end(),{memory_bar,total_bar,roots,error});
  const WeightedStorageIncrement storage(p,std::move(borrowed));auto remaining=[&](){admit(storage.bytes(),budget);return budget-storage.bytes();};
  auto mb=owner_order(p,s,memory_bar),tb=owner_order(p,s,total_bar),rb=owner_order(p,s,roots);
  auto flags=at::zeros({2,o,q,3},roots.options()),weight_bars=at::zeros_like(t.plan.weights),logit_bar=at::zeros({o,q,k},memory_bar.options());
  auto value_bar=at::zeros({o,k,ds},memory_bar.options());p.zero(logit_bar);p.zero(value_bar);
  for(int64_t channel=0;channel<2;++channel){auto pool_roots=at::zeros({o,q,2},roots.options());put(p,pool_roots,0,columns(p,rb,channel,1));put(p,pool_roots,1,columns(p,rb,channel+2,1));
    auto pool=append_block_kv_pool_vjp(p,t.pools[channel],columns(p,mb,channel*ds,ds),columns(p,tb,channel,1).reshape({o,q}),pool_roots,error,remaining());
    p.copy(flags[channel],pool.connected);p.copy(weight_bars[channel],pool.weights);p.add(logit_bar,pool.logits);p.add(value_bar,pool.values);}
  auto connected=append_block_kv_read_roots(p,t.plan,flags,t.query?1:0,error);
  BlockKVMemoryGradient out;out.bank={at::zeros_like(b.keys),at::zeros_like(b.values),at::zeros_like(b.summaries),
    append_block_kv_weight_vjp(p,t.plan,weight_bars,error),connected.snapshot,s};
  out.input=at::zeros({rows,d},memory_bar.options());out.input_connected=at::zeros({rows},roots.options());
  if(t.query){auto match=append_block_kv_match_vjp(p,*t.match,logit_bar,error,remaining());
    auto query=append_weighted_projection_vjp(p,*t.query,row_order(p,s,match.query),connected.query,error,remaining());
    auto key=append_weighted_projection_vjp(p,*t.key,match.keys.reshape({o*k,ds}),connected.keys,error,remaining());
    auto value=append_weighted_projection_vjp(p,*t.value,value_bar.reshape({o*k,ds}),connected.values,error,remaining());
    out.bank.keys=key.input.reshape({o,k,d});out.bank.values=value.input.reshape({o,k,d});out.input=query.input;out.input_connected=query.row_connected;
    out.parameters={query.parameters.matrix,key.parameters.matrix,value.parameters.matrix};out.query_connected=query.owner_connected;out.key_connected=key.owner_connected;out.value_connected=value.owner_connected;
  }else out.bank.summaries=value_bar;
  admit(storage.bytes(),budget);return out;
}
}
