#include "weighted_kv_query_internal.h"
#include "device_launch_tide_weighted_query_roots.h"
namespace tide::device_online {
using namespace kv_query_detail;
WeightedKVQueryGradient append_weighted_kv_query_vjp(DeviceProgram& p,const WeightedKVQueryTape& t,const at::Tensor& memory_bar,
    const at::Tensor& total_bar,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(!t.memory.defined()||t.memory.dim()!=3||t.memory.size(1)!=2||t.memory.size(0)<1||!t.pool.attention)
    throw std::invalid_argument("invalid KV qry tape");
  const auto device=t.memory.device();const auto rows=t.memory.size(0),ds=t.memory.size(2),capacity=t.window.layout.capacity(),d=t.window.layout.field("keys").width,banks=t.query.banks;
  check(memory_bar,device,at::kFloat,{rows,2,ds});check(total_bar,device,at::kFloat,{rows,2});check(roots,device,at::kBool,{rows,4});check(error,device,at::kInt,{1});
  check(t.scale,device,at::kFloat,{1});
  budget(rows*(256.L*capacity*(static_cast<long double>(d)*ds+d+ds+1)+256)+banks*(32.L*d*ds+32),available);
  auto indices=at::tensor(std::vector<int64_t>{0,2,1,3},at::kLong).to(device),pool_roots=at::zeros_like(roots);p.index_select(roots,1,indices,pool_roots);
  auto pool=append_weighted_pool_vjp(p,t.pool,memory_bar.reshape({2*rows,ds}),total_bar.reshape({2*rows}),pool_roots.reshape({2*rows,2}),error,available);
  auto logits=at::zeros({rows,capacity},memory_bar.options()),scaled=at::zeros_like(logits),values=at::zeros({rows,capacity,ds},memory_bar.options());
  p.sum(pool.logits.reshape({rows,2,capacity}),1,false,logits);p.multiply(logits,t.scale,scaled);p.sum(pool.values.reshape({rows,2,capacity,ds}),1,false,values);
  auto qroots=at::zeros({rows},roots.options()),kroots=at::zeros({rows*capacity},roots.options()),vroots=at::zeros_like(kroots),wroots=at::zeros({rows,5},roots.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_query_roots)(1,stream,
    ptr(pool.connected),ptr(t.key.owners),ptr(qroots),ptr(kroots),ptr(vroots),ptr(wroots),ptr(error),rows,capacity),
    "KV qry independent four-return roots");},{pool.connected,t.key.owners,qroots,kroots,vroots,wroots,error});
  auto qbar=at::zeros({rows,ds},memory_bar.options()),kbar=at::zeros({rows,capacity,ds},memory_bar.options());
  p.batch_matmul(scaled.unsqueeze(1),t.rotary.output,qbar.unsqueeze(1));p.batch_matmul(scaled.unsqueeze(2),t.query.output.unsqueeze(1),kbar);
  auto unrotated=append_weighted_rotary_vjp(p,t.rotary,kbar,qroots,error,available);
  auto query=append_weighted_projection_vjp(p,t.query,qbar,qroots,error,available);
  auto key=append_weighted_projection_vjp(p,t.key,unrotated.reshape({rows*capacity,ds}),kroots,error,available);
  auto value=append_weighted_projection_vjp(p,t.value,values.reshape({rows*capacity,ds}),vroots,error,available);
  auto raw=append_weighted_kv_window_vjp(p,t.window,key.input.reshape({rows,capacity,d}),value.input.reshape({rows,capacity,d}),at::zeros_like(t.window.summaries),
    pool.weights.reshape({rows,2,capacity}),wroots,error,available);
  return {raw.raw,raw.connected,query.input,query.row_connected,{query.parameters.matrix,key.parameters.matrix,value.parameters.matrix},query.owner_connected,key.owner_connected,value.owner_connected};
}
}
