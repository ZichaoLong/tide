#include "weighted_kv_full_internal.h"
#include "device_launch_tide_weighted_full_roots.h"
namespace tide::device_online {
using namespace kv_full_detail;
WeightedKVFullGradient append_weighted_kv_full_vjp(DeviceProgram& p,const WeightedKVFullTape& t,const at::Tensor& gradient,
    const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(!t.output.defined()||t.output.dim()!=2||t.output.size(0)<1||t.pool.empty()||t.pool.size()!=t.rotary.size())
    throw std::invalid_argument("invalid KV Full tape");
  const auto device=t.output.device();const auto rows=t.output.size(0),d=t.output.size(1),heads=int64_t(t.pool.size()),dh=d/heads,capacity=t.window.layout.capacity(),banks=t.query.banks;
  check(gradient,device,at::kFloat,{rows,d});check(roots,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  budget((rows+banks)*(256.L*(capacity*(d+dh+heads+1.L)+static_cast<long double>(d)*d+3.L*d*t.residual.hidden.size(1)+d)+256),available);
  auto residual=append_weighted_residual_vjp(p,t.residual,gradient,roots,error,available);auto bars=split(p,residual.memory,heads);
  auto pool_roots=at::zeros({rows,2},roots.options()),qroots=at::zeros_like(roots),wroots=at::zeros({rows,5},roots.options());
  auto flags=at::zeros({heads,rows,3},roots.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_full_roots)(1,stream,
    ptr(residual.connected),ptr(flags),ptr(pool_roots),ptr(qroots),ptr(wroots),ptr(error),rows,heads,0),"KV Full output roots");},
    {residual.connected,flags,pool_roots,qroots,wroots,error});
  auto key_bars=at::zeros({heads,rows,capacity,dh},gradient.options()),value_bars=at::zeros_like(key_bars),query_bars=at::zeros({heads,rows,1,dh},gradient.options());
  auto weight_bars=at::zeros({rows,2,capacity},gradient.options()),long_bars=at::zeros({rows,capacity},gradient.options());
  p.zero(long_bars);
  for(int64_t h=0;h<heads;++h){auto pool=append_weighted_pool_vjp(p,t.pool[h],bars[h].reshape({rows,dh}),at::zeros({rows},gradient.options()),pool_roots,error,available);
    p.copy(flags[h],pool.connected);auto scaled=at::zeros_like(pool.logits);p.multiply(pool.logits,t.scale,scaled);
    p.batch_matmul(scaled.unsqueeze(1),t.rotary[h].output,query_bars[h]);
    auto rotated=at::zeros({rows,capacity,dh},gradient.options());p.batch_matmul(scaled.unsqueeze(2),t.queries[h],rotated);
    // Only vector output roots exist; absent/empty pools produce zero bars.
    auto unrotated=append_weighted_rotary_vjp(p,t.rotary[h],rotated,residual.connected,error,available);
    p.copy(key_bars[h],unrotated);p.copy(value_bars[h],pool.values);p.add(long_bars,pool.weights);
  }
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_full_roots)(1,stream,
    ptr(residual.connected),ptr(flags),ptr(pool_roots),ptr(qroots),ptr(wroots),ptr(error),rows,heads,1),"KV Full actual query/raw dependencies");},
    {residual.connected,flags,pool_roots,qroots,wroots,error});
  auto ids=at::ones({1},t.window.ages.options());p.index_copy(weight_bars,1,ids,long_bars.unsqueeze(1));
  auto query=append_weighted_projection_vjp(p,t.query,merge(p,query_bars).reshape({rows,d}),qroots,error,available);
  auto raw=append_weighted_kv_window_vjp(p,t.window,merge(p,key_bars),merge(p,value_bars),at::zeros_like(t.window.summaries),weight_bars,wroots,error,available);
  auto input=at::zeros_like(gradient);p.copy(input,residual.input);p.add(input,query.input);
  return {raw.raw,raw.connected,input,residual.connected,{query.parameters.matrix,residual.parameters},query.owner_connected,
    residual.out_connected,residual.gate_connected,residual.up_connected,residual.down_connected};
}
}
