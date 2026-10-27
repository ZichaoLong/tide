#include "weighted_kv_update_internal.h"
#include "device_launch_tide_weighted_kv_roots.h"
#include "device_launch_tide_weighted_kv_merge.h"
namespace tide::device_online {
using namespace kv_update_detail;
WeightedKVUpdateGradient append_weighted_kv_update_vjp(DeviceProgram& p,const WeightedKVUpdateTape& t,
    const at::Tensor& bar,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  t.config.validate();const auto& l=t.raw.layout;validate_weighted_raw_buffers(l,t.raw.result);
  const auto device=t.raw.result.floating.device();const auto rows=t.raw.result.floating.size(0),d=t.config.width,ds=t.config.summary_width,rank=t.config.adapter_rank,banks=t.banks;
  if(!l.kv()||device.type()!=resident_device_type||banks<1)throw std::invalid_argument("invalid resident KV update tape");
  check(t.owners,device,at::kLong,{rows});check(error,device,at::kInt,{1});check(t.kv_indices,device,at::kLong,{2*d});check(t.summary_indices,device,at::kLong,{ds});
  check(t.saved_rows.left,device,at::kFloat,{rows,2*d,rank});check(t.saved_rows.right,device,at::kFloat,{rows,rank,d});
  budget(rows*(12.L*l.floating_width()+100.L*(d*static_cast<long double>(d)+rank*static_cast<long double>(d)+ds*static_cast<long double>(d))+100.L*(d+ds)+128)
    +banks*(20.L*d*d+20.L*ds*d+40.L*rank*d+32),available);
  auto raw=append_weighted_raw_update_vjp(p,t.raw,bar,roots,error,available);
  auto kv_roots=at::zeros({rows},roots.options()),summary_roots=at::zeros_like(kv_roots);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_kv_roots)(1,stream,
    ptr(raw.proposal_connected),ptr(kv_roots),ptr(summary_roots),ptr(error),rows),"KV independent K/V/summary roles");},
    {raw.proposal_connected,kv_roots,summary_roots,error});
  auto kv_bar=at::zeros({rows,2*d},bar.options()),summary_bar=at::zeros({rows,ds},bar.options());
  p.index_select(raw.proposal,1,t.kv_indices,kv_bar);p.index_select(raw.proposal,1,t.summary_indices,summary_bar);
  auto kv=append_weighted_projection_vjp(p,t.kv,kv_bar,kv_roots,error,available);
  auto summary=append_weighted_projection_vjp(p,t.summary,summary_bar,summary_roots,error,available);
  auto left=at::zeros_like(t.saved_rows.left),right=at::zeros_like(t.saved_rows.right);
  p.batch_matmul(kv.parameters.matrix,transpose(p,t.saved_rows.right),left);
  p.batch_matmul(transpose(p,t.saved_rows.left),kv.parameters.matrix,right);
  WeightedKVUpdateGradient g{raw.old_values,raw.old_connected,at::zeros({rows,d+1},bar.options()),at::zeros_like(kv_roots),
    {at::zeros({banks,2*d,d},bar.options()),at::zeros({banks,2*d,rank},bar.options()),at::zeros({banks,rank,d},bar.options()),at::zeros({banks,ds,d},bar.options())},
    at::zeros({banks},roots.options()),at::zeros({banks},roots.options())};
  reduce(p,t.owners,kv.row_connected,{kv.parameters.matrix,left,right},{g.parameters.base,g.parameters.left,g.parameters.right},g.kv_connected,error);
  reduce(p,t.owners,summary.row_connected,{summary.parameters.matrix,{},{}},{g.parameters.summary,{},{}},g.summary_connected,error);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_kv_merge)(1,stream,
    ptr(raw.mass),ptr(raw.mass_connected),ptr(kv.input),ptr(kv.row_connected),ptr(summary.input),ptr(summary.row_connected),ptr(g.content),ptr(g.content_connected),ptr(error),rows,d),
    "KV complete aggregated-content adjoint");},{raw.mass,raw.mass_connected,kv.input,kv.row_connected,summary.input,summary.row_connected,g.content,g.content_connected,error});
  return g;
}
}
