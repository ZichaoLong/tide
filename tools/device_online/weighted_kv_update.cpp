#include "weighted_kv_update_internal.h"
#include "device_launch_tide_weighted_kv_prepare.h"
namespace tide::device_online {
using namespace kv_update_detail;
WeightedKVUpdateTape append_weighted_kv_update(DeviceProgram& p,const weighted::KVConfig& c,const WeightedRawState& old,
    const at::Tensor& h,const at::Tensor& time,const at::Tensor& owners,const at::Tensor& present,
    const WeightedKVParameters& w,const at::Tensor& error,int64_t available){
  c.validate();if(!old.integer.defined()||old.integer.dim()!=2||old.integer.size(1)<=4||!w.base.defined()
    ||w.base.dim()!=3||w.base.size(0)<1)throw std::invalid_argument("invalid KV raw capacity/parameter banks");
  WeightedRawLayout l(c,old.integer.size(1)-4);validate_weighted_raw_buffers(l,old);const auto device=old.floating.device();
  if(device.type()!=resident_device_type)throw std::invalid_argument("resident KV update requires explicit accelerator");validate_kernel_device(device);
  const auto rows=old.floating.size(0),banks=w.base.size(0),d=c.width,ds=c.summary_width,rank=c.adapter_rank;
  check(old.floating,device,at::kFloat,{rows,l.floating_width()});check(h,device,at::kFloat,{rows,d+1});
  check(time,device,at::kLong,{rows});check(owners,device,at::kLong,{rows});check(present,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  check(w.base,device,at::kFloat,{banks,2*d,d});check(w.left,device,at::kFloat,{banks,2*d,rank});
  check(w.right,device,at::kFloat,{banks,rank,d});check(w.summary,device,at::kFloat,{banks,ds,d});
  // Conservative complete composition bound, including child tapes/guards.
  budget(rows*(20.L*l.floating_width()+32.L*l.integer_width()+100.L*(d*static_cast<long double>(d)+rank*static_cast<long double>(d)+ds*static_cast<long double>(d))+64.L*(d+ds)+400)+64.L*(d+ds),available);
  auto u=at::zeros({rows,d},h.options()),mass=at::zeros({rows},h.options()),positive=at::zeros_like(present);
  auto ids=at::zeros_like(owners),indices=at::zeros_like(owners),row_ids=at::zeros_like(owners);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_kv_prepare)(1,stream,
    ptr(h),ptr(owners),ptr(present),ptr(u),ptr(mass),ptr(ids),ptr(indices),ptr(row_ids),ptr(positive),ptr(error),rows,d,banks),
    "KV positive reception and input plan");},{h,owners,present,u,mass,ids,indices,row_ids,positive,error});
  WeightedKVParameters saved{gather(p,w.base,indices,positive,error),gather(p,w.left,indices,positive,error),
    gather(p,w.right,indices,positive,error),gather(p,w.summary,indices,positive,error)};
  auto effective=at::zeros_like(saved.base);p.batch_matmul(saved.left,saved.right,effective);p.add(effective,saved.base);
  effective=guard(p,effective,positive,error);
  auto kv=append_weighted_projection(p,u,row_ids,{effective,{}},c.epsilon,error,available);
  auto summary=append_weighted_projection(p,u,row_ids,{saved.summary,{}},c.epsilon,error,available);
  std::vector<int64_t> kv_columns,summary_columns;
  for(const std::string name:{"keys","values"})for(int64_t j=0;j<d;++j)kv_columns.push_back(l.field(name).offset+j);
  for(int64_t j=0;j<ds;++j)summary_columns.push_back(l.field("summaries").offset+j);
  auto kv_indices=at::tensor(kv_columns,at::kLong).to(device),summary_indices=at::tensor(summary_columns,at::kLong).to(device),proposed=at::zeros_like(old.floating);
  p.index_copy(proposed,1,kv_indices,kv.output);p.index_copy(proposed,1,summary_indices,summary.output);
  auto raw=append_weighted_raw_update(p,l,old,time,mass,present,proposed,error,available);
  return {c,raw,kv,summary,saved,ids,kv_indices,summary_indices,banks};
}
}
