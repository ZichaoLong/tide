#include "weighted_raw_update.h"
#include "device_launch_tide_weighted_raw_plan.h"
#include "device_launch_tide_weighted_raw_apply.h"
#include "device_launch_tide_weighted_raw_reverse.h"
#include "device_launch_tide_weighted_raw_adopt.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void check(const at::Tensor& t,at::Device d,at::ScalarType type,at::IntArrayRef shape){
  if(!t.defined()||t.device()!=d||t.scalar_type()!=type||t.sizes()!=shape||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid weighted raw update buffer");
}
void bounded(long double bytes,int64_t budget){if(budget<1||bytes>budget)throw std::invalid_argument("weighted raw update tensor budget exceeded");}
void error_buffer(const at::Tensor& error,at::Device device){
  if(at::GradMode::is_enabled()||device.type()!=resident_device_type)throw std::invalid_argument("weighted raw update requires explicit accelerator no-grad");
  validate_kernel_device(device);check(error,device,at::kInt,{1});
}
}
WeightedRawUpdateTape append_weighted_raw_update(DeviceProgram& p,const WeightedRawLayout& l,const WeightedRawState& old,
    const at::Tensor& times,const at::Tensor& mass,const at::Tensor& present,const at::Tensor& proposal,const at::Tensor& error,int64_t budget){
  validate_weighted_raw_buffers(l,old);const auto device=old.floating.device();error_buffer(error,device);
  const auto rows=old.floating.size(0),width=l.floating_width(),capacity=l.capacity(),count=int64_t(l.fields().size());
  check(old.floating,device,at::kFloat,{rows,width});check(times,device,at::kLong,{rows});
  check(mass,device,at::kFloat,{rows});check(present,device,at::kBool,{rows});check(proposal,device,at::kFloat,{rows,width});
  bounded(rows*(8.L*width+16.L*l.integer_width()+8.L*std::max<int64_t>(1,capacity)+9)+32.L*count,budget);
  std::vector<int64_t> fields;for(const auto& f:l.fields())fields.insert(fields.end(),
    {f.offset,f.width,int64_t(f.ragged),f.name=="value"?0:(f.name=="weights"?2:(f.name=="W"?3:1))});
  WeightedRawUpdateTape t{l,{at::zeros_like(old.floating),at::zeros_like(old.integer)},at::zeros_like(old.integer),
    at::zeros_like(proposal),at::tensor(fields,at::kLong).to(device),
    at::zeros({rows,std::max<int64_t>(1,capacity)},old.integer.options()),at::zeros({rows,2},mass.options()),at::zeros_like(present)};
  p.copy(t.old_integer,old.integer);p.copy(t.proposal,proposal);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_raw_plan)(1,stream,
    ptr(t.fields),ptr(t.old_integer),ptr(old.floating),ptr(times),ptr(mass),ptr(present),ptr(t.proposal),
    ptr(t.result.integer),ptr(t.mapping),ptr(t.receive),ptr(t.written),ptr(error),rows,width,capacity,count,
    int64_t(l.kv()),int64_t(l.convolution()),l.horizon()),"weighted raw clocks/cache plan");},
    {t.fields,t.old_integer,old.floating,times,mass,present,t.proposal,t.result.integer,t.mapping,t.receive,t.written,error});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_raw_apply)(1,stream,
    ptr(t.fields),ptr(old.floating),ptr(t.proposal),ptr(t.mapping),ptr(t.receive),ptr(t.written),ptr(t.result.floating),ptr(error),
    rows,width,capacity,count,int64_t(l.convolution())),"weighted raw exact retention and append");},
    {t.fields,old.floating,t.proposal,t.mapping,t.receive,t.written,t.result.floating,error});return t;
}
WeightedRawUpdateGradient append_weighted_raw_update_vjp(DeviceProgram& p,const WeightedRawUpdateTape& t,
    const at::Tensor& bar,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  const auto& l=t.layout;validate_weighted_raw_buffers(l,t.result);const auto device=t.result.floating.device();error_buffer(error,device);
  const auto rows=t.result.floating.size(0),width=l.floating_width(),capacity=l.capacity(),count=int64_t(l.fields().size());
  check(t.result.floating,device,at::kFloat,{rows,width});check(t.old_integer,device,at::kLong,{rows,l.integer_width()});
  check(t.proposal,device,at::kFloat,{rows,width});check(t.fields,device,at::kLong,{4*count});
  check(t.mapping,device,at::kLong,{rows,std::max<int64_t>(1,capacity)});check(t.receive,device,at::kFloat,{rows,2});
  check(t.written,device,at::kBool,{rows});check(bar,device,at::kFloat,{rows,width});check(roots,device,at::kBool,{rows,count});
  bounded(rows*(8.L*width+5+2.L*count),budget);
  WeightedRawUpdateGradient g{at::zeros_like(bar),at::zeros_like(bar),at::zeros({rows},bar.options()),
    at::zeros_like(roots),at::zeros_like(roots),at::zeros({rows},roots.options())};
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_raw_reverse)(1,stream,
    ptr(t.fields),ptr(t.old_integer),ptr(t.result.integer),ptr(t.proposal),ptr(t.mapping),ptr(t.receive),ptr(t.written),
    ptr(bar),ptr(roots),ptr(g.old_values),ptr(g.proposal),ptr(g.mass),ptr(g.old_connected),ptr(g.proposal_connected),ptr(g.mass_connected),ptr(error),
    rows,width,capacity,count,int64_t(l.convolution())),"weighted raw independent field adjoints");},
    {t.fields,t.old_integer,t.result.integer,t.proposal,t.mapping,t.receive,t.written,bar,roots,g.old_values,g.proposal,
     g.mass,g.old_connected,g.proposal_connected,g.mass_connected,error});return g;
}
void append_weighted_raw_adopt(DeviceProgram& p,const WeightedRawState& target,const WeightedRawState& source,const at::Tensor& error){
  if(!target.floating.defined()||!target.integer.defined())throw std::invalid_argument("undefined weighted raw destination");
  const auto device=target.floating.device();error_buffer(error,device);
  check(target.floating,device,at::kFloat,target.floating.sizes());check(target.integer,device,at::kLong,target.integer.sizes());
  check(source.floating,device,at::kFloat,target.floating.sizes());check(source.integer,device,at::kLong,target.integer.sizes());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_raw_adopt)(1,stream,
    ptr(source.floating),ptr(source.integer),ptr(target.floating),ptr(target.integer),ptr(error),target.floating.numel(),target.integer.numel()),
    "weighted raw state commit after shared decision");},{source.floating,source.integer,target.floating,target.integer,error});
}
}
