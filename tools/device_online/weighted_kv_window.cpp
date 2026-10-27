#include "weighted_kv_window.h"
#include "device_launch_tide_weighted_window_prepare.h"
#include "device_launch_tide_weighted_window_reverse.h"
#include <cmath>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid weighted KV window buffer");
}
void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("weighted KV window tensor budget exceeded");}
}
WeightedKVWindowTape append_weighted_kv_window(DeviceProgram& p,const weighted::KVConfig& c,const WeightedRawState& raw,
    const at::Tensor& times,const at::Tensor& present,const at::Tensor& error,int64_t available){
  c.validate();if(!raw.integer.defined()||raw.integer.dim()!=2||raw.integer.size(1)<=4)
    throw std::invalid_argument("invalid KV window raw capacity");
  WeightedRawLayout l(c,raw.integer.size(1)-4);validate_weighted_raw_buffers(l,raw);const auto device=raw.floating.device();
  if(device.type()!=resident_device_type||raw.floating.scalar_type()!=at::kFloat||!std::isfinite(float(c.beta_short))||float(c.beta_short)<=0)
    throw std::invalid_argument("KV window needs explicit FP32 accelerator and representable positive beta");
  validate_kernel_device(device);const auto rows=raw.floating.size(0),capacity=l.capacity(),d=c.width,ds=c.summary_width;
  check(times,device,at::kLong,{rows});check(present,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  budget(rows*(4.L*capacity*(2.L*d+ds+4)+16.L*capacity+9),available);
  auto f=raw.floating.options(),i=raw.integer.options();
  WeightedKVWindowTape t{l,at::zeros({rows,capacity,d},f),at::zeros({rows,capacity,d},f),at::zeros({rows,capacity,ds},f),
    at::zeros({rows,2,capacity},f),at::zeros({rows,2,capacity},f),at::zeros({rows,capacity},i),at::zeros({rows,capacity},i),
    at::zeros({rows},i),at::zeros_like(present)};
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_window_prepare)(1,stream,
    ptr(raw.floating),ptr(raw.integer),ptr(times),ptr(present),ptr(t.keys),ptr(t.values),ptr(t.summaries),ptr(t.weights),ptr(t.decay),
    ptr(t.ages),ptr(t.mapping),ptr(t.lengths),ptr(t.present),ptr(error),rows,capacity,d,ds,c.window,float(c.beta_short)),
    "KV original-time visible snapshot and weight channels");},{raw.floating,raw.integer,times,present,t.keys,t.values,t.summaries,t.weights,t.decay,t.ages,t.mapping,t.lengths,t.present,error});return t;
}
WeightedKVWindowGradient append_weighted_kv_window_vjp(DeviceProgram& p,const WeightedKVWindowTape& t,const at::Tensor& keys,
    const at::Tensor& values,const at::Tensor& summaries,const at::Tensor& weights,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(!t.layout.kv()||!t.keys.defined()||t.keys.dim()!=3||t.keys.size(0)<1||t.keys.device().type()!=resident_device_type)
    throw std::invalid_argument("invalid KV window tape");
  const auto device=t.keys.device();const auto rows=t.keys.size(0),capacity=t.layout.capacity(),d=t.layout.field("keys").width,ds=t.layout.field("summaries").width;
  check(keys,device,at::kFloat,{rows,capacity,d});check(values,device,at::kFloat,{rows,capacity,d});check(summaries,device,at::kFloat,{rows,capacity,ds});
  check(weights,device,at::kFloat,{rows,2,capacity});check(t.decay,device,at::kFloat,{rows,2,capacity});
  check(t.mapping,device,at::kLong,{rows,capacity});check(t.lengths,device,at::kLong,{rows});check(t.present,device,at::kBool,{rows});
  check(roots,device,at::kBool,{rows,5});check(error,device,at::kInt,{1});budget(rows*(4.L*t.layout.floating_width()+5),available);
  WeightedKVWindowGradient g{at::zeros({rows,t.layout.floating_width()},keys.options()),at::zeros_like(roots)};
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_window_reverse)(1,stream,
    ptr(t.mapping),ptr(t.lengths),ptr(t.present),ptr(t.decay),ptr(keys),ptr(values),ptr(summaries),ptr(weights),ptr(roots),ptr(g.raw),ptr(g.connected),ptr(error),rows,capacity,d,ds),
    "KV visible snapshot independent raw field adjoints");},{t.mapping,t.lengths,t.present,t.decay,keys,values,summaries,weights,roots,g.raw,g.connected,error});return g;
}
}
