#include "weighted_reverse_route.h"
#include "device_launch_tide_weighted_reverse_route.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("weighted reverse route shape/device/dtype mismatch");
}
}
void append_weighted_reverse_route(DeviceProgram& p,const AtomBatch& a,const at::Tensor& indices,const at::Tensor& lengths,
    const at::Tensor& gradients,const at::Tensor& roots,const at::Tensor& edges,const WeightedReverseEvents& e,
    const WeightedReverseBoundary& b,const at::Tensor& stage,const at::Tensor& cut,const at::Tensor& error){
  if(at::GradMode::is_enabled()||!gradients.defined()||gradients.dim()!=3||!e.times.defined()||e.times.dim()!=3
      ||!b.values.defined()||b.values.dim()!=2||!a.values.defined()||a.values.dim()!=2||!edges.defined()||edges.dim()!=1)
    throw std::invalid_argument("weighted reverse route requires no-grad typed journal buffers");
  const auto device=gradients.device();if(device.type()!=resident_device_type)throw std::invalid_argument("weighted reverse requires explicit accelerator");
  validate_kernel_device(device);
  const auto rows=gradients.size(0),slots=gradients.size(1),width=gradients.size(2),atoms=a.values.size(0),boundary=b.values.size(0);
  const auto stages=e.times.size(0),nodes=e.times.size(1),samples=e.times.size(2);
  if(rows<1||slots<1||width<2||atoms<1||boundary<1||stages<1||nodes<1||samples<1)throw std::invalid_argument("weighted reverse route capacities must be positive");
  check(a.coordinates,device,at::kLong,{atoms,6});check(a.valid,device,at::kBool,{atoms});check(a.values,device,at::kFloat,{atoms,width});
  check(indices,device,at::kLong,{rows,slots});check(lengths,device,at::kLong,{rows});
  check(gradients,device,at::kFloat,{rows,slots,width});check(roots,device,at::kBool,{rows,slots});check(edges,device,at::kLong,{edges.numel()});
  check(e.times,device,at::kLong,{stages,nodes,samples});check(e.present,device,at::kBool,e.times.sizes());
  check(e.values,device,at::kFloat,{stages,nodes,samples,width});check(e.connected,device,at::kBool,e.times.sizes());
  check(b.coordinates,device,at::kLong,{boundary,6});check(b.present,device,at::kBool,{boundary});
  check(b.values,device,at::kFloat,{boundary,width});check(b.connected,device,at::kBool,{boundary});
  check(stage,device,at::kLong,{1});check(cut,device,at::kLong,{1});check(error,device,at::kInt,{1});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_reverse_route)(1,stream,
    ptr(a.coordinates),ptr(a.valid),ptr(indices),ptr(lengths),ptr(gradients),ptr(roots),ptr(edges),
    ptr(e.times),ptr(e.present),ptr(e.values),ptr(e.connected),ptr(b.coordinates),ptr(b.present),ptr(b.values),ptr(b.connected),
    ptr(stage),ptr(cut),ptr(error),rows,slots,atoms,width,edges.numel(),stages,nodes,samples,boundary),
    "weighted actual-journal message adjoint routing");},
    {a.coordinates,a.valid,indices,lengths,gradients,roots,edges,e.times,e.present,e.values,e.connected,b.coordinates,b.present,b.values,b.connected,stage,cut,error});
}
}
