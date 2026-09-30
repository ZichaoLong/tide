#include "packed_sum.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_content_sum.h"
#include "aclrtlaunch_tide_sum_plan.h"
#include "aclrtlaunch_tide_vector_sum.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}}
PackedSum append_packed_sum(CannProgram& p,const ReadyBatch& r,const at::Tensor& sources,
    const at::Tensor& scales,int64_t nodes,int64_t inputs,int64_t edges,
    const at::Tensor& error,bool vectorized,const at::Tensor& input_origins) {
  if(at::GradMode::is_enabled())throw std::invalid_argument("packed sum has no autograd contract");
  const auto device=r.atoms.values.device();
  for(const auto& t:{r.atoms.values,r.atoms.coordinates,r.fiber_offsets,r.fibers,r.counts,sources,scales,error})
    if(!t.is_contiguous()||t.device()!=device||t.requires_grad())throw std::invalid_argument("packed sum requires contiguous colocated inference buffers");
  if(device.type()!=c10::DeviceType::PrivateUse1||r.atoms.values.scalar_type()!=at::kFloat
      ||scales.scalar_type()!=at::kFloat||r.atoms.values.dim()!=2||nodes<1||inputs<0||edges<0)
    throw std::invalid_argument("packed sum requires NPU FP32 and valid dimensions");
  const auto capacity=r.atoms.values.size(0),width=r.atoms.values.size(1);
  if(capacity<1||width<1||r.atoms.coordinates.sizes()!=at::IntArrayRef({capacity,6})
      ||r.fibers.sizes()!=at::IntArrayRef({capacity,4})||r.fiber_offsets.sizes()!=at::IntArrayRef({capacity+1})
      ||r.counts.numel()!=3||sources.sizes()!=at::IntArrayRef({std::max<int64_t>(1,inputs+edges),2})
      ||scales.numel()!=sources.size(0)||error.scalar_type()!=at::kInt||error.numel()!=1)
    throw std::invalid_argument("packed sum buffer shape mismatch");
  for(const auto& t:{r.atoms.coordinates,r.fiber_offsets,r.fibers,r.counts,sources})
    if(t.scalar_type()!=at::kLong)throw std::invalid_argument("packed sum coordinates must be int64");
  const int64_t project=input_origins.defined();
  auto origins=project?input_origins:at::zeros({1,2},r.counts.options());
  if(origins.device()!=device||origins.scalar_type()!=at::kLong||!origins.is_contiguous()
      ||origins.requires_grad()||origins.sizes()!=at::IntArrayRef{project?std::max<int64_t>(1,edges):1,2})
    throw std::invalid_argument("packed sum origin table requires colocated int64 [edges,2]");
  PackedSum out{at::zeros({capacity,width},scales.options()),at::zeros_like(r.atoms.values)};
  auto keys=at::zeros({capacity},r.counts.options()),order=at::zeros_like(keys);
  out.keys=keys;out.order=order;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_sum_plan)(1,stream,
    ptr(r.atoms.coordinates),ptr(r.fiber_offsets),ptr(r.fibers),ptr(r.counts),ptr(sources),ptr(origins),
    ptr(keys),ptr(order),ptr(error),capacity,nodes,inputs,edges,project),"packed sum metadata preflight");},
    {r.atoms.coordinates,r.fiber_offsets,r.fibers,r.counts,sources,origins,keys,order,error});
  const std::vector<at::Tensor> buffers{r.atoms.values,r.fiber_offsets,r.counts,keys,order,scales,out.content,out.weighted,error};
  if(!vectorized) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_content_sum)(1,stream,
      ptr(r.atoms.values),ptr(r.fiber_offsets),ptr(r.counts),ptr(keys),ptr(order),
      ptr(scales),ptr(out.content),ptr(out.weighted),ptr(error),width),
      "scalar packed sum");},buffers);
    return out;
  }
  // Each block processes whole, disjoint (fiber,payload-tile) work items. The
  // grid is static; the device's current counts determine actual work at replay.
  const uint32_t blocks=std::min<int64_t>(32,capacity);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_vector_sum)(blocks,stream,
    ptr(r.atoms.values),ptr(r.fiber_offsets),ptr(r.counts),ptr(keys),ptr(order),ptr(scales),ptr(out.content),
    ptr(out.weighted),ptr(error),width),"vector packed sum");},
    buffers);
  return out;
}
} // namespace tide::device_online
