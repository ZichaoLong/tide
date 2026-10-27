#include "weighted_allocation.h"
#include "device_launch_tide_weighted_allocate.h"
#include "device_launch_tide_weighted_allocation_vjp.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tide::device_online {
namespace {
using namespace weighted_allocation;
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType type,at::Device device){
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=type||x.device()!=device
      ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid weighted allocation buffer");
}
std::pair<int64_t,int64_t> layout(const at::Tensor& scores,int64_t budget,int64_t lane_bytes,int64_t row_bytes){
  if(!scores.defined()||scores.dim()!=2||scores.size(0)<1||scores.size(1)<1
      ||at::GradMode::is_enabled()||scores.device().type()!=resident_device_type)
    throw std::invalid_argument("resident weighted allocation requires accelerator rows and explicit no-grad");
  validate_kernel_device(scores.device());
  const auto rows=scores.size(0),width=scores.size(1);
  if(budget<1||width>(std::numeric_limits<int64_t>::max()-row_bytes)/lane_bytes
      ||rows>budget/(width*lane_bytes+row_bytes))
    throw std::invalid_argument("weighted allocation scratch budget exceeded");
  check(scores,{rows,width},at::kFloat,scores.device());return {rows,width};
}
void status(DeviceProgram& p,const at::Tensor& row_errors,const at::Tensor& error){
  const auto rows=row_errors.numel();
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(
    1,stream,address(row_errors),address(error),rows),"weighted allocation all-row decision");},{row_errors,error});
}
}
WeightedAllocationTape append_weighted_allocation(DeviceProgram& p,const at::Tensor& scores,
    const at::Tensor& caps,const at::Tensor& bounds,const at::Tensor& lengths,const at::Tensor& error,
    weighted_allocation::Rule rule,float alpha,bool has_shift,int64_t budget){
  const auto [rows,width]=layout(scores,budget,16,28);const auto device=scores.device();
  check(caps,{rows,width},at::kFloat,device);check(bounds,{rows,BoundFields},at::kFloat,device);
  check(lengths,{rows},at::kLong,device);check(error,{1},at::kInt,device);
  if(rule<Quad||rule>Entropy||!std::isfinite(alpha)||(rule==Entropy?(alpha<1||alpha>2):alpha!=1))
    throw std::invalid_argument("invalid/inapplicable resident allocation regularizer/alpha");
  WeightedAllocationTape out{at::zeros_like(scores),at::zeros({rows,width,SlopeFields},scores.options()),
    at::zeros({rows,2},scores.options()),at::full({rows},Absent,lengths.options()),at::empty_like(lengths),has_shift};
  auto errors=at::zeros({rows},error.options());p.copy(out.lengths,lengths);
  // One scalar worker owns compact rows; different rows can share cache lines.
  // A parallel lowering must use padded scratch or vector GM publication.
  const uint32_t blocks=1;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocate)(blocks,stream,
    address(scores),address(caps),address(bounds),address(out.lengths),address(out.controls),address(out.slopes),
    address(out.statistics),address(out.branches),address(errors),address(error),rows,width,rule,alpha),
    "weighted allocation response solve");},
    {scores,caps,bounds,out.lengths,out.controls,out.slopes,out.statistics,out.branches,errors,error});
  status(p,errors,error);return out;
}
WeightedAllocationGradient append_weighted_allocation_vjp(DeviceProgram& p,const WeightedAllocationTape& tape,
    const at::Tensor& cotangent,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  const auto [rows,width]=layout(cotangent,budget,8,26);const auto device=cotangent.device();
  check(tape.controls,{rows,width},at::kFloat,device);check(tape.slopes,{rows,width,SlopeFields},at::kFloat,device);
  check(tape.statistics,{rows,2},at::kFloat,device);check(tape.branches,{rows},at::kLong,device);
  check(tape.lengths,{rows},at::kLong,device);check(roots,{rows},at::kBool,device);check(error,{1},at::kInt,device);
  WeightedAllocationGradient out{at::zeros({rows,2,width},cotangent.options()),
    at::zeros({rows,BoundFields},cotangent.options()),at::zeros({rows,Connections},roots.options())};
  auto errors=at::zeros({rows},error.options());const uint32_t blocks=1;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_vjp)(blocks,stream,
    address(tape.lengths),address(tape.branches),address(tape.slopes),address(cotangent),address(roots),
    address(out.vectors),address(out.scalars),address(out.connected),address(errors),address(error),rows,width,tape.has_shift),
    "weighted exact allocation VJP");},
    {tape.lengths,tape.branches,tape.slopes,cotangent,roots,out.vectors,out.scalars,out.connected,errors,error});
  status(p,errors,error);return out;
}
}
