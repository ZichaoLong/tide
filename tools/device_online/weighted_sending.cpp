#include "weighted_sending.h"
#include "device_launch_tide_weighted_send.h"
#include "device_launch_tide_weighted_sending_vjp.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
#include <limits>
namespace tide::device_online {
namespace {
using namespace weighted_sending;
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType type,at::Device device){
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=type||x.device()!=device
     ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid weighted sending buffer");
}
std::pair<int64_t,int64_t> layout(const at::Tensor& s,int64_t budget,int64_t lane,int64_t row){
  if(!s.defined()||s.dim()!=2||s.size(0)<1||s.size(1)<1||at::GradMode::is_enabled()
     ||s.device().type()!=resident_device_type)throw std::invalid_argument("resident sending requires accelerator rows and explicit no-grad");
  validate_kernel_device(s.device());const auto rows=s.size(0),width=s.size(1);
  if(budget<1||width>(std::numeric_limits<int64_t>::max()-row)/lane||rows>budget/(width*lane+row))
    throw std::invalid_argument("weighted sending scratch budget exceeded");
  check(s,{rows,width},at::kFloat,s.device());return {rows,width};
}
void status(DeviceProgram& p,const at::Tensor& rows,const at::Tensor& error){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(
    1,stream,address(rows),address(error),rows.numel()),"weighted sending all-row decision");},{rows,error});
}
}
WeightedSendingTape append_weighted_sending(DeviceProgram& p,const at::Tensor& scores,
    const at::Tensor& caps,const at::Tensor& received,const at::Tensor& parameters,const at::Tensor& ids,
    const at::Tensor& lengths,const at::Tensor& error,weighted_sending::Kind kind,weighted_sending::Base base,
    int64_t k,int64_t budget){
  const auto [rows,width]=layout(scores,budget,20,32);const auto device=scores.device();
  for(const auto& x:{caps,received})check(x,{rows,width},at::kFloat,device);
  check(parameters,{rows,Parameters},at::kFloat,device);check(ids,{rows,width},at::kLong,device);
  check(lengths,{rows},at::kLong,device);check(error,{1},at::kInt,device);
  if(kind<Direct||kind>Hard||((kind==Proportional||kind==Hard)?k<1:k!=0)
     ||(kind==Direct&&base!=CapBase&&base!=ReceivedBase)||(kind==Unit&&base!=CapBase&&base!=UniformBase)
     ||(kind>=Pooled&&base!=CapBase))throw std::invalid_argument("inapplicable resident sending configuration");
  WeightedSendingTape out{at::zeros_like(scores),at::zeros({rows,width,Details},scores.options()),
    at::zeros({rows,Statistics},scores.options()),at::full({rows},-1,lengths.options()),at::empty_like(lengths),kind,base,k};
  auto errors=at::zeros({rows},error.options());p.copy(out.lengths,lengths);
  // Compact rows share cache lines. One scalar worker owns their publication.
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_send)(1,stream,
    address(scores),address(caps),address(received),address(parameters),address(ids),address(out.lengths),
    address(out.controls),address(out.details),address(out.statistics),address(out.branches),address(errors),address(error),
    rows,width,kind,base,k),"weighted sending forward");},
    {scores,caps,received,parameters,ids,out.lengths,out.controls,out.details,out.statistics,out.branches,errors,error});
  status(p,errors,error);return out;
}
WeightedSendingGradient append_weighted_sending_vjp(DeviceProgram& p,const WeightedSendingTape& tape,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  const auto [rows,width]=layout(gradient,budget,12,17);const auto device=gradient.device();
  check(tape.controls,{rows,width},at::kFloat,device);check(tape.details,{rows,width,Details},at::kFloat,device);
  check(tape.statistics,{rows,Statistics},at::kFloat,device);check(tape.branches,{rows},at::kLong,device);
  check(tape.lengths,{rows},at::kLong,device);check(roots,{rows},at::kBool,device);check(error,{1},at::kInt,device);
  WeightedSendingGradient out{at::zeros({rows,3,width},gradient.options()),at::zeros({rows,Parameters},gradient.options()),
    at::zeros({rows,Connections},roots.options())};auto errors=at::zeros({rows},error.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_sending_vjp)(1,stream,
    address(tape.lengths),address(tape.branches),address(tape.details),address(tape.statistics),address(gradient),
    address(roots),address(out.vectors),address(out.scalars),address(out.connected),address(errors),address(error),
    rows,width,tape.kind,tape.base,tape.k),"weighted sending declared VJP");},
    {tape.lengths,tape.branches,tape.details,tape.statistics,gradient,roots,out.vectors,out.scalars,out.connected,errors,error});
  status(p,errors,error);return out;
}
}
