#include "weighted_head.h"
#include "device_launch_tide_weighted_head_plan.h"
#include "device_launch_tide_weighted_head_guard.h"
#include "device_launch_tide_weighted_head_reduce.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType type,at::Device device){
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=type||x.device()!=device||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident head buffer");
}
void input(const at::Tensor& x){
  if(!x.defined()||x.dim()!=2||x.size(0)<1||x.size(1)<1||at::GradMode::is_enabled()
     ||x.device().type()!=resident_device_type)throw std::invalid_argument("resident head requires accelerator rows and explicit no-grad");
  validate_kernel_device(x.device());check(x,x.sizes(),at::kFloat,x.device());
}
void budget(long double required,int64_t available){
  if(available<1||required>available)throw std::invalid_argument("resident head scratch budget exceeded");
}
void status(DeviceProgram& p,const at::Tensor& rows,const at::Tensor& error){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(
    1,stream,ptr(rows),ptr(error),rows.numel()),"head all-row decision");},{rows,error});
}
void guard(DeviceProgram& p,const at::Tensor& owners,const WeightedHeadParameters& values,
    const at::Tensor& roots,const at::Tensor& gradient,const at::Tensor& guarded,const at::Tensor& connected,
    const at::Tensor& errors,const at::Tensor& error,int64_t stage){
  const auto rows=owners.numel();
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_guard)(1,stream,
    ptr(owners),ptr(values.matrix),ptr(values.hidden_bias),ptr(values.projection),ptr(values.bias),ptr(roots),
    ptr(gradient),ptr(guarded),ptr(connected),ptr(errors),ptr(error),rows,values.matrix.numel()/rows,
    values.hidden_bias.numel()/rows,values.projection.numel()/rows,values.bias.numel()/rows,stage),"head row guard");},
    {owners,values.matrix,values.hidden_bias,values.projection,values.bias,roots,gradient,guarded,connected,errors,error});
  status(p,errors,error);
}
}
WeightedHeadTape append_weighted_head(DeviceProgram& p,const at::Tensor& x,const at::Tensor& owners,
    const WeightedHeadParameters& w,const at::Tensor& error,int64_t available){
  input(x);const auto device=x.device();const auto rows=x.size(0),in=x.size(1);
  if(!w.matrix.defined()||w.matrix.dim()!=3||w.matrix.size(0)<1||w.matrix.size(1)<1
     ||!w.projection.defined()||w.projection.dim()!=3||w.projection.size(1)<1)
    throw std::invalid_argument("invalid resident head bank dimensions");
  const auto banks=w.matrix.size(0),hidden=w.matrix.size(1),out=w.projection.size(1);
  check(w.matrix,{banks,hidden,in},at::kFloat,device);check(w.hidden_bias,{banks,hidden},at::kFloat,device);
  check(w.projection,{banks,out,hidden},at::kFloat,device);check(w.bias,{banks,out},at::kFloat,device);
  check(owners,{rows},at::kLong,device);check(error,{1},at::kInt,device);
  budget(rows*(4.L*(in+2.L*hidden*in+2.L*out*hidden+3.L*hidden+2.L*out)+20),available);
  WeightedHeadParameters rw{at::zeros({rows,hidden,in},x.options()),at::zeros({rows,hidden},x.options()),
    at::zeros({rows,out,hidden},x.options()),at::zeros({rows,out},x.options())};
  WeightedHeadTape t{at::zeros_like(x),at::empty_like(owners),at::zeros_like(owners),at::zeros({rows,hidden},x.options()),
    at::zeros({rows,hidden},x.options()),at::zeros({rows,out},x.options()),rw,banks};
  auto errors=at::zeros({rows},error.options());p.copy(t.owners,owners);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_plan)(1,stream,
    ptr(x),ptr(t.owners),ptr(t.input),ptr(t.indices),ptr(errors),ptr(error),rows,in,banks),"head input/owner plan");},
    {x,t.owners,t.input,t.indices,errors,error});status(p,errors,error);
  p.index_select(w.matrix,0,t.indices,rw.matrix);p.index_select(w.hidden_bias,0,t.indices,rw.hidden_bias);
  p.index_select(w.projection,0,t.indices,rw.projection);p.index_select(w.bias,0,t.indices,rw.bias);
  // Unused guard arguments have valid storage but are not read at these stages.
  guard(p,t.owners,rw,error,error,error,error,errors,error,0);
  auto first_transpose=at::zeros({rows,in,hidden},x.options()),second_transpose=at::zeros({rows,hidden,out},x.options());
  p.permute(rw.matrix,{0,2,1},first_transpose);p.permute(rw.projection,{0,2,1},second_transpose);
  p.batch_matmul(t.input.unsqueeze(1),first_transpose,t.preactivation.unsqueeze(1));
  p.add(t.preactivation,rw.hidden_bias);p.silu(t.preactivation,t.hidden);
  p.batch_matmul(t.hidden.unsqueeze(1),second_transpose,t.output.unsqueeze(1));p.add(t.output,rw.bias);
  guard(p,t.owners,{t.output,t.output,t.output,t.output},error,error,error,error,errors,error,1);
  return t;
}
WeightedHeadGradient append_weighted_head_vjp(DeviceProgram& p,const WeightedHeadTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(!t.hidden.defined()||t.hidden.dim()!=2||t.hidden.size(1)<1||!t.output.defined()
     ||t.output.dim()!=2||t.output.size(1)<1||t.parameter_owners<1)
    throw std::invalid_argument("invalid resident head tape dimensions");
  input(t.input);const auto rows=t.input.size(0),in=t.input.size(1),hidden=t.hidden.size(1),out=t.output.size(1),banks=t.parameter_owners;
  const auto device=t.input.device();check(gradient,{rows,out},at::kFloat,device);check(roots,{rows},at::kBool,device);
  check(t.owners,{rows},at::kLong,device);check(error,{1},at::kInt,device);
  check(t.hidden,{rows,hidden},at::kFloat,device);check(t.preactivation,{rows,hidden},at::kFloat,device);
  check(t.output,{rows,out},at::kFloat,device);check(t.rows.matrix,{rows,hidden,in},at::kFloat,device);
  check(t.rows.hidden_bias,{rows,hidden},at::kFloat,device);check(t.rows.projection,{rows,out,hidden},at::kFloat,device);
  check(t.rows.bias,{rows,out},at::kFloat,device);
  const long double bank_size=static_cast<long double>(hidden)*in+hidden+static_cast<long double>(out)*hidden+out;
  budget(4.L*(rows*(bank_size+hidden+in)+banks*bank_size)+rows*5.L+banks*5.L,available);
  auto dg=at::zeros_like(gradient),dh=at::zeros_like(t.hidden),dz=at::zeros_like(t.preactivation);
  WeightedHeadParameters row{at::zeros_like(t.rows.matrix),dz,at::zeros_like(t.rows.projection),dg};
  WeightedHeadParameters result{at::zeros({banks,hidden,in},gradient.options()),at::zeros({banks,hidden},gradient.options()),
    at::zeros({banks,out,hidden},gradient.options()),at::zeros({banks,out},gradient.options())};
  WeightedHeadGradient g{at::zeros_like(t.input),at::zeros_like(roots),at::zeros({banks},roots.options()),result};
  auto errors=at::zeros({rows},error.options());
  guard(p,t.owners,{gradient,gradient,gradient,gradient},roots,gradient,dg,g.row_connected,errors,error,2);
  p.batch_matmul(dg.unsqueeze(2),t.hidden.unsqueeze(1),row.projection);
  p.batch_matmul(dg.unsqueeze(1),t.rows.projection,dh.unsqueeze(1));p.silu_backward(dh,t.preactivation,dz);
  p.batch_matmul(dz.unsqueeze(2),t.input.unsqueeze(1),row.matrix);
  p.batch_matmul(dz.unsqueeze(1),t.rows.matrix,g.input.unsqueeze(1));
  auto bank_errors=at::zeros({banks},error.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_reduce)(1,stream,
    ptr(t.owners),ptr(g.row_connected),ptr(g.input),ptr(row.matrix),ptr(row.hidden_bias),ptr(row.projection),ptr(row.bias),
    ptr(result.matrix),ptr(result.hidden_bias),ptr(result.projection),ptr(result.bias),ptr(g.owner_connected),ptr(bank_errors),ptr(error),
    rows,banks,in,hidden*in,hidden,out*hidden,out),"head shared-owner adjoint reduction");},
    {t.owners,g.row_connected,g.input,row.matrix,row.hidden_bias,row.projection,row.bias,result.matrix,result.hidden_bias,
     result.projection,result.bias,g.owner_connected,bank_errors,error});status(p,bank_errors,error);return g;
}
}
