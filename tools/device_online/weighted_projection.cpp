#include "weighted_projection.h"
#include "weighted_bank_gemm.h"
#include "weighted_storage_budget.h"
#include "device_launch_tide_weighted_head_plan.h"
#include "device_launch_tide_weighted_head_guard.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include "device_launch_tide_weighted_projection_norm.h"
#include "device_launch_tide_weighted_projection_shared.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void check(const at::Tensor& t,at::Device d,at::ScalarType type,at::IntArrayRef shape){
  if(!t.defined()||t.device()!=d||t.scalar_type()!=type||t.sizes()!=shape||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid weighted projection buffer");
}
void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("weighted projection tensor budget exceeded");}
void shared(DeviceProgram& p,const at::Tensor& owners,const at::Tensor& roots,
    const WeightedProjectionParameters& source,const WeightedProjectionParameters& saved,
    const at::Tensor& input,const at::Tensor& connected,const at::Tensor& error,int64_t mode){
  const auto rows=owners.numel(),out=saved.matrix.size(1),in=saved.matrix.size(2);
  const bool bias=saved.bias.defined();auto sb=bias?source.bias:error,db=bias?saved.bias:error;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_projection_shared)(1,stream,
    ptr(owners),ptr(roots),ptr(source.matrix),ptr(sb),ptr(saved.matrix),ptr(db),ptr(input),ptr(connected),ptr(error),
    rows,in,out,bias,mode),"single-bank projection snapshot/transpose guard");},
    {owners,roots,source.matrix,sb,saved.matrix,db,input,connected,error});
}
void status(DeviceProgram& p,const at::Tensor& rows,const at::Tensor& error){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(1,stream,
    ptr(rows),ptr(error),rows.numel()),"weighted projection all-row decision");},{rows,error});
}
void guard(DeviceProgram& p,const WeightedProjectionTape& t,const at::Tensor& value,
    const at::Tensor& roots,const at::Tensor& clean,const at::Tensor& flags,const at::Tensor& error,int64_t mode){
  auto errors=at::zeros_like(t.owners,t.owners.options().dtype(at::kInt));
  const auto rows=t.owners.numel(),aw=t.output.size(1);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_guard)(1,stream,
    ptr(t.owners),ptr(value),ptr(error),ptr(error),ptr(error),ptr(roots),ptr(value),ptr(clean),ptr(flags),ptr(errors),ptr(error),
    rows,aw,0,0,0,mode),"weighted projection finite/connection guard");},
    {t.owners,value,roots,clean,flags,errors,error});status(p,errors,error);
}
}
WeightedProjectionTape append_weighted_projection(DeviceProgram& p,const at::Tensor& input,
    const at::Tensor& owners,const WeightedProjectionParameters& w,std::optional<double> epsilon,const at::Tensor& error,int64_t available){
  if(at::GradMode::is_enabled()||!input.defined()||input.dim()!=2||input.size(0)<1||input.size(1)<1||input.device().type()!=resident_device_type
    ||!w.matrix.defined()||w.matrix.dim()!=3||w.matrix.size(0)<1||w.matrix.size(1)<1||epsilon&&(!std::isfinite(*epsilon)||*epsilon<=0))
    throw std::invalid_argument("weighted projection requires explicit bounded FP32/FP16 accelerator rows and RMS option");
  const auto device=input.device();validate_kernel_device(device);const auto rows=input.size(0),in=input.size(1),out=w.matrix.size(1),banks=w.matrix.size(0);
  const WeightedStorageIncrement storage(p,{input,owners,w.matrix,w.bias,error});
  const long double base=weighted_projection_forward_bytes(rows,in,out,banks);
  if(input.scalar_type()==at::kHalf){
    check(input,device,at::kHalf,{rows,in});check(owners,device,at::kLong,{rows});check(error,device,at::kInt,{1});
    check(w.matrix,device,at::kHalf,{banks,out,in});if(w.bias.defined())check(w.bias,device,at::kHalf,{banks,out});
    const long double extra=4.L*(input.numel()+w.matrix.numel()+(w.bias.defined()?w.bias.numel():0))+rows*(6.L*out+4);
    budget(base+extra,available);
    auto x=at::empty(input.sizes(),input.options().dtype(at::kFloat));
    WeightedProjectionParameters widened{at::empty(w.matrix.sizes(),x.options()),w.bias.defined()?at::empty(w.bias.sizes(),x.options()):at::Tensor{}};
    p.cast(input,x);p.cast(w.matrix,widened.matrix);if(w.bias.defined())p.cast(w.bias,widened.bias);
    auto t=append_weighted_projection(p,x,owners,widened,epsilon,error,available-static_cast<int64_t>(extra));
    auto result=at::empty(t.output.sizes(),input.options()),checked=at::empty_like(t.output);
    p.cast(t.output,result);p.cast(result,checked);guard(p,t,checked,error,error,error,error,1);
    t.output=result;budget(storage.bytes(),available);return t;
  }
  check(input,device,at::kFloat,{rows,in});check(owners,device,at::kLong,{rows});check(error,device,at::kInt,{1});
  check(w.matrix,device,at::kFloat,{banks,out,in});if(w.bias.defined())check(w.bias,device,at::kFloat,{banks,out});
  budget(base,available);
  WeightedProjectionTape t{at::zeros_like(input),at::ones({rows,1},input.options()),at::zeros_like(owners),at::zeros_like(owners),
    at::zeros({banks,out,in},input.options()),at::zeros({banks,out},input.options()),at::zeros({rows,out},input.options()),banks,epsilon.has_value(),w.bias.defined()};
  auto clean=at::zeros_like(input),errors=at::zeros({rows},error.options());p.copy(t.owners,owners);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_plan)(1,stream,
    ptr(input),ptr(t.owners),ptr(clean),ptr(t.indices),ptr(errors),ptr(error),rows,in,banks),"projection input/owner plan");},
    {input,t.owners,clean,t.indices,errors,error});status(p,errors,error);
  if(epsilon)p.rms_norm(clean,*epsilon,t.input,t.rstd);else p.copy(t.input,clean);
  if(banks==1){
    shared(p,t.owners,error,w,{t.matrix,t.has_bias?t.bias:at::Tensor{}},error,error,error,0);
    auto transpose=at::zeros({1,in,out},input.options());p.permute(t.matrix,{0,2,1},transpose);
    p.batch_matmul(t.input.unsqueeze(0),transpose,t.output.unsqueeze(0));if(t.has_bias)p.add(t.output,t.bias);
  }else{
    auto plan=append_weighted_bank_plan(p,t.owners,error,banks,error,false);
    append_weighted_bank_snapshot(p,plan.owner_connected,w.matrix,w.bias,t.matrix,t.bias,error);
    append_weighted_bank_gemm(p,plan,t.input,t.matrix,t.has_bias?t.bias:at::Tensor{},t.output,error);
  }
  guard(p,t,t.output,error,error,error,error,1);budget(storage.bytes(),available);return t;
}
WeightedProjectionGradient append_weighted_projection_vjp(DeviceProgram& p,const WeightedProjectionTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(at::GradMode::is_enabled()||!t.input.defined()||t.input.dim()!=2||t.input.size(0)<1||t.input.size(1)<1
    ||!t.output.defined()||t.output.dim()!=2||t.output.size(1)<1||t.banks<1||t.input.device().type()!=resident_device_type)
    throw std::invalid_argument("invalid weighted projection tape");
  const auto device=t.input.device();const auto rows=t.input.size(0),in=t.input.size(1),out=t.output.size(1),banks=t.banks;
  check(t.input,device,at::kFloat,{rows,in});check(t.rstd,device,at::kFloat,{rows,1});check(t.owners,device,at::kLong,{rows});
  check(t.matrix,device,at::kFloat,{banks,out,in});check(t.bias,device,at::kFloat,{banks,out});
  if(t.output.scalar_type()!=at::kFloat&&t.output.scalar_type()!=at::kHalf)throw std::invalid_argument("projection output storage must be FP32/FP16");
  check(t.output,device,t.output.scalar_type(),{rows,out});
  check(gradient,device,at::kFloat,{rows,out});check(roots,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  const WeightedStorageIncrement storage(p,{t.input,t.rstd,t.owners,t.indices,t.matrix,t.bias,t.output,gradient,roots,error});
  budget(weighted_projection_reverse_bytes(rows,in,out,banks,t.has_bias),available);
  auto dg=at::zeros_like(gradient);
  WeightedProjectionGradient g{at::zeros_like(t.input),at::zeros_like(roots),at::zeros({banks},roots.options()),
    {at::zeros({banks,out,in},gradient.options()),t.has_bias?at::zeros({banks,out},gradient.options()):at::Tensor{}}};
  guard(p,t,gradient,roots,dg,g.row_connected,error,2);
  if(banks==1){
    auto transpose=at::zeros({1,out,rows},gradient.options());p.permute(dg.unsqueeze(0),{0,2,1},transpose);
    p.batch_matmul(transpose,t.input.unsqueeze(0),g.parameters.matrix);
    p.batch_matmul(dg.unsqueeze(0),t.matrix,g.input.unsqueeze(0));
    if(t.has_bias)p.sum(dg,0,true,g.parameters.bias);
  }else{
    auto plan=append_weighted_bank_plan(p,t.owners,g.row_connected,banks,error,true);
    g.owner_connected=plan.owner_connected;
    append_weighted_bank_gemm_vjp(p,plan,t.input,t.matrix,dg,g.input,g.parameters.matrix,g.parameters.bias,error);
  }
  if(t.normalized)p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_projection_norm)(1,stream,
    ptr(t.input),ptr(t.rstd),ptr(g.input),ptr(g.row_connected),ptr(error),rows,in),"projection saved RMS first-order contraction");},
    {t.input,t.rstd,g.input,g.row_connected,error});
  if(banks==1){
    shared(p,t.owners,g.row_connected,{error,error},g.parameters,g.input,g.owner_connected,error,1);
    budget(storage.bytes(),available);return g;
  }
  budget(storage.bytes(),available);return g;
}
}
