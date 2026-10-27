#include "weighted_projection.h"
#include "device_launch_tide_weighted_head_plan.h"
#include "device_launch_tide_weighted_head_guard.h"
#include "device_launch_tide_weighted_head_reduce.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include "device_launch_tide_weighted_projection_norm.h"
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
void status(DeviceProgram& p,const at::Tensor& rows,const at::Tensor& error){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(1,stream,
    ptr(rows),ptr(error),rows.numel()),"weighted projection all-row decision");},{rows,error});
}
void guard(DeviceProgram& p,const WeightedProjectionTape& t,const at::Tensor& value,
    const at::Tensor& roots,const at::Tensor& clean,const at::Tensor& flags,const at::Tensor& error,int64_t mode){
  auto errors=at::zeros_like(t.owners,t.owners.options().dtype(at::kInt));
  const auto rows=t.owners.numel(),aw=mode==0?t.matrix.numel()/rows:t.output.size(1),bw=mode==0&&t.has_bias?t.bias.size(1):0;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_guard)(1,stream,
    ptr(t.owners),ptr(mode==0?t.matrix:value),ptr(t.bias),ptr(error),ptr(error),ptr(roots),ptr(value),ptr(clean),ptr(flags),ptr(errors),ptr(error),
    rows,aw,bw,0,0,mode),"weighted projection finite/connection guard");},
    {t.owners,t.matrix,t.bias,value,roots,clean,flags,errors,error});status(p,errors,error);
}
}
WeightedProjectionTape append_weighted_projection(DeviceProgram& p,const at::Tensor& input,
    const at::Tensor& owners,const WeightedProjectionParameters& w,std::optional<double> epsilon,const at::Tensor& error,int64_t available){
  if(at::GradMode::is_enabled()||!input.defined()||input.dim()!=2||input.size(0)<1||input.size(1)<1||input.device().type()!=resident_device_type
    ||!w.matrix.defined()||w.matrix.dim()!=3||w.matrix.size(0)<1||w.matrix.size(1)<1||epsilon&&(!std::isfinite(*epsilon)||*epsilon<=0))
    throw std::invalid_argument("weighted projection requires explicit bounded FP32 accelerator rows and RMS option");
  const auto device=input.device();validate_kernel_device(device);const auto rows=input.size(0),in=input.size(1),out=w.matrix.size(1),banks=w.matrix.size(0);
  check(input,device,at::kFloat,{rows,in});check(owners,device,at::kLong,{rows});check(error,device,at::kInt,{1});
  check(w.matrix,device,at::kFloat,{banks,out,in});if(w.bias.defined())check(w.bias,device,at::kFloat,{banks,out});
  budget(rows*(4.L*(2.L*in+2.L*in*out+2.L*out+1)+28),available);
  WeightedProjectionTape t{at::zeros_like(input),at::ones({rows,1},input.options()),at::zeros_like(owners),at::zeros_like(owners),
    at::zeros({rows,out,in},input.options()),at::zeros({rows,out},input.options()),at::zeros({rows,out},input.options()),banks,epsilon.has_value(),w.bias.defined()};
  auto clean=at::zeros_like(input),errors=at::zeros({rows},error.options());p.copy(t.owners,owners);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_plan)(1,stream,
    ptr(input),ptr(t.owners),ptr(clean),ptr(t.indices),ptr(errors),ptr(error),rows,in,banks),"projection input/owner plan");},
    {input,t.owners,clean,t.indices,errors,error});status(p,errors,error);
  if(epsilon)p.rms_norm(clean,*epsilon,t.input,t.rstd);else p.copy(t.input,clean);
  p.index_select(w.matrix,0,t.indices,t.matrix);if(t.has_bias)p.index_select(w.bias,0,t.indices,t.bias);
  guard(p,t,error,error,error,error,error,0);auto transpose=at::zeros({rows,in,out},input.options());
  p.permute(t.matrix,{0,2,1},transpose);p.batch_matmul(t.input.unsqueeze(1),transpose,t.output.unsqueeze(1));if(t.has_bias)p.add(t.output,t.bias);
  guard(p,t,t.output,error,error,error,error,1);return t;
}
WeightedProjectionGradient append_weighted_projection_vjp(DeviceProgram& p,const WeightedProjectionTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(at::GradMode::is_enabled()||!t.input.defined()||t.input.dim()!=2||t.input.size(0)<1||t.input.size(1)<1
    ||!t.output.defined()||t.output.dim()!=2||t.output.size(1)<1||t.banks<1||t.input.device().type()!=resident_device_type)
    throw std::invalid_argument("invalid weighted projection tape");
  const auto device=t.input.device();const auto rows=t.input.size(0),in=t.input.size(1),out=t.output.size(1),banks=t.banks;
  check(t.input,device,at::kFloat,{rows,in});check(t.rstd,device,at::kFloat,{rows,1});check(t.owners,device,at::kLong,{rows});
  check(t.matrix,device,at::kFloat,{rows,out,in});check(t.bias,device,at::kFloat,{rows,out});check(t.output,device,at::kFloat,{rows,out});
  check(gradient,device,at::kFloat,{rows,out});check(roots,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  budget(rows*(4.L*(in+out+static_cast<long double>(out)*in)+9)+banks*(4.L*(out*static_cast<long double>(in)+(t.has_bias?out:0))+5),available);
  auto dg=at::zeros_like(gradient),dw=at::zeros_like(t.matrix);
  WeightedProjectionGradient g{at::zeros_like(t.input),at::zeros_like(roots),at::zeros({banks},roots.options()),
    {at::zeros({banks,out,in},gradient.options()),t.has_bias?at::zeros({banks,out},gradient.options()):at::Tensor{}}};
  guard(p,t,gradient,roots,dg,g.row_connected,error,2);
  p.batch_matmul(dg.unsqueeze(2),t.input.unsqueeze(1),dw);p.batch_matmul(dg.unsqueeze(1),t.matrix,g.input.unsqueeze(1));
  if(t.normalized)p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_projection_norm)(1,stream,
    ptr(t.input),ptr(t.rstd),ptr(g.input),ptr(g.row_connected),ptr(error),rows,in),"projection saved RMS first-order contraction");},
    {t.input,t.rstd,g.input,g.row_connected,error});
  auto errors=at::zeros({banks},error.options());const auto bias=t.has_bias?g.parameters.bias:error;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_reduce)(1,stream,
    ptr(t.owners),ptr(g.row_connected),ptr(g.input),ptr(dw),ptr(dg),ptr(error),ptr(error),ptr(g.parameters.matrix),ptr(bias),ptr(error),ptr(error),
    ptr(g.owner_connected),ptr(errors),ptr(error),rows,banks,in,out*in,t.has_bias?out:0,0,0),"projection shared parameter reduction");},
    {t.owners,g.row_connected,g.input,dw,dg,g.parameters.matrix,bias,g.owner_connected,errors,error});status(p,errors,error);return g;
}
}
