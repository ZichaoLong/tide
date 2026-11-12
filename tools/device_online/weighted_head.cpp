#include "weighted_head.h"
#include "weighted_projection.h"
#include "weighted_storage_budget.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
namespace {
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
  budget(weighted_head_forward_bytes(rows,in,hidden,out,banks),available);
  const WeightedStorageIncrement storage(p,{x,owners,w.matrix,w.hidden_bias,w.projection,w.bias,error});
  auto first=append_weighted_projection(p,x,owners,{w.matrix,w.hidden_bias},std::nullopt,error,available);
  auto activation=at::zeros_like(first.output);p.silu(first.output,activation);
  // Reuse the saved owner schedule; no mutable owner is reread between maps.
  auto second=append_weighted_projection(p,activation,first.owners,{w.projection,w.bias},std::nullopt,error,available);
  WeightedHeadTape t{first.input,first.owners,first.indices,first.output,second.input,second.output,
    {first.matrix,first.bias,second.matrix,second.bias},banks};
  budget(storage.bytes(),available);return t;
}
WeightedHeadGradient append_weighted_head_vjp(DeviceProgram& p,const WeightedHeadTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(!t.hidden.defined()||t.hidden.dim()!=2||t.hidden.size(1)<1||!t.output.defined()
     ||t.output.dim()!=2||t.output.size(1)<1||t.parameter_owners<1)
    throw std::invalid_argument("invalid resident head tape dimensions");
  input(t.input);const auto rows=t.input.size(0),in=t.input.size(1),hidden=t.hidden.size(1),out=t.output.size(1),banks=t.parameter_owners;
  const auto device=t.input.device();check(gradient,{rows,out},at::kFloat,device);check(roots,{rows},at::kBool,device);
  check(t.owners,{rows},at::kLong,device);check(t.indices,{rows},at::kLong,device);check(error,{1},at::kInt,device);
  check(t.hidden,{rows,hidden},at::kFloat,device);check(t.preactivation,{rows,hidden},at::kFloat,device);
  check(t.output,{rows,out},at::kFloat,device);check(t.rows.matrix,{banks,hidden,in},at::kFloat,device);
  check(t.rows.hidden_bias,{banks,hidden},at::kFloat,device);check(t.rows.projection,{banks,out,hidden},at::kFloat,device);
  check(t.rows.bias,{banks,out},at::kFloat,device);
  budget(weighted_head_reverse_bytes(rows,in,hidden,out,banks),available);
  const WeightedStorageIncrement storage(p,{t.input,t.owners,t.indices,t.preactivation,t.hidden,t.output,
    t.rows.matrix,t.rows.hidden_bias,t.rows.projection,t.rows.bias,gradient,roots,error});
  // Both maps are unnormalized, so this shape-only rstd is never consumed by
  // their transpose. All numerical saved operands are actual forward values.
  auto rstd=at::ones({rows,1},gradient.options());
  WeightedProjectionTape second{t.hidden,rstd,t.owners,t.indices,t.rows.projection,t.rows.bias,t.output,banks,false,true};
  auto upper=append_weighted_projection_vjp(p,second,gradient,roots,error,available);
  auto dz=at::zeros_like(t.preactivation);p.silu_backward(upper.input,t.preactivation,dz);
  WeightedProjectionTape first{t.input,rstd,t.owners,t.indices,t.rows.matrix,t.rows.hidden_bias,t.preactivation,banks,false,true};
  auto lower=append_weighted_projection_vjp(p,first,dz,upper.row_connected,error,available);
  WeightedHeadGradient g{lower.input,lower.row_connected,lower.owner_connected,
    {lower.parameters.matrix,lower.parameters.bias,upper.parameters.matrix,upper.parameters.bias}};
  budget(storage.bytes(),available);return g;
}
}
