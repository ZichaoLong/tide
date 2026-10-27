#include "weighted_residual.h"
#include "device_launch_tide_weighted_head_plan.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include "device_launch_tide_weighted_residual_guard.h"
#include <cmath>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident residual buffer");
}
void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("resident residual tensor budget exceeded");}
void guard(DeviceProgram& p,const at::Tensor& owners,const at::Tensor& input,const at::Tensor& roots,
    const at::Tensor& output,const at::Tensor& connected,const at::Tensor& error,bool reverse){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_residual_guard)(1,stream,
    ptr(owners),ptr(input),ptr(roots),ptr(output),ptr(connected),ptr(error),input.size(0),input.size(1),reverse),
    "weighted residual independent row guard");},{owners,input,roots,output,connected,error});
}
}
WeightedResidualTape append_weighted_residual(DeviceProgram& p,const at::Tensor& input,
    const at::Tensor& memory,const at::Tensor& owners,const WeightedResidualParameters& w,
    double epsilon,double alpha_memory,double alpha_ffn,const at::Tensor& error,int64_t available){
  if(!input.defined()||input.dim()!=2||input.size(0)<1||input.size(1)<1||!memory.defined()||memory.dim()!=2||memory.size(1)<1
    ||!w.out.defined()||w.out.dim()!=3||w.out.size(0)<1||!w.gate.defined()||w.gate.dim()!=3||w.gate.size(1)<1
    ||!std::isfinite(epsilon)||epsilon<=0||!std::isfinite(float(epsilon))||float(epsilon)<=0
    ||!std::isfinite(alpha_memory)||!std::isfinite(float(alpha_memory))||alpha_memory<0
    ||!std::isfinite(alpha_ffn)||!std::isfinite(float(alpha_ffn))||alpha_ffn<0)
    throw std::invalid_argument("invalid weighted residual dimensions/scales");
  const auto device=input.device();const auto rows=input.size(0),d=input.size(1),dm=memory.size(1),df=w.gate.size(1),banks=w.out.size(0);
  check(input,device,at::kFloat,{rows,d});check(memory,device,at::kFloat,{rows,dm});check(owners,device,at::kLong,{rows});check(error,device,at::kInt,{1});
  check(w.out,device,at::kFloat,{banks,d,dm});check(w.gate,device,at::kFloat,{banks,df,d});check(w.up,device,at::kFloat,{banks,df,d});check(w.down,device,at::kFloat,{banks,d,df});
  budget(rows*(64.L*(d*static_cast<long double>(dm)+3.L*d*df+d+df)+64),available);
  auto out=append_weighted_projection(p,memory,owners,{w.out,{}},std::nullopt,error,available);
  auto clean=at::zeros_like(input),indices=at::zeros_like(owners),errors=at::zeros_like(owners,owners.options().dtype(at::kInt));
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_plan)(1,stream,
    ptr(input),ptr(out.owners),ptr(clean),ptr(indices),ptr(errors),ptr(error),rows,d,banks),"Full actual input rows");},
    {input,out.owners,clean,indices,errors,error});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(1,stream,
    ptr(errors),ptr(error),rows),"Full input status");},{errors,error});
  auto memory_scale=at::full({1},float(alpha_memory),input.options()),ffn_scale=at::full({1},float(alpha_ffn),input.options());
  auto x=at::zeros_like(input);p.multiply(out.output,memory_scale,x);p.add(x,clean);
  auto gate=append_weighted_projection(p,x,owners,{w.gate,{}},epsilon,error,available);
  auto up=append_weighted_projection(p,x,owners,{w.up,{}},epsilon,error,available);
  auto hidden=at::zeros_like(gate.output),product=at::zeros_like(hidden);p.silu(gate.output,hidden);p.multiply(hidden,up.output,product);
  auto down=append_weighted_projection(p,product,owners,{w.down,{}},std::nullopt,error,available);
  auto output=at::zeros_like(input);p.multiply(down.output,ffn_scale,output);p.add(output,x);
  guard(p,out.owners,output,error,output,error,error,false);
  return {out,gate,up,down,clean,hidden,output,memory_scale,ffn_scale};
}
WeightedResidualGradient append_weighted_residual_vjp(DeviceProgram& p,const WeightedResidualTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(!t.input.defined()||t.input.dim()!=2||t.input.size(0)<1||!t.hidden.defined()||t.hidden.dim()!=2)
    throw std::invalid_argument("invalid weighted residual tape");
  const auto device=t.input.device();const auto rows=t.input.size(0),d=t.input.size(1),dm=t.out.input.size(1),df=t.hidden.size(1),banks=t.out.banks;
  check(gradient,device,at::kFloat,{rows,d});check(roots,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  check(t.memory_scale,device,at::kFloat,{1});check(t.ffn_scale,device,at::kFloat,{1});
  budget((rows+banks)*(64.L*(d*static_cast<long double>(dm)+3.L*d*df+d+df)+64),available);
  auto clean=at::zeros_like(gradient),flags=at::zeros_like(roots),down_bar=at::zeros_like(gradient);
  guard(p,t.out.owners,gradient,roots,clean,flags,error,true);p.multiply(clean,t.ffn_scale,down_bar);
  auto down=append_weighted_projection_vjp(p,t.down,down_bar,flags,error,available);
  auto gate_bar=at::zeros_like(t.hidden),product_bar=at::zeros_like(t.hidden),up_bar=at::zeros_like(t.hidden);
  p.multiply(down.input,t.up.output,product_bar);p.silu_backward(product_bar,t.gate.output,gate_bar);p.multiply(down.input,t.hidden,up_bar);
  auto gate=append_weighted_projection_vjp(p,t.gate,gate_bar,flags,error,available);
  auto up=append_weighted_projection_vjp(p,t.up,up_bar,flags,error,available);
  auto input_bar=at::zeros_like(gradient);p.copy(input_bar,clean);p.add(input_bar,gate.input);p.add(input_bar,up.input);
  auto out_bar=at::zeros_like(gradient);p.multiply(input_bar,t.memory_scale,out_bar);
  auto out=append_weighted_projection_vjp(p,t.out,out_bar,flags,error,available);
  guard(p,t.out.owners,input_bar,flags,input_bar,flags,error,true);
  return {input_bar,out.input,flags,{out.parameters.matrix,gate.parameters.matrix,up.parameters.matrix,down.parameters.matrix},
    out.owner_connected,gate.owner_connected,up.owner_connected,down.owner_connected};
}
}
