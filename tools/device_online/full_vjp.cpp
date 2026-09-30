#include "full_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_full_vjp_plan.h"
#include "aclrtlaunch_tide_full_vjp_payload.h"
#include "aclrtlaunch_tide_full_vjp_reduce.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType dtype,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=dtype||x.sizes()!=shape
      ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid Full VJP buffer");
}
}
FullVjp append_full_vjp(CannProgram& p,const FullTape& tape,const at::Tensor& gradient,
                       const at::Tensor& connected,const at::Tensor& error,int64_t max_rows,int64_t budget) {
  if(at::GradMode::is_enabled()||!tape.values.defined()||tape.values.dim()!=2
      ||!tape.kinds.defined()||tape.kinds.dim()!=1)
    throw std::invalid_argument("Full VJP requires explicit no-grad and packed device buffers");
  const auto device=tape.values.device();
  const auto capacity=tape.values.size(0),nodes=tape.kinds.size(0),width=tape.width,samples=tape.samples;
  // Persistent outputs, sanitized gather banks, flags and bounded matrix work.
  // Caller-owned tape/cotangents and CannProgram operator workspace are separate.
  const long double fixed=12.L*(capacity+2.L)*width+2.L*capacity+nodes+256
      +(tape.has_tanh?4.L*nodes*(width*static_cast<long double>(width)+width):0.L);
  const long double per_row=tape.has_tanh?12.L*width*width+48.L*width+64.L:64.L;
  if(device.type()!=c10::DeviceType::PrivateUse1||capacity<1||nodes<1||width<1||samples<1
      ||budget<1||max_rows<1||fixed+per_row>budget)
    throw std::invalid_argument("one Full VJP row exceeds tensor workspace budget");
  const int64_t chunk=tape.has_tanh?std::min<int64_t>({capacity,max_rows,static_cast<int64_t>((budget-fixed)/per_row)}):1;
  tensor(tape.metadata,device,at::kLong,{capacity,13});tensor(tape.values,device,at::kFloat,{capacity,5*width+2});
  tensor(tape.count,device,at::kLong,{1});tensor(tape.kinds,device,at::kLong,{nodes});
  tensor(gradient,device,at::kFloat,{capacity,width});tensor(connected,device,at::kBool,{capacity});tensor(error,device,at::kInt,{1});
  if(tape.has_tanh){tensor(tape.weights,device,at::kFloat,{nodes+1,width,width});tensor(tape.biases,device,at::kFloat,{nodes+1,width});}
  else if(tape.weights.defined()||tape.biases.defined())throw std::invalid_argument("identity Full has no parameter banks");
  auto floats=tape.values.options(),longs=tape.kinds.options(),booleans=connected.options();
  auto contents=at::empty({capacity+1,width},floats),comparisons=at::empty_like(contents);
  auto output=at::empty({capacity+chunk,width},floats);
  FullVjp out{contents.narrow(0,0,capacity),output.narrow(0,0,capacity),at::empty({capacity},booleans),
    at::empty({capacity},booleans),{}, {},at::empty({nodes},booleans),at::empty({1},longs),chunk};
  auto source=at::empty({chunk},longs),parameters=at::empty_like(source),destination=at::empty_like(source),owners=at::empty_like(source);
  auto cursor=at::empty({1},longs),owner_count=at::empty_like(cursor),branch=at::empty_like(error);
  for(const auto& x:{contents,comparisons,output,out.content_connected,out.comparison_connected,out.parameter_connected,out.chunks,cursor})p.zero(x);
  if(tape.has_tanh){out.weights=at::empty({nodes,width,width},floats);out.biases=at::empty({nodes,width},floats);p.zero(out.weights);p.zero(out.biases);}
  auto plan=[&] {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_vjp_plan)(1,stream,
      ptr(tape.metadata),ptr(tape.count),ptr(tape.kinds),ptr(connected),ptr(out.content_connected),ptr(out.comparison_connected),
      ptr(out.parameter_connected),ptr(cursor),ptr(source),ptr(parameters),ptr(destination),ptr(owners),ptr(owner_count),
      ptr(branch),ptr(out.chunks),ptr(error),capacity,nodes,samples,chunk,int64_t(tape.has_tanh)),"plan connected Full VJP rows");},
      {tape.metadata,tape.count,tape.kinds,connected,out.content_connected,out.comparison_connected,out.parameter_connected,
       cursor,source,parameters,destination,owners,owner_count,branch,out.chunks,error});
  };
  plan(); // Whole-tape validation precedes all numerical work, even identity.
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_vjp_payload)(32,stream,
    ptr(tape.values),ptr(gradient),ptr(out.content_connected),ptr(out.comparison_connected),ptr(tape.count),
    ptr(contents),ptr(comparisons),ptr(error),width),"sanitize connected Full VJP payload");},
    {tape.values,gradient,out.content_connected,out.comparison_connected,tape.count,contents,comparisons,error});
  if(!tape.has_tanh)return out;
  auto x=at::empty({chunk,width},floats),upstream=at::empty_like(x),bias=at::empty_like(x),product=at::empty_like(x);
  auto activation=at::empty_like(x),dactivation=at::empty_like(x),dx=at::empty_like(x);
  auto weights=at::empty({chunk,width,width},floats),transposed=at::empty_like(weights),dw=at::empty_like(weights);
  const auto head=p.label(),body=p.label(),done=p.label();p.mark(head);p.branch(branch,{done,body});p.mark(body);
  p.index_select(comparisons,0,source,x);p.index_select(contents,0,source,upstream);
  p.index_select(tape.weights,0,parameters,weights);p.index_select(tape.biases,0,parameters,bias);
  p.batch_matmul(x.reshape({chunk,1,width}),weights,product.reshape({chunk,1,width}));
  p.add(product,bias);p.tanh(product,activation);p.tanh_backward(upstream,activation,dactivation);
  p.permute(weights,{0,2,1},transposed);
  p.batch_matmul(dactivation.reshape({chunk,1,width}),transposed,dx.reshape({chunk,1,width}));
  p.batch_matmul(x.reshape({chunk,width,1}),dactivation.reshape({chunk,1,width}),dw);
  // Each padding row has its own scratch destination; no conflicting scatter.
  p.index_copy(output,0,destination,dx);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_vjp_reduce)(32,stream,
    ptr(owners),ptr(owner_count),ptr(parameters),ptr(dw),ptr(dactivation),ptr(out.weights),ptr(out.biases),ptr(error),width,chunk),
    "ordered Full parameter-owner partial reduction");},{owners,owner_count,parameters,dw,dactivation,out.weights,out.biases,error});
  plan();p.branch(branch,{head});p.mark(done);
  return out;
}
} // namespace tide::device_online
