#include "parameter_plan.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_parameter_vjp.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <map>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid parameter VJP buffer");
}
}
ParameterVjp append_parameter_vjp(CannProgram& p,const Graph& g,const ParameterRegistry& registry,
                                 const GraphVjp& gradient,const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!gradient.decay.defined()||gradient.decay.dim()!=2||g.nodes.empty())
    throw std::invalid_argument("parameter VJP requires explicit no-grad graph adjoints");
  const auto device=gradient.decay.device();const int64_t nodes=g.nodes.size(),width=gradient.decay.size(1);
  const int64_t inputs=g.inputs.size(),edges=g.edges.size(),ports=g.outputs.size(),scales=inputs+2*edges+ports;
  if(device.type()!=c10::DeviceType::PrivateUse1||width<1||budget<1)throw std::invalid_argument("parameter VJP requires bounded NPU FP32");
  tensor(gradient.decay,device,at::kFloat,{nodes,width});tensor(gradient.retention,device,at::kFloat,{nodes});
  tensor(gradient.scales,device,at::kFloat,{std::max<int64_t>(1,scales)});
  for(const auto& flag:{gradient.full_connected,gradient.decay_connected,gradient.retention_connected})tensor(flag,device,at::kBool,{nodes});
  tensor(gradient.scale_connected,device,at::kBool,{std::max<int64_t>(1,scales)});tensor(error,device,at::kInt,{1});
  const auto plan=plan_parameters(g,registry,width,budget);const bool tanh=plan.has_tanh;
  if(tanh){tensor(gradient.weights,device,at::kFloat,{nodes,width,width});tensor(gradient.biases,device,at::kFloat,{nodes,width});}
  else if(gradient.weights.defined()||gradient.biases.defined())throw std::invalid_argument("identity profile fabricated Full parameter banks");
  ParameterVjp out;out.owners=plan.owners;out.offsets=plan.offsets;
  const auto& owners=plan.owner_table;const auto& refs=plan.references;const auto& tiles=plan.tiles;
  const int64_t count=out.owners.size(),tasks=tiles.back(),total=plan.elements;
  auto owner_table=at::tensor(owners.empty()?std::vector<int64_t>(4,0):owners,at::kLong).reshape({-1,4}).to(device);
  auto references=at::tensor(refs.empty()?std::vector<int64_t>(3,0):refs,at::kLong).reshape({-1,3}).to(device);
  auto tile_offsets=at::tensor(tiles,at::kLong).to(device);
  out.values=at::empty({std::max<int64_t>(1,total)},gradient.decay.options());out.connected=at::empty({std::max<int64_t>(1,count)},gradient.full_connected.options());
  p.zero(out.values);p.zero(out.connected);
  auto dummy=at::zeros({1},gradient.decay.options());
  const auto weights=tanh?gradient.weights:dummy,bias=tanh?gradient.biases:dummy;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_parameter_vjp)(32,stream,
    ptr(owner_table),ptr(references),ptr(tile_offsets),ptr(weights),ptr(bias),ptr(gradient.decay),ptr(gradient.retention),ptr(gradient.scales),
    ptr(gradient.full_connected),ptr(gradient.decay_connected),ptr(gradient.retention_connected),ptr(gradient.scale_connected),
    ptr(out.values),ptr(out.connected),ptr(error),count,tasks),"reduce declared parameter aliases on device");},
    {owner_table,references,tile_offsets,weights,bias,gradient.decay,gradient.retention,gradient.scales,gradient.full_connected,
     gradient.decay_connected,gradient.retention_connected,gradient.scale_connected,out.values,out.connected,error});
  return out;
}
} // namespace tide::device_online
