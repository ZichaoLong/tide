#include "parameter_publish.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_parameter_publish.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <map>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
struct Ref {int64_t bank,offset;std::vector<int64_t> shape;};
void buffer(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid device parameter publication buffer");
}
}
void append_parameter_publish(CannProgram& p,const ParameterBanks& b,const ParameterVjp& registry,
    const at::Tensor& values,const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!b.graph||!b.decay.defined()||b.decay.dim()!=2||budget<1
      ||registry.owners.size()!=registry.offsets.size()||!registry.values.defined())
    throw std::invalid_argument("parameter publication requires explicit no-grad owner/banks");
  const auto& g=*b.graph;const int64_t nodes=g.nodes.size(),width=b.decay.size(1),inputs=g.inputs.size(),edges=g.edges.size(),ports=g.outputs.size();
  const auto device=b.decay.device();if(device.type()!=c10::DeviceType::PrivateUse1||width<1||nodes<1)throw std::invalid_argument("parameter publication requires NPU FP32");
  buffer(values,device,at::kFloat,registry.values.sizes());buffer(error,device,at::kInt,{1});
  buffer(b.decay,device,at::kFloat,{nodes,width});buffer(b.retention,device,at::kFloat,{nodes});buffer(b.read,device,at::kFloat,{nodes,width});
  buffer(b.sources,device,at::kFloat,{std::max<int64_t>(1,inputs+edges)});buffer(b.emission,device,at::kFloat,{edges+ports+1,1});
  std::map<std::string,Ref> refs;bool tanh=false;
  for(int64_t n=0;n<nodes;++n) {
    const auto& node=g.nodes[n];const auto name="nodes."+std::to_string(n)+".";
    if(node.aggregation!="sum"||(!node.identity&&(node.emission!="broadcast"||
        (node.full!="identity"&&node.full!="tanh")||(node.memory!="identity"&&node.memory!="ema"&&node.memory!="lh-add-repeat-v1"))))
      throw std::invalid_argument("parameter publication module contract unavailable");
    if(node.identity)continue;
    refs[name+"read"]={4,n*width,{width}};
    if(node.full=="tanh"){tanh=true;refs[name+"weight"]={0,n*width*width,{width,width}};refs[name+"bias"]={1,n*width,{width}};}
    if(node.memory=="ema")refs[name+"decay"]={2,n*width,{width}};
    if(node.memory=="lh-add-repeat-v1")refs[name+"extra.add_retention"]={3,n,{}};
  }
  if(tanh){buffer(b.weights,device,at::kFloat,{nodes+1,width,width});buffer(b.biases,device,at::kFloat,{nodes+1,width});}
  for(int64_t i=0;i<inputs;++i)refs["input_scale."+std::to_string(i)]={5,i,{}};
  for(int64_t i=0;i<edges;++i)refs["agg_scale."+std::to_string(i)]={5,inputs+i,{}};
  for(size_t i=0;i<g.outgoing_ports.bindings.size();++i) {
    const auto binding=g.outgoing_ports.bindings[i];refs[(binding.kind?"edge_scale.":"output_scale.")+std::to_string(binding.id)]={6,int64_t(i),{}};
  }
  std::vector<int64_t> plan,tiles{0};
  for(size_t i=0;i<registry.owners.size();++i) {
    const auto& owner=registry.owners[i];const auto offset=registry.offsets[i],size=owner.value.numel();if(offset==-1)continue;
    if(offset<0||size<1||offset>values.numel()-size)throw std::invalid_argument("invalid parameter publication owner offset");
    for(const auto& name:owner.aliases)if(auto it=refs.find(name);it!=refs.end()) {
      if(owner.value.sizes()!=at::IntArrayRef(it->second.shape))throw std::invalid_argument("parameter publication alias shape mismatch");
      plan.insert(plan.end(),{offset,it->second.bank,it->second.offset,size});tiles.push_back(tiles.back()+(size+255)/256);
    }
  }
  const int64_t count=plan.size()/4,tasks=tiles.back();
  if(8.L*(std::max<size_t>(4,plan.size())+tiles.size())+4>budget)throw std::invalid_argument("parameter publication tensor budget exceeded");
  auto table=at::tensor(plan.empty()?std::vector<int64_t>(4,0):plan,at::kLong).reshape({-1,4}).to(device);
  auto offsets=at::tensor(tiles,at::kLong).to(device),dummy=at::zeros({1},values.options());
  const auto w=tanh?b.weights:dummy,bias=tanh?b.biases:dummy;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_parameter_publish)(32,stream,
    ptr(table),ptr(offsets),ptr(values),ptr(w),ptr(bias),ptr(b.decay),ptr(b.retention),ptr(b.read),ptr(b.sources),ptr(b.emission),ptr(error),count,tasks),
    "publish updated parameter owners into forward banks");},{table,offsets,values,w,bias,b.decay,b.retention,b.read,b.sources,b.emission,error});
}
} // namespace tide::device_online
