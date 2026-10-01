#include "parameter_publish.h"
#include "tide/fiber_attention.h"
#include "packed_lh_full.h"
#include "event_reverse.h"
#include "packed_aggregate.h"
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
  const auto device=b.decay.device();const auto dtype=b.decay.scalar_type();
  if(device.type()!=c10::DeviceType::PrivateUse1||width<1||nodes<1||(dtype!=at::kFloat&&dtype!=at::kHalf))
    throw std::invalid_argument("parameter publication requires NPU FP32/FP16 payload banks");
  const int64_t fp16=dtype==at::kHalf;
  buffer(values,device,at::kFloat,registry.values.sizes());buffer(error,device,at::kInt,{1});
  buffer(b.decay,device,dtype,{nodes,width});buffer(b.retention,device,dtype,{nodes});buffer(b.read,device,dtype,{nodes,width});
  buffer(b.sources,device,dtype,{std::max<int64_t>(1,inputs+edges)});buffer(b.emission,device,dtype,{edges+ports+1,1});
  std::map<std::string,Ref> refs;bool tanh=false,lh=false,normalized=false;int64_t swiglu=0,emission=0;
  const auto slots=aggregate_slots(g);
  for(int64_t n=0;n<nodes;++n) {
    const auto& node=g.nodes[n];const auto name="nodes."+std::to_string(n)+".";
    const auto aggregate=aggregate_kind(node.aggregation);normalized|=aggregate!=0;
    if(aggregate>=2)for(int64_t slot=0;slot<g.source_counts[n];++slot)
      refs[name+"extra."+(aggregate==2?"agg_mass_":"agg_logit_")+std::to_string(slot)]={12,n*slots+slot,{}};
    const auto kind=node.identity?0:lh_full_kind(node.full);
    if(!node.identity&&((node.emission!="broadcast"&&node.emission!="slot_affine")||
        (node.full!="identity"&&node.full!="tanh"&&node.full!="swiglu"&&!kind)||(node.memory!="identity"&&node.memory!="ema"&&node.memory!="lh-add-repeat-v1"&&node.memory!="attention"&&!is_fiber_attention_profile(node.memory))))
      throw std::invalid_argument("parameter publication module contract unavailable");
    if(node.identity)continue;
    refs[name+"read"]={4,n*width,{width}};
    if(node.emission=="slot_affine")for(int64_t slot=0;slot<g.outgoing_ports.offsets[n+1]-g.outgoing_ports.offsets[n];++slot,++emission) {
      refs[name+"extra.emit_w_"+std::to_string(slot)]={19,emission*width*width,{width,width}};
      refs[name+"extra.emit_b_"+std::to_string(slot)]={20,emission*width,{width}};
    }
    if(node.full=="tanh"){tanh=true;refs[name+"weight"]={0,n*width*width,{width,width}};refs[name+"bias"]={1,n*width,{width}};}
    if(kind) {
      lh=true;const auto norm=(kind-1)%3;
      if(norm)refs[name+"extra.lh_norm_weight"]={7,n*width,{width}};
      if(norm==2)refs[name+"extra.lh_norm_bias"]={8,n*width,{width}};
    }
    if(node.full=="swiglu") {
      const auto offset=swiglu++*2*width*width;
      refs[name+"extra.ffn_gate"]={9,offset,{width,2*width}};refs[name+"extra.ffn_up"]={10,offset,{width,2*width}};
      refs[name+"extra.ffn_down"]={11,offset,{2*width,width}};
    }
    if(node.memory=="ema")refs[name+"decay"]={2,n*width,{width}};
    if(node.memory=="lh-add-repeat-v1")refs[name+"extra.add_retention"]={3,n,{}};
  }
  if(emission){buffer(b.projections.weights,device,dtype,{emission+1,width,width});buffer(b.projections.biases,device,dtype,{emission+1,width});}
  if(tanh){buffer(b.weights,device,dtype,{nodes+1,width,width});buffer(b.biases,device,dtype,{nodes+1,width});}
  if(lh){buffer(b.extra.lh_weights,device,dtype,{nodes+1,width});buffer(b.extra.lh_biases,device,dtype,{nodes+1,width});}
  if(swiglu){buffer(b.extra.gate,device,dtype,{swiglu+1,width,2*width});buffer(b.extra.up,device,dtype,{swiglu+1,width,2*width});
    buffer(b.extra.down,device,dtype,{swiglu+1,2*width,width});}
  if(normalized)buffer(b.aggregate.weights,device,at::kFloat,{nodes,slots});
  std::vector<int64_t> fiber_nodes;
  for(int64_t n=0;n<nodes;++n)if(!g.nodes[n].identity&&is_fiber_attention_profile(g.nodes[n].memory))fiber_nodes.push_back(n);
  if(b.fiber.nodes!=fiber_nodes)throw std::invalid_argument("fiber publication node map mismatch");
  const int64_t fibers=fiber_nodes.size();bool learned_pool=false;
  if(fibers) {
    buffer(b.fiber.qkv,device,dtype,{fibers+1,width,3*width});
    buffer(b.fiber.qkv_bias,device,dtype,{fibers+1,3*width});
    buffer(b.fiber.projection,device,dtype,{fibers+1,width,width});
    buffer(b.fiber.projection_bias,device,dtype,{fibers+1,width});
    buffer(b.fiber.decay,device,dtype,{fibers});
    int64_t pool_slots=1;for(const auto n:fiber_nodes)pool_slots=std::max(pool_slots,g.source_counts[n]);
    for(int64_t i=0;i<fibers;++i) {
      const auto n=fiber_nodes[i];const auto name="nodes."+std::to_string(n)+".extra.";
      refs[name+"fiber_qkv"]={13,i*3*width*width,{width,3*width}};
      refs[name+"fiber_qkv_bias"]={14,i*3*width,{3*width}};
      refs[name+"fiber_out"]={15,i*width*width,{width,width}};
      refs[name+"fiber_out_bias"]={16,i*width,{width}};
      refs[name+"fiber_decay"]={17,i,{}};
      if(g.nodes[n].memory!="lh-fiber-attention-sum-repeat-v1"&&g.nodes[n].memory!="lh-fiber-attention-mean-repeat-v1") {
        learned_pool=true;refs[name+"fiber_pool"]={18,i*pool_slots,{g.source_counts[n]}};
      }
    }
    if(learned_pool)buffer(b.fiber.pool,device,at::kFloat,{fibers+1,pool_slots});
  }
  for(int64_t i=0;i<inputs;++i)refs["input_scale."+std::to_string(i)]={5,i,{}};
  for(int64_t i=0;i<edges;++i)refs["agg_scale."+std::to_string(i)]={5,inputs+i,{}};
  for(size_t i=0;i<g.outgoing_ports.bindings.size();++i) {
    const auto binding=g.outgoing_ports.bindings[i];refs[(binding.kind?"edge_scale.":"output_scale.")+std::to_string(binding.id)]={6,int64_t(i),{}};
  }
  std::vector<int64_t> plan,tiles{0};
  for(size_t i=0;i<registry.owners.size();++i) {
    const auto& owner=registry.owners[i];const auto offset=registry.offsets[i],size=owner.value.numel();if(offset==-1)continue;
    if(owner.value.scalar_type()!=dtype)throw std::invalid_argument("publication payload owner dtype disagrees with banks");
    if(offset<0||size<1||offset>values.numel()-size)throw std::invalid_argument("invalid parameter publication owner offset");
    for(const auto& name:owner.aliases)if(auto it=refs.find(name);it!=refs.end()) {
      if(owner.value.sizes()!=at::IntArrayRef(it->second.shape))throw std::invalid_argument("parameter publication alias shape mismatch");
      plan.insert(plan.end(),{offset,it->second.bank,it->second.offset,size});tiles.push_back(tiles.back()+(size+255)/256);
    }
  }
  const int64_t count=plan.size()/4,tasks=tiles.back();
  if(8.L*(std::max<size_t>(4,plan.size())+tiles.size())+4>(b.attention.empty()?budget:budget/2))throw std::invalid_argument("parameter publication tensor budget exceeded");
  auto table=at::tensor(plan.empty()?std::vector<int64_t>(4,0):plan,at::kLong).reshape({-1,4}).to(device);
  auto offsets=at::tensor(tiles,at::kLong).to(device),dummy=at::zeros({1},values.options());
  const auto w=tanh?b.weights:dummy,bias=tanh?b.biases:dummy;
  const auto lw=lh?b.extra.lh_weights:dummy,lb=lh?b.extra.lh_biases:dummy;
  const auto gate=swiglu?b.extra.gate:dummy,up=swiglu?b.extra.up:dummy,down=swiglu?b.extra.down:dummy;
  const auto aggregate=normalized?b.aggregate.weights:dummy;
  const auto fq=fibers?b.fiber.qkv:dummy,fqb=fibers?b.fiber.qkv_bias:dummy;
  const auto fo=fibers?b.fiber.projection:dummy,fob=fibers?b.fiber.projection_bias:dummy;
  const auto fd=fibers?b.fiber.decay:dummy,fp=learned_pool?b.fiber.pool:dummy;
  const auto ew=emission?b.projections.weights:dummy,eb=emission?b.projections.biases:dummy;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_parameter_publish)(32,stream,
    ptr(table),ptr(offsets),ptr(values),ptr(w),ptr(bias),ptr(b.decay),ptr(b.retention),ptr(b.read),ptr(b.sources),ptr(b.emission),ptr(lw),ptr(lb),ptr(gate),ptr(up),ptr(down),ptr(aggregate),
    ptr(fq),ptr(fqb),ptr(fo),ptr(fob),ptr(fd),ptr(fp),ptr(ew),ptr(eb),ptr(error),count,tasks,fp16),
    "publish updated parameter owners into forward banks");},{table,offsets,values,w,bias,b.decay,b.retention,b.read,b.sources,b.emission,lw,lb,gate,up,down,aggregate,fq,fqb,fo,fob,fd,fp,ew,eb,error});
  if(!b.attention.empty())append_event_publish(p,b.attention,registry,values,error,budget/2);
}
} // namespace tide::device_online
