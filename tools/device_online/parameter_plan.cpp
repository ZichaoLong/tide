#include "parameter_plan.h"
#include "tide/fiber_attention.h"
#include "packed_lh_full.h"
#include "packed_aggregate.h"
#include <algorithm>
#include <map>
#include <stdexcept>

namespace tide::device_online {
namespace {struct Ref {int64_t bank,connection,offset;std::vector<int64_t> shape;};}
ParameterPlan plan_parameters(const Graph& g,const ParameterRegistry& registry,int64_t width,int64_t budget,bool controls) {
  const int64_t nodes=g.nodes.size(),inputs=g.inputs.size(),edges=g.edges.size();
  if(nodes<1||width<1||budget<1)throw std::invalid_argument("invalid parameter layout budget/shape");
  std::map<std::string,Ref> by_name;
  const auto attention=event_parameter_offsets(g,width);
  const auto fiber=fiber_parameter_offsets(g,width);
  bool tanh=false,lh=false,normalized=false;int64_t swiglu=0;const auto slots=aggregate_slots(g);
  for(int64_t n=0;n<nodes;++n) {
    const auto& node=g.nodes[n];const auto prefix="nodes."+std::to_string(n)+".";
    if(controls&&!node.identity&&node.readout=="linear-v1")by_name[prefix+"read"]={11,n,n*width,{width}};
    const auto aggregate=aggregate_kind(node.aggregation);normalized|=aggregate!=0;
    if(aggregate>=2)for(int64_t slot=0;slot<g.source_counts[n];++slot)
      by_name[prefix+"extra."+(aggregate==2?"agg_mass_":"agg_logit_")+std::to_string(slot)]={10,n*slots+slot,n*slots+slot,{}};
    const auto kind=node.identity?0:lh_full_kind(node.full);
    if(!node.identity&&node.full!="identity"&&node.full!="tanh"&&node.full!="swiglu"&&!kind)throw std::invalid_argument("parameter Full VJP contract unavailable");
    if(!node.identity&&node.full=="tanh") {
      tanh=true;by_name[prefix+"weight"]={0,n,n*width*width,{width,width}};by_name[prefix+"bias"]={1,n,n*width,{width}};
    }
    if(kind) {
      lh=true;const auto norm=(kind-1)%3;
      if(norm)by_name[prefix+"extra.lh_norm_weight"]={5,n,n*width,{width}};
      if(norm==2)by_name[prefix+"extra.lh_norm_bias"]={6,n,n*width,{width}};
    }
    if(!node.identity&&node.full=="swiglu") {
      const auto offset=swiglu++*2*width*width;
      by_name[prefix+"extra.ffn_gate"]={7,n,offset,{width,2*width}};
      by_name[prefix+"extra.ffn_up"]={8,n,offset,{width,2*width}};
      by_name[prefix+"extra.ffn_down"]={9,n,offset,{2*width,width}};
    }
    if(!node.identity&&node.memory=="ema")by_name[prefix+"decay"]={2,n,n*width,{width}};
    else if(!node.identity&&node.memory=="lh-add-repeat-v1")by_name[prefix+"extra.add_retention"]={3,n,n,{}};
    else if(!node.identity&&node.memory=="attention") {
      const auto kv=width/node.query_heads*node.kv_heads,at=attention[n];
      by_name[prefix+"extra.attn_q"]={12,n*4,at,{width,width}};
      by_name[prefix+"extra.attn_k"]={12,n*4+1,at+width*width,{width,kv}};
      by_name[prefix+"extra.attn_v"]={12,n*4+2,at+width*(width+kv),{width,kv}};
      by_name[prefix+"extra.attn_out"]={12,n*4+3,at+width*(width+2*kv),{width,width}};
    }
    else if(!node.identity&&is_fiber_attention_profile(node.memory)) {
      const auto at=fiber[n];
      by_name[prefix+"extra.fiber_qkv"]={13,n*6,at,{width,3*width}};
      by_name[prefix+"extra.fiber_qkv_bias"]={13,n*6+1,at+3*width*width,{3*width}};
      by_name[prefix+"extra.fiber_out"]={13,n*6+2,at+3*width*width+3*width,{width,width}};
      by_name[prefix+"extra.fiber_out_bias"]={13,n*6+3,at+4*width*width+3*width,{width}};
      by_name[prefix+"extra.fiber_decay"]={13,n*6+4,at+4*width*width+4*width,{}};
      if(node.memory!="lh-fiber-attention-sum-repeat-v1"&&node.memory!="lh-fiber-attention-mean-repeat-v1")
        by_name[prefix+"extra.fiber_pool"]={13,n*6+5,at+4*width*width+4*width+1,{g.source_counts[n]}};
    }
    else if(!node.identity&&node.memory!="identity")throw std::invalid_argument("parameter state VJP contract unavailable");
    if(!node.identity&&node.emission!="broadcast")throw std::invalid_argument("parameter graph VJP contract unavailable");
  }
  for(int64_t i=0;i<inputs;++i)by_name["input_scale."+std::to_string(i)]={4,i,i,{}};
  for(int64_t i=0;i<edges;++i)by_name["agg_scale."+std::to_string(i)]={4,inputs+i,inputs+i,{}};
  for(size_t slot=0;slot<g.outgoing_ports.bindings.size();++slot) {
    const auto binding=g.outgoing_ports.bindings[slot];const int64_t i=inputs+edges+slot;
    by_name[(binding.kind?"edge_scale.":"output_scale.")+std::to_string(binding.id)]={4,i,i,{}};
  }
  ParameterPlan out;out.owners=registry.owners();out.has_tanh=tanh;out.has_lh=lh;out.swiglu_count=swiglu;
  out.aggregate_slots=normalized?slots:0;out.attention_elements=attention.back();out.fiber_elements=fiber.back();
  std::vector<int64_t> owners,refs,tiles{0};int64_t total=0;
  for(const auto& owner:out.owners) {
    if(owner.value.scalar_type()!=at::kFloat&&owner.value.scalar_type()!=at::kHalf)
      throw std::invalid_argument("parameter registry must describe FP32/FP16 payload owners");
    const auto first=int64_t(refs.size()/3),size=owner.value.numel();
    for(const auto& name:owner.aliases)if(auto it=by_name.find(name);it!=by_name.end()) {
      const auto& ref=it->second;if(owner.value.sizes()!=at::IntArrayRef(ref.shape))throw std::invalid_argument("parameter alias shape disagrees with graph VJP");
      refs.insert(refs.end(),{ref.bank,ref.connection,ref.offset});
    }
    const auto end=int64_t(refs.size()/3),offset=end>first?total:-1;
    const long double bound=4.L*(total+static_cast<long double>(end>first?size:0))+64.L*(out.owners.size()+1)+16.L*refs.size()+256;
    if(bound>budget)throw std::invalid_argument("parameter owner adjoint tensor budget exceeded");
    out.offsets.push_back(offset);owners.insert(owners.end(),{first,end,offset,size});
    if(offset>=0)total+=size;
    tiles.push_back(tiles.back()+(offset>=0?(size+255)/256:0));
  }
  const int64_t count=out.owners.size(),tasks=tiles.back();
  // The empty registry still allocates dummy tables/buffers for valid kernel
  // arguments. Apply admission even when the owner loop did not execute.
  const long double bytes=4.L*std::max<int64_t>(1,total)+std::max<int64_t>(1,count)+4.L+
    8.L*(std::max<size_t>(4,owners.size())+std::max<size_t>(3,refs.size())+tiles.size());
  if(bytes>budget)throw std::invalid_argument("parameter owner adjoint tensor budget exceeded");
  out.owner_table=std::move(owners);out.references=std::move(refs);out.tiles=std::move(tiles);out.elements=std::max<int64_t>(1,total);return out;
}
ParameterVjp parameter_layout(const Graph& graph,const ParameterRegistry& registry,int64_t width,at::Device device,int64_t budget,bool controls) {
  if(device.type()!=c10::DeviceType::PrivateUse1||device.index()<0)throw std::invalid_argument("parameter layout requires an explicit NPU");
  auto p=plan_parameters(graph,registry,width,budget,controls);auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
  return {p.owners,p.offsets,at::zeros({p.elements},opts),at::zeros({std::max<int64_t>(1,p.owners.size())},opts.dtype(at::kBool))};
}
} // namespace tide::device_online
