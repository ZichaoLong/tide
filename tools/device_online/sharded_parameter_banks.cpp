#include "sharded_parameter_banks.h"
#include "tide/fiber_attention.h"
#include "packed_lh_full.h"
#include "packed_aggregate.h"
#include <stdexcept>
namespace tide::device_online {
std::map<std::string,ParameterDestination> sharded_parameter_destinations(const ShardedParameterBanks& banks) {
  const auto& b=banks.coordinator;
  if(!b.graph||!b.decay.defined()||b.decay.dim()!=2)throw std::invalid_argument("incomplete sharded parameter banks");
  const auto& g=*b.graph;const auto dtype=b.decay.scalar_type();const auto width=b.decay.size(1);
  if(dtype!=at::kFloat&&dtype!=at::kHalf)throw std::invalid_argument("sharded publication requires FP32/FP16 payloads");
  std::map<std::string,ParameterDestination> out;
  auto add=[&](const std::string& name,at::Tensor value) {
    if(!value.defined()||value.device().type()!=c10::DeviceType::PrivateUse1||value.requires_grad()
        ||(value.scalar_type()!=dtype&&value.scalar_type()!=at::kFloat)||value.numel()<1
        ||!out.emplace(name,ParameterDestination{value,dtype}).second)
      throw std::invalid_argument("invalid sharded parameter destination");
  };
  std::vector<bool> seen(g.nodes.size(),false);
  for(const auto& shard:banks.full) {
    int64_t swiglu=0;const auto& f=shard.full;
    for(size_t i=0;i<shard.nodes.size();++i) {
      const auto n=shard.nodes[i];
      if(n<0||n>=int64_t(seen.size())||seen[n])throw std::invalid_argument("invalid Full publication node map");
      seen[n]=true;const auto& node=g.nodes[n];if(node.identity)continue;
      const auto prefix="nodes."+std::to_string(n)+".";const auto kind=lh_full_kind(node.full);
      if(node.full=="tanh"){add(prefix+"weight",f.weights[i]);add(prefix+"bias",f.biases[i]);}
      if(kind) {
        const auto norm=(kind-1)%3;
        if(norm)add(prefix+"extra.lh_norm_weight",f.extra.lh_weights[i]);
        if(norm==2)add(prefix+"extra.lh_norm_bias",f.extra.lh_biases[i]);
      }
      if(node.full=="swiglu") {
        add(prefix+"extra.ffn_gate",f.extra.gate[swiglu]);add(prefix+"extra.ffn_up",f.extra.up[swiglu]);
        add(prefix+"extra.ffn_down",f.extra.down[swiglu++]);
      }
    }
  }
  for(bool yes:seen)if(!yes)throw std::invalid_argument("incomplete Full publication node map");
  for(size_t n=0;n<g.nodes.size();++n) {
    const auto& node=g.nodes[n];const auto prefix="nodes."+std::to_string(n)+".";
    const auto aggregate=aggregate_kind(node.aggregation);
    if(aggregate>=2)for(int64_t slot=0;slot<g.source_counts[n];++slot)
      add(prefix+"extra."+(aggregate==2?"agg_mass_":"agg_logit_")+std::to_string(slot),b.aggregate.weights[n][slot]);
    if(node.identity)continue;
    add(prefix+"read",b.read[n]);
    if(node.memory=="ema")add(prefix+"decay",b.decay[n]);
    if(node.memory=="lh-add-repeat-v1")add(prefix+"extra.add_retention",b.retention[n]);
  }
  for(size_t i=0;i<g.inputs.size();++i)add("input_scale."+std::to_string(i),b.sources[i]);
  for(size_t i=0;i<g.edges.size();++i)add("agg_scale."+std::to_string(i),b.sources[g.inputs.size()+i]);
  for(size_t i=0;i<g.outgoing_ports.bindings.size();++i) {
    const auto binding=g.outgoing_ports.bindings[i];add((binding.kind?"edge_scale.":"output_scale.")+std::to_string(binding.id),b.emission[i][0]);
  }
  for(const auto& a:b.attention) {
    const int64_t kv=width/a.heads*a.kv_heads;
    for(size_t i=0;i<a.nodes.size();++i) {
      const auto prefix="nodes."+std::to_string(a.nodes[i])+".extra.";
      add(prefix+"attn_q",a.qkv[i].narrow(1,0,width));add(prefix+"attn_k",a.qkv[i].narrow(1,width,kv));
      add(prefix+"attn_v",a.qkv[i].narrow(1,width+kv,kv));add(prefix+"attn_out",a.projection[i]);
    }
  }
  for(size_t i=0;i<b.fiber.nodes.size();++i) {
    const auto n=b.fiber.nodes[i];const auto prefix="nodes."+std::to_string(n)+".extra.";
    add(prefix+"fiber_qkv",b.fiber.qkv[i]);add(prefix+"fiber_qkv_bias",b.fiber.qkv_bias[i]);
    add(prefix+"fiber_out",b.fiber.projection[i]);add(prefix+"fiber_out_bias",b.fiber.projection_bias[i]);add(prefix+"fiber_decay",b.fiber.decay[i]);
    if(g.nodes[n].memory!="lh-fiber-attention-sum-repeat-v1"&&g.nodes[n].memory!="lh-fiber-attention-mean-repeat-v1")
      add(prefix+"fiber_pool",b.fiber.pool[i].narrow(0,0,g.source_counts[n]));
  }
  return out;
}
} // namespace tide::device_online
