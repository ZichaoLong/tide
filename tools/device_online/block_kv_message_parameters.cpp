#include "block_kv_messages.h"

namespace tide::device_online {
std::map<std::string,WeightedRowAdjoint> block_kv_message_parameter_roots(const BlockKVMessageGradient& g){
  std::map<std::string,WeightedRowAdjoint> roles{
    {"full.query",{g.full.parameters.query,g.full.query_connected}},
    {"full.out",{g.full.parameters.residual.out,g.full.out_connected}},
    {"full.gate",{g.full.parameters.residual.gate,g.full.gate_connected}},
    {"full.up",{g.full.parameters.residual.up,g.full.up_connected}},
    {"full.down",{g.full.parameters.residual.down,g.full.down_connected}}};
  if(g.read.query.query.defined()){
    roles["read.query"]={g.read.query.query,g.read.query_connected};roles["read.key"]={g.read.query.key,g.read.key_connected};
    roles["read.value"]={g.read.query.value,g.read.value_connected};}
  for(const auto& [name,root]:g.read.describe)roles["describe."+name]=root;
  auto head=[&](const std::string& prefix,const std::optional<WeightedHeadGradient>& h){if(!h)return;
    roles[prefix+"matrix"]={h->parameters.matrix,h->owner_connected};
    roles[prefix+"hidden_bias"]={h->parameters.hidden_bias,h->owner_connected};
    roles[prefix+"projection"]={h->parameters.projection,h->owner_connected};
    roles[prefix+"bias"]={h->parameters.bias,h->owner_connected};};
  head("region.context.",g.read.region_context);head("region.query.",g.read.region_query);
  return roles;
}
ParameterVjp append_block_kv_message_parameter_vjp(DeviceProgram& p,const BlockKVMessageGradient& g,
    const ParameterRegistry& registry,const BlockKVParameterRoles& names,const at::Tensor& error,int64_t budget){
  const auto roles=block_kv_message_parameter_roots(g);
  std::vector<WeightedParameterPartial> partials;
  for(const auto& [role,mapping]:names){
    const auto found=roles.find(role);if(found==roles.end())throw std::invalid_argument("inapplicable KV message parameter role: "+role);
    const auto& root=found->second;
    if(!root.value.defined()||root.value.dim()<1||mapping.size()!=size_t(root.value.size(0))||
       root.connected.sizes()!=at::IntArrayRef{root.value.size(0)})throw std::invalid_argument("KV message parameter bank mapping differs");
    for(size_t bank=0;bank<mapping.size();++bank){
      if(!registry.contains(mapping[bank]))throw std::invalid_argument("unknown KV message canonical parameter name");
      partials.push_back({mapping[bank],root.value[bank],root.connected[bank]});
    }
  }
  return append_weighted_parameter_vjp(p,registry,partials,error,budget);
}
}
