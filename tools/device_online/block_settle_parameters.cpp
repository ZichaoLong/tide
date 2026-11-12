#include "block_settle_parameters.h"
namespace tide::device_online {
std::map<std::string,WeightedRowAdjoint> block_kv_local_parameter_roots(const BlockKVLocalGradient& g){
  auto out=block_kv_message_parameter_roots(g.messages);const auto& w=g.sequence.update;
  out["update.base"]={w.parameters.base,w.kv_connected};out["update.left"]={w.parameters.left,w.kv_connected};
  out["update.right"]={w.parameters.right,w.kv_connected};out["update.summary"]={w.parameters.summary,w.summary_connected};
  if(g.aggregate.parameters){const auto& a=g.source_adaptation?g.source_adaptation->base:*g.aggregate.parameters;
    out["source.a"]={a.a,g.aggregate.owner_connected};out["source.b"]={a.b,g.aggregate.owner_connected};out["source.embedding"]={a.embedding,g.aggregate.owner_connected};}
  if(g.source_adaptation){const auto& f=g.source_adaptation->factors;const auto flags=g.source_adaptation->connected;
    out["source.a_left"]={f.a_left,flags};out["source.a_right"]={f.a_right,flags};out["source.b_left"]={f.b_left,flags};out["source.b_right"]={f.b_right,flags};}
  return out;
}
ParameterVjp append_block_settle_parameter_vjp(DeviceProgram& p,const BlockSettleFlowGradient& g,
    const ParameterRegistry& registry,const std::vector<BlockKVParameterRoles>& names,const at::Tensor& error,int64_t budget){
  if(names.size()!=g.regions.size())throw std::invalid_argument("block graph parameter region mapping differs");
  std::vector<WeightedParameterPartial> parts;
  for(size_t r=0;r<names.size();++r){const auto roles=block_kv_local_parameter_roots(g.regions[r]);
    for(const auto& [role,mapping]:names[r]){const auto found=roles.find(role);
      if(found==roles.end()&&role.rfind("source.",0)==0&&!g.regions[r].aggregate.parameters){
        for(const auto& name:mapping)if(!registry.contains(name))throw std::invalid_argument("unknown inactive source parameter name");continue;}
      if(found==roles.end())throw std::invalid_argument("inapplicable block graph parameter role: "+role);
      const auto& root=found->second;
      if(!root.value.defined()||root.value.dim()<1||mapping.size()!=size_t(root.value.size(0))||root.connected.sizes()!=at::IntArrayRef{root.value.size(0)})
        throw std::invalid_argument("block graph parameter bank mapping differs");
      for(size_t bank=0;bank<mapping.size();++bank){if(!registry.contains(mapping[bank]))throw std::invalid_argument("unknown block graph parameter name");
        parts.push_back({mapping[bank],root.value[bank],root.connected[bank]});}
    }
  }
  return append_weighted_parameter_vjp(p,registry,parts,error,budget);
}
std::map<std::string,at::Tensor> block_kv_local_parameters(const BlockKVLocalParameters& w){
  std::map<std::string,at::Tensor> out{{"update.base",w.update.base},{"update.left",w.update.left},{"update.right",w.update.right},{"update.summary",w.update.summary},
    {"read.query",w.message.read.query.query},{"read.key",w.message.read.query.key},{"read.value",w.message.read.query.value},
    {"full.query",w.message.full.query},{"full.out",w.message.full.residual.out},{"full.gate",w.message.full.residual.gate},
    {"full.up",w.message.full.residual.up},{"full.down",w.message.full.residual.down}};
  if(w.source.a.defined()){out["source.a"]=w.source.a;out["source.b"]=w.source.b;out["source.embedding"]=w.source.embedding;}
  if(w.source_factors){const auto& f=*w.source_factors;out["source.a_left"]=f.a_left;out["source.a_right"]=f.a_right;out["source.b_left"]=f.b_left;out["source.b_right"]=f.b_right;}
  for(const auto& [name,x]:w.message.read.describe)out["describe."+name]=x;
  auto head=[&](const std::string& name,const WeightedHeadParameters& h){if(h.matrix.defined()){
    out[name+"matrix"]=h.matrix;out[name+"hidden_bias"]=h.hidden_bias;out[name+"projection"]=h.projection;out[name+"bias"]=h.bias;}};
  head("region.context.",w.message.read.region.context);head("region.query.",w.message.read.region.query);return out;
}
BlockKVLocalParameters block_kv_local_parameters(const std::map<std::string,at::Tensor>& values,const BlockKVLocalParameters& binding){
  auto w=binding;const auto expected=block_kv_local_parameters(binding);
  if(values.size()!=expected.size())throw std::invalid_argument("block parameter roles changed during binding");
  auto get=[&](const std::string& name){const auto& value=values.at(name);
    if(value.sizes()!=expected.at(name).sizes())throw std::invalid_argument("block parameter role shape differs");return value;};
  w.update={get("update.base"),get("update.left"),get("update.right"),get("update.summary")};
  w.message.read.query={get("read.query"),get("read.key"),get("read.value")};
  w.message.full={get("full.query"),{get("full.out"),get("full.gate"),get("full.up"),get("full.down")}};
  if(w.source.a.defined())w.source={get("source.a"),get("source.b"),get("source.embedding")};
  if(w.source_factors)w.source_factors=WeightedSourceFactors{get("source.a_left"),get("source.a_right"),get("source.b_left"),get("source.b_right")};
  for(auto& [name,x]:w.message.read.describe)x=get("describe."+name);
  auto head=[&](const std::string& name,WeightedHeadParameters& h){if(h.matrix.defined())h={get(name+"matrix"),get(name+"hidden_bias"),get(name+"projection"),get(name+"bias")};};
  head("region.context.",w.message.read.region.context);head("region.query.",w.message.read.region.query);return w;
}
}
