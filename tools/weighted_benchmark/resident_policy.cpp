#include "resident_policy.h"
#include "tide/parameters.h"
#include <algorithm>
#include <limits>
#include <set>
namespace weighted_benchmark {
namespace {
std::string capability(const Json& j,const Fixture& f,bool prefill,at::ScalarType dtype,at::Device device,
    const std::vector<at::Device>& peers,const std::vector<at::Device>& owners,const std::string& returns,
    std::optional<SettleGraph>& spec){
  if(returns=="complete"||j.value("__trace_records",std::string("auto"))=="complete")
    return "complete event records retain the general owner";
  if(!prefill||j.value("__input_interface",std::string("auto"))!="auto")return "requires prefill with tensor inputs";
  if(dtype!=at::kFloat)return "requires FP32 payload and scoring";
  if(device.is_cpu()||device.index()<0||!peers.empty()||std::any_of(owners.begin(),owners.end(),[&](auto x){return x!=device;}))
    return "requires one indexed accelerator owner";
  if(!j.contains("workload")||!j.at("workload").contains("stride")||!j.at("workload").contains("graph")||
      !j.at("workload").at("graph").contains("ranks"))return "fixture has no explicit Settle ranks/stride";
  const auto& w=j.at("workload");const auto stride=w.at("stride").get<Index>();
  try{spec.emplace(f.graph,w.at("graph").at("ranks").get<std::vector<Index>>(),"ports",stride);}
  catch(const std::invalid_argument&){return "graph is outside the declared Settle rank/time binding";}
  if(stride<=0||f.positions>std::numeric_limits<Index>::max()/stride||f.span!=f.positions*stride)
    throw std::invalid_argument("fixture span differs from declared Settle positions");
  const auto& g=f.graph;
  if(!g.origins.empty())return "requires physical Settle ports without origin remapping";
  for(const auto& n:g.nodes)if(n.clear||n.identity||n.next_state!="adopt-v1"||!(n.state_clock==StateClock{})||
      n.emit_period!=1||!n.emit_phases.empty())return "requires retained logical-time state and physical broadcasts";
  for(const auto& p:f.model.parameters(false).owners())if(!p.value.device().is_cpu()||p.value.scalar_type()!=at::kFloat)
    return "requires CPU FP32 boundary parameters";
  Index width=-1;
  for(size_t r=0;r<g.regions.size();++r){
    if(!g.regions[r].observe_all)return "requires complete regional observations";
    const auto regional=weighted::region_binding(f.model.regions.at(r));
    std::optional<weighted::KVConfig> memory;double alpha=0;std::set<bool> factors;size_t sources=0;
    for(size_t v=0;v<g.nodes.size();++v)if(g.nodes[v].region==Index(r)){
      const auto& n=f.model.nodes.at(v);const auto b=weighted::node_binding(n);
      const auto* kv=std::get_if<weighted::KVConfig>(&b.memory);
      if(!kv||(kv->memory_read!="sum"&&kv->memory_read!="qry"))return "requires KV sum/query sequence capability";
      if(width<0)width=kv->width;if(width!=kv->width)return "requires one physical payload width";
      if(!weighted::same_configuration(b.regional,regional))return "node/region mathematical bindings differ";
      if(!memory){memory=*kv;alpha=b.aggregate_alpha;}
      else if(!weighted::same_configuration(*memory,*kv)||alpha!=b.aggregate_alpha)
        return "requires compatible KV geometry and Aggregate alpha within each region";
      std::set<Index> labels;
      for(size_t p=0;p<g.inputs.size();++p)if(g.inputs[p]==Index(v))labels.insert(b.sources.inputs.at(p));
      for(size_t e=0;e<g.edges.size();++e)if(g.edges[e].target==Index(v))labels.insert(b.sources.edges.at(e));
      for(auto label:labels){++sources;const auto prefix="aggregate_program.transforms."+std::to_string(label)+".";
        factors.insert(n.extra.count(prefix+"a_left")!=0);}
    }
    if(!memory)return "requires nonempty regions";
    if(alpha!=0&&(!sources||factors.size()>1))return "source factor presence requires compatible nonempty banks";
  }
  return {};
}
}
BlockSelection select_resident_block(const Json& j,const Fixture& f,bool prefill,at::ScalarType dtype,at::Device d,
    const std::vector<at::Device>& peers,const std::vector<at::Device>& owners,const std::string& returns){
  const auto requested=j.value("__resident_executor",std::string("general"));
  if(requested!="auto"&&requested!="general"&&requested!="block-v2")throw std::invalid_argument("invalid resident executor");
  BlockSelection out;std::string reason;
  if(requested=="general")reason="explicit general resident owner";
  else if(j.value("__resident_general_limits",false))reason="explicit historical resident_limits retain the general owner";
  else reason=capability(j,f,prefill,dtype,d,peers,owners,returns,out.graph);
  if(!reason.empty()&&(requested=="block-v2"||j.contains("__resident_block_limits")))
    throw std::invalid_argument("resident block selection refused: "+reason);
  const bool selected=reason.empty();if(!selected)out.graph.reset();
  out.record={{"schema","tide-standalone-resident-selection-v1"},{"requested",requested},
    {"effective",selected?"block-v2":"general"},{"schedule",selected?"settle-block-v2":"device-general"},
    {"interface",selected?"tensor-window-v1":"records"},
    {"reason",selected?"compatible complete B×L Settle ports":reason},
    {"capacity_field",selected?"resident_block_limits":"resident_limits"},
    {"checkpoint_codec",selected?"block-schema-2":"general-resident-v1"}};
  return out;
}
Json resident_block_limits(const Json& j,const Fixture& f){
  Json out={{"cache",64},{"forward_bytes",int64_t{512}<<20},{"backward_bytes",int64_t{1024}<<20},
    {"optimizer_bytes",int64_t{128}<<20},{"total_bytes",int64_t{3}<<30},
    {"workspace_bytes",int64_t{512}<<20},{"windows",4},{"timeout_ms",30000}};
  const auto defaults=out;
  const auto windows=j.value("__logical_windows",Index{1});
  if(windows<1||f.positions<1||windows>std::numeric_limits<Index>::max()/f.positions)
    throw std::invalid_argument("block raw capacity estimate exceeds int64");
  const auto writes=windows*f.positions;Index needed=1;
  for(const auto& n:f.model.nodes){const auto c=std::get<weighted::KVConfig>(weighted::node_binding(n).memory);
    needed=std::max(needed,c.window?std::min(writes,c.window):writes);}
  out["cache"]=std::max<Index>(64,needed);
  if(j.contains("__resident_block_resume_limits")){
    out=j.at("__resident_block_resume_limits");
    if(!out.is_object()||out.size()!=defaults.size())throw std::invalid_argument("saved block capacity schema differs");
  }
  const auto given=j.value("__resident_block_limits",Json::object());
  if(!given.is_object())throw std::invalid_argument("resident block limits must be an object");
  for(const auto& [name,value]:given.items()){
    if(!defaults.contains(name))throw std::invalid_argument("unknown block capacity: "+name);
    out[name]=value;
  }
  for(const auto& [name,value]:out.items()){
    if(!defaults.contains(name))throw std::invalid_argument("unknown block capacity: "+name);
    if(!value.is_number_integer()||value.is_boolean()||value.get<int64_t>()<1||
       (name=="timeout_ms"&&value.get<int64_t>()>std::numeric_limits<int32_t>::max()))
      throw std::invalid_argument("invalid block capacity: "+name);
  }
  return out;
}
}
