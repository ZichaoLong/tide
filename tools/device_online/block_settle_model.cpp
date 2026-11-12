#include "block_settle_model.h"
#include "block_parameter_views.h"
#include "weighted_flow_parameters.h"
#include "tide/weighted_column_checkpoint.h"
#include "tide/ops.h"
#include "tide/kernel.h"
#include <set>
namespace tide::device_online {
namespace {
void require(bool x,const char* message){if(!x)throw std::invalid_argument(message);}
std::string local_role(const std::string& name){
  if(name.rfind("upd.",0)==0)return "update."+name.substr(4);
  if(name.rfind("mem.",0)==0)return "read."+name.substr(4);return name;
}
}
BlockSettleModel lower_block_settle_model(const SettleGraph& spec,const Model& model,const Continuation& q,
    int64_t capacity,int64_t budget){
  require(capacity>0&&budget>0,"block Model lowering requires positive raw/binding capacity");
  weighted::columns::validate_owner(spec,model,q,q.batch_size);weighted::validate_transport(spec.graph(),model);
  const auto& g=spec.graph();require(model.nodes.size()==g.nodes.size()&&model.regions.size()==g.regions.size(),"block Model graph domains differ");
  validate_model(g,model);auto structural=g;
  for(auto& node:structural.nodes){require(!node.clear&&!node.identity&&node.next_state=="adopt-v1"&&node.state_clock==StateClock{}&&node.emit_period==1&&node.emit_phases.empty(),"block Model requires retained proposal logical-time state and physical broadcast");
    // validate_model and installed node_binding below establish the weighted
    // local program. The lowerer transports concrete (c,G), not profile tags.
    node.emission="broadcast";node.emit_period=1;node.emit_phases.clear();}
  structural.compile();BlockSettleModel out{SettleGraph(structural,spec.ranks(),"ports",spec.stride()),{},model.parameters(false),{},{},{}};
  long double charged=0;auto charge=[&](const at::Tensor& x){charged+=x.nbytes();require(charged<=budget,"block Model packed binding budget exceeded");};
  for(const auto& p:out.registry.owners())require(p.value.device().is_cpu()&&p.value.scalar_type()==at::kFloat,"block Model canonical parameters must be CPU FP32");
  for(size_t region=0;region<g.regions.size();++region){
    require(g.regions[region].observe_all,"block Model requires complete-region observations");
    const auto config=weighted::region_binding(model.regions[region]);BlockSettleKVBinding binding;binding.regional=config;
    binding.input_owners.assign(g.inputs.size(),0);binding.edge_owners.assign(g.edges.size(),0);BlockKVParameterRoles names;
    std::map<std::string,std::vector<at::Tensor>> row_values;std::vector<int64_t> nodes;
    for(size_t v=0;v<g.nodes.size();++v)if(g.nodes[v].region==int64_t(region))nodes.push_back(v);
    require(!nodes.empty(),"block Model has an empty region");
    std::vector<State> states,bound;std::map<std::pair<int64_t,int64_t>,int64_t> source_bank;
    for(size_t n=0;n<nodes.size();++n){const auto v=nodes[n];const auto& w=model.nodes[v];const auto b=weighted::node_binding(w);
      require(std::holds_alternative<weighted::KVConfig>(b.memory),"block Model local family requires the KV sequence implementation");
      const auto c=std::get<weighted::KVConfig>(b.memory);require(c.memory_read=="qry","block Model Read implementation requires KV qry");
      require(weighted::same_configuration(b.regional,config),"block Model node/region configuration differs");
      if(!n){binding.memory=c;binding.aggregate_alpha=b.aggregate_alpha;}
      else require(weighted::same_configuration(binding.memory,c)&&binding.aggregate_alpha==b.aggregate_alpha,"block Model region needs one compatible KV/source geometry");
      binding.node_owners.push_back(n);const auto prefix="nodes."+std::to_string(v)+".extra.";WeightedBanks banks(w.extra,prefix,at::kCPU);
      const auto remaining=[&](){require(charged<budget,"block Model bank budget exhausted");return int64_t(budget-charged);};
      auto update=bind_weighted_kv_update(banks,remaining());auto query=bind_weighted_kv_query(banks,remaining());
      auto full=bind_weighted_kv_full(banks,remaining());auto describe=bind_weighted_describe(banks,c.width,c.summary_width,config,remaining());
      for(const auto& [role,part]:banks.roles()){const auto name=local_role(role);row_values[name].push_back(part.bank[0]);names[name].push_back(part.parameter);charge(part.bank);}
      std::set<int64_t> labels;
      for(size_t p=0;p<g.inputs.size();++p)if(g.inputs[p]==v)labels.insert(b.sources.inputs[p]);
      for(size_t e=0;e<g.edges.size();++e)if(g.edges[e].target==v)labels.insert(b.sources.edges[e]);
      for(auto label:labels){const auto index=int64_t(source_bank.size());source_bank[{v,label}]=index;
        if(b.aggregate_alpha){const auto path="aggregate_program.transforms."+std::to_string(label)+".";
          for(const auto& role:std::vector<std::string>{"a","b","embedding","a_left","a_right","b_left","b_right"}){
            const auto found=w.extra.find(path+role);if(found==w.extra.end()){require(role=="a_left"||role=="a_right"||role=="b_left"||role=="b_right","missing block source parameter");continue;}
            row_values["source."+role].push_back(found->second.detach());names["source."+role].push_back(prefix+path+role);}}
      }
      for(size_t p=0;p<g.inputs.size();++p)if(g.inputs[p]==v)binding.input_owners[p]=source_bank.at({v,b.sources.inputs[p]});
      for(size_t e=0;e<g.edges.size();++e)if(g.edges[e].target==v)binding.edge_owners[e]=source_bank.at({v,b.sources.edges[e]});
    }
    // Existing local rows use sample-major private state, then region node.
    for(int64_t b=0;b<q.batch_size;++b)for(auto v:nodes){const auto& w=model.nodes[v];auto initial=w.kernel->initial(w);
      bound.push_back(initial);const auto found=q.states.find({b,v});states.push_back(found==q.states.end()?initial:found->second);w.kernel->validate_state(w,states.back());}
    WeightedBanks region_banks(model.regions[region].extra,"regions."+std::to_string(region)+".",at::kCPU);
    binding.parameters.message.read.region=bind_weighted_region(region_banks,config,1,int64_t(budget-charged));
    for(const auto& [role,part]:region_banks.roles()){row_values[role].push_back(part.bank[0]);names[role].push_back(part.parameter);charge(part.bank);}
    int64_t hidden=0,rank=0;if(binding.aggregate_alpha){
      require(row_values.count("source.a"),"block Model needs a nonempty source bank for nonzero Aggregate alpha");
      for(const auto& x:row_values.at("source.a"))hidden=std::max(hidden,x.size(0));
      for(const auto& role:{"source.a_left","source.b_left"})if(row_values.count(role)){
        require(row_values.at(role).size()==row_values.at("source.a").size(),"block Model source factor presence differs across banks");
        for(const auto& x:row_values.at(role))rank=std::max(rank,x.size(1));}}
    std::map<std::string,at::Tensor> packed;for(const auto& [role,rows]:row_values){auto shape=rows.front().sizes().vec();
      if(role=="source.a")shape={hidden,binding.memory.width};else if(role=="source.b")shape={binding.memory.width,hidden};
      else if(role=="source.embedding")shape={hidden};else if(role=="source.a_left")shape={hidden,rank};
      else if(role=="source.a_right")shape={rank,binding.memory.width};else if(role=="source.b_left")shape={binding.memory.width,rank};
      else if(role=="source.b_right")shape={rank,hidden};
      std::vector<at::Tensor> padded;for(const auto& x:rows)padded.push_back(block_parameter_pack(role,x,shape));
      packed[role]=at::stack(padded);charge(packed[role]);}
    auto get=[&](const std::string& role){return packed.at(role);};
    binding.parameters.update={get("update.base"),get("update.left"),get("update.right"),get("update.summary")};
    binding.parameters.message.read.query={get("read.query"),get("read.key"),get("read.value")};
    for(const auto& [role,value]:packed)if(role.rfind("describe.",0)==0)binding.parameters.message.read.describe[role.substr(9)]=value;
    binding.parameters.message.full={get("full.query"),{get("full.out"),get("full.gate"),get("full.up"),get("full.down")}};
    if(binding.aggregate_alpha){binding.parameters.source={get("source.a"),get("source.b"),get("source.embedding")};
      if(packed.count("source.a_left"))binding.parameters.source_factors=WeightedSourceFactors{get("source.a_left"),get("source.a_right"),get("source.b_left"),get("source.b_right")};}
    auto head=[&](const std::string& role,WeightedHeadParameters& h){if(packed.count(role+"matrix"))h={get(role+"matrix"),get(role+"hidden_bias"),get(role+"projection"),get(role+"bias")};};
    head("region.context.",binding.parameters.message.read.region.context);head("region.query.",binding.parameters.message.read.region.query);
    binding.read_mode=g.regions[region].read_mode=="proposal"?"+":g.regions[region].read_mode=="old"?"-":g.regions[region].read_mode=="content"?"0":"";
    require(!binding.read_mode.empty(),"block Model unknown Read snapshot binding");
    WeightedRawLayout layout(binding.memory,capacity);out.initial.push_back(encode_weighted_raw_state(layout,states,budget));out.bound_initial.push_back(encode_weighted_raw_state(layout,bound,budget));
    charge(out.initial.back().floating);charge(out.initial.back().integer);charge(out.bound_initial.back().floating);charge(out.bound_initial.back().integer);
    out.bindings.push_back(std::move(binding));out.roles.push_back(std::move(names));
  }
  return out;
}
}
