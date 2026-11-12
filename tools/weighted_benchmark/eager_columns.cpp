#include "eager_columns.h"
#include "tide/communication.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <limits>

namespace weighted_benchmark {
namespace col=weighted::columns;
namespace {
void require(bool ok,const char* text){if(!ok)throw std::invalid_argument(text);}
Index sample(const col::Layout& layout,Index row){
  return std::upper_bound(layout.offsets.begin(),layout.offsets.end(),row)-layout.offsets.begin()-1;
}
}
EagerColumns::EagerColumns(SettleGraph spec,const Fixture& f,ExecutionPlacement placement)
  :executor_(std::move(spec),f.model,Index{4}*1024*1024*1024,placement),positions_(f.positions){
  at::NoGradGuard guard;
  require(f.samples<=std::numeric_limits<Index>::max()/f.positions,"column fixture size overflow");
  layout_.offsets={0};
  for(Index b=0;b<f.samples;++b){
    for(Index p=0;p<f.positions;++p)layout_.times.push_back(p);
    layout_.offsets.push_back(layout_.times.size());
  }
  std::map<Index,std::map<Index,const External*>> ports;
  executor_.spec().validate_alignment(Continuation{},f.inputs);
  for(const auto& x:f.inputs){
    require(x.batch>=0&&x.batch<f.samples,"column fixture sample outside domain");
    const auto position=x.time/executor_.spec().stride();
    require(position>=0&&position<f.positions,"column fixture position outside domain");
    require(ports[x.port].emplace(x.batch*f.positions+position,&x).second,"duplicate column fixture port position");
  }
  // Broadcast ports share the same material column only when the actual input
  // tensor handles agree. Equal values alone never establish shared roots.
  std::map<std::vector<const void*>,Tensor> packed;
  for(const auto& [p,entries]:ports){
    std::vector<Tensor> values;std::vector<const void*> key;std::map<Index,Index> occurrences;
    for(const auto& [row,x]:entries){
      require(x->position==occurrences[x->batch]++,"column fixture requires canonical per-port occurrences");
      rows_[p].push_back(row);values.push_back(x->value);key.push_back(x->value.unsafeGetTensorImpl());
    }
    if(!packed.count(key))packed.emplace(key,at::stack(values).detach());
    values_.emplace(p,packed.at(key));devices_.emplace(p,f.model.nodes[f.graph.inputs[p]].bias.device());
  }
}
col::WindowResult EagerColumns::run(const Continuation& q,Index index,bool training)const{
  require(index>=0&&index<=std::numeric_limits<Index>::max()/positions_-1,"column window position overflow");
  auto layout=layout_;for(auto& t:layout.times)t+=index*positions_;
  std::map<Index,col::Message> ports;std::map<std::pair<const void*,std::string>,Tensor> copies;
  for(const auto& [p,value]:values_){
    const auto device=devices_.at(p);const auto key=std::make_pair(value.unsafeGetTensorImpl(),device.str());
    if(!copies.count(key)){
      auto x=value.detach().clone().set_requires_grad(training);
      copies[key]=x.device()==device?x:communication_copy(x,device,x.scalar_type());
    }
    auto x=copies.at(key);ports[p]={rows_.at(p),x.select(1,0),x.slice(1,1)};
  }
  return executor_.run_ports(q,layout,ports,(index+1)*positions_);
}
std::unique_ptr<EagerColumns> column_owner(const Json& j,const Fixture& f,bool prefill,bool scalar,
    const Options& options,ExecutionPlacement placement,const std::string& returns,std::string& fallback){
  const auto input=j.value("__input_interface",std::string("auto"));
  const auto query=j.value("__query_execution",std::string("auto"));
  require(input=="auto"||input=="records","invalid input-interface policy");
  if(input!="auto")fallback="explicit record input interface";
  else if(returns=="complete"||options.trace)fallback="complete record/intermediate-root contract";
  else if(!prefill||scalar||!options.component_window)fallback="explicit scalar/streaming/greedy schedule";
  else if(options.workers!=1||(query!="auto"&&query!="dense-v1"))fallback="explicit worker or query algorithm";
  else if(!j.contains("workload")||!j.at("workload").contains("stride")||!j.at("workload").at("graph").contains("ranks"))
    fallback="fixture has no explicit Settle ranks/stride";
  if(!fallback.empty())return nullptr;
  const auto stride=j.at("workload").at("stride").get<Index>();
  auto ranks=j.at("workload").at("graph").at("ranks").get<std::vector<Index>>();
  std::optional<SettleGraph> spec;
  try{spec.emplace(f.graph,std::move(ranks),"ports",stride);}
  catch(const std::invalid_argument&){fallback="graph is outside the declared Settle rank/time binding";return nullptr;}
  if(stride<=0||f.positions>std::numeric_limits<Index>::max()/stride||f.span!=f.positions*stride)
    throw std::invalid_argument("fixture span differs from declared Settle positions");
  for(size_t v=0;v<f.model.nodes.size();++v){
    const auto& n=f.graph.nodes[v];const auto& w=f.model.nodes[v];
    if(n.clear||n.next_state!="adopt-v1"||!f.graph.regions[n.region].observe_all){fallback="custom local/state-adoption program";return nullptr;}
    if(!resolve_placement(placement,w.bias.device()).events.is_cpu()){fallback="placement requires device-resident scheduling";return nullptr;}
    weighted::node_binding(w); // Fixture owns validated built-in local functions.
  }
  return std::make_unique<EagerColumns>(std::move(*spec),f,placement);
}
Tensor column_loss(const col::Result& r,bool allow_empty,double scale){
  std::vector<Tensor> parts;
  for(const auto& [_,p]:r.outputs)if(!p.rows.empty()){
    auto c=p.strength,u=p.content;if(u.scalar_type()==at::kHalf){c=c.to(at::kFloat);u=u.to(at::kFloat);}
    auto part=(c.unsqueeze(1)*u).square().sum()*scale;
    if(!parts.empty()&&part.device()!=parts.front().device())
      part=communication_copy(part,parts.front().device(),part.scalar_type());
    parts.push_back(part);
  }
  if(parts.empty()){
    if(allow_empty)return at::zeros({});
    throw std::runtime_error("benchmark produced no outputs");
  }
  return at::stack(parts).sum();
}
Json column_work(const col::WindowResult& r){
  const auto& x=r.columns;Index outputs=0;Json rows=Json::array();
  std::vector<Index> candidates(x.layout.samples()),messages(candidates),count(candidates);
  for(const auto& [_,node]:x.nodes)for(auto row:node.rows)++candidates[sample(x.layout,row)];
  for(const auto& [_,p]:x.messages)for(auto row:p.rows)++messages[sample(x.layout,row)];
  for(const auto& [_,p]:x.outputs){outputs+=p.rows.size();for(auto row:p.rows)++count[sample(x.layout,row)];}
  for(Index b=0;b<x.layout.samples();++b)rows.push_back({{"sample",b},{"candidates",candidates[b]},
    {"messages",messages[b]},{"outputs",count[b]}});
  auto stats=x.stats;stats["recorded_trace"]=0;stats["column_windows"]=1;
  return {{"candidate_updates",stats.at("candidate_events")},{"full_calls",stats.at("selected_events")},
    {"messages",stats.at("visited_edges")},{"outputs",outputs},{"pending",r.continuation.pending.size()},
    {"actual_stats",stats},{"boundary_activity",{{"scope","actual column row metadata; no event trace or floating support reconstruction"},{"samples",rows}}}};
}
Result column_projected(const SettleGraph& spec,const col::WindowResult& r,const std::string& returns){
  at::NoGradGuard guard;Result out;
  out.continuation.identity=r.continuation.identity;out.continuation.batch_size=r.continuation.batch_size;
  out.continuation.cut=r.continuation.cut;out.stats=r.columns.stats;out.stats["recorded_trace"]=0;out.stats["column_windows"]=1;
  if(returns=="metrics")return out;
  require(returns=="outputs","column return requires outputs or metrics");
  for(const auto& [p,value]:r.columns.outputs){
    auto packed=at::cat({value.strength.unsqueeze(1),value.content},1).detach();
    packed=communication_copy(packed,at::Device(at::kCPU),packed.scalar_type());
    for(size_t j=0;j<value.rows.size();++j){const auto row=value.rows[j];
      out.outputs.push_back({sample(r.columns.layout,row),r.columns.layout.times[row]*spec.stride()+spec.rank(spec.graph().outputs[p]),p,packed[j]});}
  }
  std::sort(out.outputs.begin(),out.outputs.end(),[&](const Output& a,const Output& b){
    return std::tie(a.time,a.batch,spec.graph().outputs[a.port],a.port)<std::tie(b.time,b.batch,spec.graph().outputs[b.port],b.port);});
  return out;
}
}
