#include "tide/weighted_column_checkpoint.h"
#include "weighted_column_modules.h"
#include "tide/kernel.h"
#include "tide/region.h"
#include <algorithm>
#include <limits>

namespace tide::weighted::columns {
namespace {
void require(bool value,const char* text){if(!value)throw std::invalid_argument(text);}
void valid_owner(Owner owner,Index samples,Index count){
  require(owner.first>=0&&owner.first<samples&&owner.second>=0&&owner.second<count,
          "column checkpoint owner outside domain");
}
void validate_window(const SettleGraph& spec,const Layout& layout,Index stop,const Continuation& q){
  layout.validate();
  require(stop>=0&&stop<=std::numeric_limits<Index>::max()/spec.stride(),"invalid column checkpoint stop position");
  const auto begin=q.cut/spec.stride();
  require(stop>=begin,"column checkpoint stop precedes its cut");
  for(auto t:layout.times)require(t>=begin&&t<stop,"column checkpoint window outside its complete cut");
}
std::vector<Index> sample_owners(const Layout& layout){
  std::vector<Index> owners(layout.times.size());
  for(Index b=0;b<layout.samples();++b)
    std::fill(owners.begin()+layout.offsets[b],owners.begin()+layout.offsets[b+1],b);
  return owners;
}
}
void validate_owner(const SettleGraph& spec,const Model& model,const Continuation& q,Index samples){
  const auto& g=spec.graph();
  require(spec.binding()=="ports"&&samples>0&&q.identity==g.identity&&q.batch_size==samples&&
    q.cut>=0&&q.cut%spec.stride()==0&&q.pending.empty(),
    "column checkpoint requires matching complete-position owner with no pending work");
  for(const auto& [owner,s]:q.states){
    valid_owner(owner,samples,g.nodes.size());
    require(s.last_time<q.cut,"column checkpoint state is beyond the cut");
  }
  for(const auto& [owner,h]:q.history){
    valid_owner(owner,samples,g.regions.size());
    model.regions.at(owner.second).kernel->validate_history(h,region_layout(g,owner.second));
    require(h.last_time<q.cut,"column checkpoint history is beyond the cut");
  }
  for(const auto& [owner,entry]:q.ledger){
    valid_owner(owner,samples,g.inputs.size());
    require(entry.first>=0&&entry.second>=0&&entry.second<q.cut,"column checkpoint ledger outside domain");
  }
  spec.validate_alignment(q,{});
}
std::map<Index,LocalState> import_state(const SettleGraph& spec,const Model& model,const Continuation& q){
  validate_owner(spec,model,q,q.batch_size);std::map<Index,LocalState> states;
  for(Index v=0;v<static_cast<Index>(model.nodes.size());++v){
    const auto& w=model.nodes[v];const auto config=node_binding(w).memory;
    std::vector<State> values;values.reserve(q.batch_size);
    for(Index b=0;b<q.batch_size;++b){
      auto found=q.states.find({b,v});values.push_back(found==q.states.end()?w.kernel->initial(w):found->second);
      w.kernel->validate_state(w,values.back());
    }
    auto result=std::visit([&](const auto& c)->LocalState{
      using S=std::conditional_t<std::is_same_v<std::decay_t<decltype(c)>,KVConfig>,KVState,RecurrentState>;
      return S::from_states(values);
    },config);
    validate_columns(w,config,result);states.emplace(v,std::move(result));
  }
  return states;
}
Continuation export_state(const SettleGraph& spec,const Model& model,const Result& result,
                          const std::map<Index,Message>& ports,Index stop,const Continuation& previous){
  const auto& g=spec.graph();const auto& layout=result.layout;
  validate_owner(spec,model,previous,layout.samples());validate_window(spec,layout,stop,previous);
  auto q=previous;const auto owners=sample_owners(layout);
  for(const auto& [v,node]:result.nodes){
    require(v>=0&&v<static_cast<Index>(g.nodes.size()),"column checkpoint node outside graph");
    layout.select(node.rows);
    const auto decoded=std::visit([](const auto& s){return s.states();},result.states.at(v));
    require(decoded.size()==static_cast<size_t>(q.batch_size),"column checkpoint state sample count differs");
    std::map<Index,Index> last;
    for(auto row:node.rows)last[owners[row]]=row;
    for(const auto& [b,row]:last){
      q.states[{b,v}]=decoded[b];
      const auto time=layout.times[row]*spec.stride()+spec.rank(v);
      auto& h=q.history[{b,g.nodes[v].region}];
      // Built-in weighted regional history is only the actual last query time.
      h.last_time=std::max(h.last_time,time);
    }
  }
  for(const auto& [p,message]:ports){
    require(p>=0&&p<static_cast<Index>(g.inputs.size()),"column checkpoint input port outside graph");
    layout.select(message.rows);std::map<Index,std::vector<Index>> rows;
    for(auto row:message.rows)rows[owners[row]].push_back(row);
    for(const auto& [b,entries]:rows){
      const auto found=q.ledger.find({b,p});const Owner before=found==q.ledger.end()?Owner{-1,-1}:found->second;
      const auto rank=spec.rank(g.inputs[p]);
      const auto first=layout.times[entries.front()]*spec.stride()+rank;
      const auto last=layout.times[entries.back()]*spec.stride()+rank;
      const auto count=static_cast<Index>(entries.size());
      require(first>before.second&&before.first<=std::numeric_limits<Index>::max()-count,
              "column checkpoint ledger fails monotonicity/capacity");
      q.ledger[{b,p}]={before.first+count,last};
    }
  }
  q.cut=stop*spec.stride();return q;
}
WindowExecutor::WindowExecutor(SettleGraph spec,Model model,Index workspace,std::optional<ExecutionPlacement> placement)
    :spec_(std::move(spec)),model_(std::move(model)),placement_(std::move(placement)),max_workspace_bytes_(workspace){
  require(spec_.binding()=="ports"&&workspace>0,"column window requires ports binding and positive workspace");
  require(model_.nodes.size()==spec_.graph().nodes.size()&&model_.regions.size()==spec_.graph().regions.size(),
          "column window graph/model owner domain differs");
  validate_transport(spec_.graph(),model_);
  for(const auto& w:model_.nodes)node_binding(w);
  for(const auto& w:model_.regions)region_binding(w);
}
WindowResult WindowExecutor::run_ports(const Continuation& q,const Layout& layout,
                                      const std::map<Index,Message>& ports,Index stop) const {
  validate_owner(spec_,model_,q,layout.samples());validate_window(spec_,layout,stop,q);
  auto result=settle(spec_,model_,layout,ports,import_state(spec_,model_,q),max_workspace_bytes_,placement_);
  auto continuation=export_state(spec_,model_,result,ports,stop,q);
  return {std::move(result),std::move(continuation)};
}
}
