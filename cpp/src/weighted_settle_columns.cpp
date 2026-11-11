// Settle §9–11: independent region-order host schedule with actual column support.
#include "weighted_column_modules.h"
#include "weighted_mean.h"
#include "weighted_source.h"
#include "finite_guard.h"
#include <set>
#include <numeric>
namespace tide::weighted::columns {
using detail::require;
namespace {
struct Received {std::vector<Index> rows;Tensor mass,content;};
std::vector<Index> positions(const std::vector<Index>& domain,const std::vector<Index>& rows){
  std::vector<Index> ids;for(auto row:rows)ids.push_back(std::lower_bound(domain.begin(),domain.end(),row)-domain.begin());return ids;
}
Received aggregate(const NodeWeights& w,const NodeBinding& binding,const std::vector<std::pair<const Message*,Index>>& sources){
  std::set<Index> present;for(const auto& [p,c]:sources)present.insert(p->rows.begin(),p->rows.end());
  Received out;out.rows={present.begin(),present.end()};std::vector<Tensor> masses,values;
  const auto width=std::visit([](const auto& c){return c.width;},binding.memory);
  const auto epsilon=std::visit([](const auto& c){return c.epsilon;},binding.memory);
  const auto n=static_cast<Index>(out.rows.size());
  for(const auto& [p,label]:sources){
    if(p->rows.empty())continue;auto z=p->content;
    if(binding.aggregate_alpha){
      auto prefix="aggregate_program.transforms."+std::to_string(label);
      auto [a,b]=detail::source_matrices(w,prefix);
      z=z+binding.aggregate_alpha*project(at::silu(project(normalize(z,epsilon),a)+detail::weight(w,prefix+".embedding")),b);
    }
    auto ids=indices(positions(out.rows,p->rows),z.device());
    masses.push_back(at::zeros({n},z.options()).index_copy(0,ids,p->strength.clamp_min(0)));
    values.push_back(at::zeros({n,width},z.options()).index_copy(0,ids,z));
  }
  if(out.rows.empty())return out;
  auto mass=at::stack(masses,1),value=at::stack(values,1);out.mass=mass.sum(1);
  auto positive=(out.mass.detach()>0).nonzero().flatten();out.content=at::zeros({n,width},value.options());
  if(positive.numel())out.content=out.content.index_copy(0,positive,weighted_mean(mass.index_select(0,positive),value.index_select(0,positive)));
  return out;
}
}
Result settle(const SettleGraph& spec,const Model& model,const Layout& layout,const std::map<Index,Message>& ports,
              const std::map<Index,LocalState>& left,Index max_workspace_bytes){
  layout.validate();const auto& g=spec.graph();const auto& ref=model.nodes.at(0).bias;
  require(spec.binding()=="ports","column Settle requires ports binding");validate_transport(g,model);
  require(max_workspace_bytes>0,"column workspace budget must be positive");
  require(ref.scalar_type()==at::kFloat||ref.scalar_type()==at::kDouble,"column Settle requires FP32/FP64");
  const Index maxrank=*std::max_element(spec.ranks().begin(),spec.ranks().end());
  for(auto t:layout.times)require(t<=(std::numeric_limits<Index>::max()-maxrank)/spec.stride(),"column logical time overflow");
  for(const auto& [port,p]:ports){
    require(port>=0&&port<static_cast<Index>(g.inputs.size()),"unknown column input port");layout.select(p.rows);
    require(p.content.sizes()==at::IntArrayRef({static_cast<Index>(p.rows.size()),ref.numel()-1})
      &&p.strength.sizes()==at::IntArrayRef({static_cast<Index>(p.rows.size())}),"message column shapes differ");
    for(const auto& x:{p.strength,p.content})require(x.device()==ref.device()&&x.scalar_type()==ref.scalar_type()
      &&at::isfinite(x).all().item<bool>(),"invalid message column metadata/value");
  }
  Result result;result.layout=layout;std::vector<NodeBinding> binding;std::map<Index,LocalState> initial;
  for(const auto& name:{"candidate_events","selected_events","visited_edges","region_column_calls","state_column_calls"})result.stats[name]=0;
  require(left.empty()||left.size()==g.nodes.size(),"column left node domain differs");
  for(Index v=0;v<static_cast<Index>(g.nodes.size());++v){
    const auto& w=model.nodes.at(v);auto b=node_binding(w);
    std::visit([](const auto& c){c.validate();},b.memory);const auto r=g.nodes[v].region;
    require(!g.nodes[v].clear&&g.nodes[v].next_state=="adopt-v1"&&g.regions[r].observe_all&&w.bias.device()==ref.device()
      &&w.bias.scalar_type()==ref.scalar_type()&&w.bias.numel()==ref.numel(),"unsupported column Settle binding");
    initial[v]=initial_columns(w,b.memory,layout.samples());
    result.states[v]=left.empty()?initial[v]:left.at(v);validate_columns(w,b.memory,result.states[v]);
    require(std::visit([](const auto& s){return s.last_times.size();},result.states[v])==static_cast<size_t>(layout.samples()),"column left sample count differs");binding.push_back(b);
    auto bytes=workspace_bytes(b.memory,layout,result.states[v],ref.element_size());
    require(bytes<=max_workspace_bytes,"column local workspace exceeds budget");
  }
  std::vector<Index> order(g.regions.size());std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](Index a,Index b){return spec.ranks()[a]<spec.ranks()[b];});
  std::map<Index,Message> outgoing;
  for(auto region:order){
    std::map<Index,std::vector<Index>> candidates;std::vector<Index> nodes;
    for(Index k=g.region_index.offsets[region];k<g.region_index.offsets[region+1];++k){
      const auto v=g.region_index.edges[k];const auto& w=model.nodes[v];const auto& b=binding[v];
      std::vector<std::pair<const Message*,Index>> sources;
      for(Index p=0;p<static_cast<Index>(g.inputs.size());++p)if(g.inputs[p]==v&&ports.count(p))sources.emplace_back(&ports.at(p),b.sources.inputs[p]);
      for(Index j=g.csc.offsets[v];j<g.csc.offsets[v+1];++j){auto e=g.csc.edges[j];
        if(outgoing.count(g.edges[e].source))sources.emplace_back(&outgoing.at(g.edges[e].source),b.sources.edges[e]);}
      auto received=aggregate(w,b,sources);if(received.rows.empty())continue;
      auto times=layout.select(received.rows);for(auto& t:times.times)t=t*spec.stride()+spec.rank(v);
      auto window=prepare_columns(w,b.memory,times,received.mass,received.content,result.states[v]);
      auto fields=read_columns(w,b,window,g.regions[region].read_mode,initial[v]);
      result.states[v]=terminal_columns(window);
      result.nodes[v]={received.rows,std::move(window),std::move(fields),at::zeros_like(received.mass)};nodes.push_back(v);
      for(auto row:received.rows)candidates[row].push_back(v);
    }
    std::map<std::vector<Index>,std::vector<Index>> groups;
    for(const auto& [row,members]:candidates)groups[members].push_back(row);
    for(const auto& [members,rows]:groups){
      std::map<std::string,std::vector<Tensor>> fields;std::map<Index,Tensor> ids;
      for(auto v:members){const auto& node=result.nodes.at(v);ids[v]=indices(positions(node.rows,rows),ref.device());
        for(const auto& [name,t]:node.descriptors)fields[name].push_back(t.index_select(0,ids.at(v)));}
      TensorFields packed;for(const auto& [name,values]:fields)packed[name]=at::stack(values,1);
      auto n=indices(members,ref.device()).expand({static_cast<Index>(rows.size()),static_cast<Index>(members.size())});
      auto values=region_columns(model.regions[region],packed,region_binding(model.regions[region]),n).controls.unbind(1);
      for(size_t i=0;i<members.size();++i){auto v=members[i];result.nodes[v].controls=result.nodes[v].controls.index_copy(0,ids.at(v),values[i]);}
      ++result.stats["region_column_calls"];
    }
    for(auto v:nodes){const auto& node=result.nodes.at(v);
      auto value=full_columns(model.nodes[v],binding[v].memory,node.window,node.controls);
      for(auto& row:value.rows)row=node.rows[row];outgoing[v]=value;
      result.stats["candidate_events"]+=node.rows.size();result.stats["selected_events"]+=value.rows.size();
      ++result.stats["state_column_calls"];
    }
  }
  Message empty{{},at::empty({0},ref.options()),at::empty({0,ref.numel()-1},ref.options())};
  for(Index e=0;e<static_cast<Index>(g.edges.size());++e){const auto v=g.edges[e].source;
    result.messages[e]=outgoing.count(v)?outgoing.at(v):empty;result.stats["visited_edges"]+=result.messages[e].rows.size();}
  for(Index p=0;p<static_cast<Index>(g.outputs.size());++p){const auto v=g.outputs[p];result.outputs[p]=outgoing.count(v)?outgoing.at(v):empty;}
  std::vector<Tensor> published;
  for(const auto& [v,n]:result.nodes){for(const auto& [name,t]:n.descriptors)published.push_back(t);
    std::visit([&](const auto& window){for(const auto& [name,t]:window.bank)published.push_back(t);
      published.insert(published.end(),{window.mass,window.content});
      if constexpr(std::is_same_v<std::decay_t<decltype(window)>,RecurrentWindow>)if(window.cache_values.defined())published.push_back(window.cache_values);
    },n.window);published.push_back(n.controls);}
  for(const auto& [v,p]:outgoing)published.insert(published.end(),{p.strength,p.content});
  for(const auto& [v,s]:result.states)std::visit([&](const auto& state){for(const auto& [name,t]:state.fields)published.push_back(t);},s);
  require(all_finite(published),"nonfinite column result");
  return result;
}
Tensor readout(const Result& result,const Tensor& like){
  auto mass=at::zeros({static_cast<Index>(result.layout.times.size())},like.options());
  auto value=at::zeros({mass.size(0),like.size(-1)},like.options());
  for(const auto& [port,p]:result.outputs){auto rows=indices(p.rows,like.device()),strength=p.strength.clamp_min(0);
    mass=mass.index_add(0,rows,strength);value=value.index_add(0,rows,strength.unsqueeze(-1)*p.content);}
  return value/mass.clamp_min(1).unsqueeze(-1);
}
}
