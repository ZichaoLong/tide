// Settle §9–11: independent region-order host schedule with actual column support.
#include "weighted_column_modules.h"
#include "weighted_column_placement.h"
#include "weighted_placement.h"
#include "weighted_mean.h"
#include "weighted_source.h"
#include "tide/ownership.h"
#include "finite_guard.h"
#include <set>
#include <numeric>
namespace tide::weighted::columns {
using detail::require;
namespace {
struct Received {std::vector<Index> rows;Tensor mass,content;};
using CopyKey=std::tuple<const Message*,int,int,int>;
using Copies=std::map<CopyKey,Message>;
const Message& transport(const Message& p,const Tensor& like,Copies& cache){
  if(p.content.device()==like.device()&&p.content.scalar_type()==like.scalar_type())return p;
  CopyKey key{&p,int(like.device().type()),like.device().index(),int(like.scalar_type())};
  if(auto found=cache.find(key);found!=cache.end())return found->second;
  auto copied=p;placement_tensors({&copied.strength,&copied.content},like.device(),like.scalar_type());
  return cache.emplace(key,std::move(copied)).first->second;
}
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
    if(p->rows==out.rows){
      masses.push_back(p->strength.clamp_min(0));values.push_back(z);
    }else{
      auto ids=indices(positions(out.rows,p->rows),z.device());
      masses.push_back(at::zeros({n},z.options()).index_copy(0,ids,p->strength.clamp_min(0)));
      values.push_back(at::zeros({n,width},z.options()).index_copy(0,ids,z));
    }
  }
  if(out.rows.empty())return out;
  auto mass=at::stack(masses,1),value=at::stack(values,1);out.mass=mass.sum(1);
  auto positive=(out.mass.detach()>0).nonzero().flatten();out.content=at::zeros({n,width},value.options());
  if(positive.numel()==n)out.content=weighted_mean(mass,value);
  else if(positive.numel())out.content=out.content.index_copy(0,positive,weighted_mean(mass.index_select(0,positive),value.index_select(0,positive)));
  return out;
}
}
Result settle(const SettleGraph& spec,const Model& model,const Layout& layout,const std::map<Index,Message>& ports,
              const std::map<Index,LocalState>& left,Index max_workspace_bytes,std::optional<ExecutionPlacement> requested){
  layout.validate();const auto& g=spec.graph();const auto& ref=model.nodes.at(0).bias;
  require(spec.binding()=="ports","column Settle requires ports binding");validate_transport(g,model);
  require(max_workspace_bytes>0,"column workspace budget must be positive");
  require(ref.scalar_type()==at::kHalf||ref.scalar_type()==at::kFloat||ref.scalar_type()==at::kDouble,"column Settle requires FP16/FP32/FP64");
  ExecutionPlacement defaults;defaults.selection="payload";
  const auto request=requested.value_or(defaults);
  std::vector<ResolvedPlacement> node_locations,region_locations;
  for(const auto& w:model.nodes){auto p=resolve_placement(request,w.bias.device());
    require(p.events.is_cpu(),"column Settle is host scheduled");node_locations.push_back(p);}
  for(Index r=0;r<static_cast<Index>(g.regions.size());++r)
    region_locations.push_back(resolve_placement(request,region_reference(g,model,r).device()));
  const Index maxrank=*std::max_element(spec.ranks().begin(),spec.ranks().end());
  for(auto t:layout.times)require(t<=(std::numeric_limits<Index>::max()-maxrank)/spec.stride(),"column logical time overflow");
  for(const auto& [port,p]:ports){
    require(port>=0&&port<static_cast<Index>(g.inputs.size()),"unknown column input port");layout.select(p.rows);
    require(p.content.sizes()==at::IntArrayRef({static_cast<Index>(p.rows.size()),ref.numel()-1})
      &&p.strength.sizes()==at::IntArrayRef({static_cast<Index>(p.rows.size())}),"message column shapes differ");
    for(const auto& x:{p.strength,p.content})require(x.device()==p.content.device()&&x.scalar_type()==ref.scalar_type()
      &&at::isfinite(x).all().item<bool>(),"invalid message column metadata/value");
  }
  Result result;result.layout=layout;std::vector<NodeBinding> binding;std::map<Index,LocalState> initial;
  for(const auto& name:{"candidate_events","selected_events","visited_edges","region_column_calls","state_column_calls"})result.stats[name]=0;
  require(left.empty()||left.size()==g.nodes.size(),"column left node domain differs");
  for(Index v=0;v<static_cast<Index>(g.nodes.size());++v){
    const auto& w=model.nodes.at(v);auto b=node_binding(w);
    std::visit([](const auto& c){c.validate();},b.memory);const auto r=g.nodes[v].region;
    require(!g.nodes[v].clear&&g.nodes[v].next_state=="adopt-v1"&&g.regions[r].observe_all
      &&w.bias.scalar_type()==ref.scalar_type()&&w.bias.numel()==ref.numel(),"unsupported column Settle binding");
    initial[v]=initial_columns(w,b.memory,layout.samples());
    result.states[v]=left.empty()?initial[v]:left.at(v);validate_columns(w,b.memory,result.states[v]);
    require(std::visit([](const auto& s){return s.last_times.size();},result.states[v])==static_cast<size_t>(layout.samples()),"column left sample count differs");binding.push_back(b);
    auto bytes=workspace_bytes(b.memory,layout,result.states[v],std::max<int64_t>(w.bias.element_size(),c10::elementSize(read_dtype(node_locations[v],w.bias.scalar_type()))));
    require(bytes<=max_workspace_bytes,"column local workspace exceeds budget");
  }
  std::vector<Index> order(g.regions.size());std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](Index a,Index b){return spec.ranks()[a]<spec.ranks()[b];});
  std::map<Index,Message> outgoing,empty;Copies copies;
  for(Index v=0;v<static_cast<Index>(g.nodes.size());++v){const auto& w=model.nodes[v];
    empty[v]={{},at::empty({0},w.bias.options()),at::empty({0,w.bias.numel()-1},w.bias.options())};}
  for(auto region:order){
    const auto& placement=region_locations[region];
    std::map<Index,std::vector<Index>> candidates;std::vector<Index> nodes;
    for(Index k=g.region_index.offsets[region];k<g.region_index.offsets[region+1];++k){
      const auto v=g.region_index.edges[k];const auto& w=model.nodes[v];const auto& b=binding[v];
      std::vector<std::pair<const Message*,Index>> sources;
      for(Index p=0;p<static_cast<Index>(g.inputs.size());++p)if(g.inputs[p]==v&&ports.count(p))sources.emplace_back(&transport(ports.at(p),w.bias,copies),b.sources.inputs[p]);
      for(Index j=g.csc.offsets[v];j<g.csc.offsets[v+1];++j){auto e=g.csc.edges[j];
        if(outgoing.count(g.edges[e].source))sources.emplace_back(&transport(outgoing.at(g.edges[e].source),w.bias,copies),b.sources.edges[e]);}
      auto received=aggregate(w,b,sources);if(received.rows.empty())continue;
      auto times=layout.select(received.rows);for(auto& t:times.times)t=t*spec.stride()+spec.rank(v);
      auto window=prepare_columns(w,b.memory,times,received.mass,received.content,result.states[v]);
      auto fields=read_placed(w,b,window,g.regions[region].read_mode,initial[v],node_locations[v]);
      result.states[v]=terminal_columns(window);
      result.nodes[v]={received.rows,std::move(window),std::move(fields),at::zeros_like(received.mass)};nodes.push_back(v);
      for(auto row:received.rows)candidates[row].push_back(v);
    }
    std::map<std::vector<Index>,std::vector<Index>> groups;
    for(const auto& [row,members]:candidates)groups[members].push_back(row);
    std::map<Index,TensorFields> regional_fields;
    for(auto v:nodes){auto fields=result.nodes.at(v).descriptors;std::vector<Tensor*> handles;
      for(auto& [name,t]:fields)handles.push_back(&t);
      placement_tensors(handles,placement.control,fields.at("w").scalar_type());regional_fields.emplace(v,std::move(fields));}
    for(const auto& [members,rows]:groups){
      std::map<std::string,std::vector<Tensor>> fields;std::map<Index,Tensor> ids;
      for(auto v:members){const auto& node=result.nodes.at(v);
        if(node.rows!=rows)ids[v]=indices(positions(node.rows,rows),regional_fields.at(v).at("w").device());
        for(const auto& [name,t]:regional_fields.at(v))fields[name].push_back(ids.count(v)?t.index_select(0,ids.at(v)):t);}
      TensorFields packed;for(const auto& [name,values]:fields)packed[name]=at::stack(values,1);
      auto n=indices(members,packed.at("w").device()).expand({static_cast<Index>(rows.size()),static_cast<Index>(members.size())});
      auto values=region_placed(model.regions[region],packed,region_binding(model.regions[region]),n,placement).controls.unbind(1);
      for(size_t i=0;i<members.size();++i){auto v=members[i];const auto& like=model.nodes[v].bias;
        auto value=control_payload(values[i],like);
        result.nodes[v].controls=ids.count(v)?result.nodes[v].controls.index_copy(0,ids.at(v).to(like.device()),value):value;}
      ++result.stats["region_column_calls"];
    }
    for(auto v:nodes){const auto& node=result.nodes.at(v);
      auto value=full_columns(model.nodes[v],binding[v].memory,node.window,node.controls);
      for(auto& row:value.rows)row=node.rows[row];outgoing[v]=value;
      result.stats["candidate_events"]+=node.rows.size();result.stats["selected_events"]+=value.rows.size();
      ++result.stats["state_column_calls"];
    }
  }
  for(Index e=0;e<static_cast<Index>(g.edges.size());++e){const auto v=g.edges[e].source;
    const auto target=g.edges[e].target;result.messages[e]=transport(outgoing.count(v)?outgoing.at(v):empty.at(target),model.nodes[target].bias,copies);
    result.stats["visited_edges"]+=result.messages[e].rows.size();}
  for(Index p=0;p<static_cast<Index>(g.outputs.size());++p){const auto v=g.outputs[p];result.outputs[p]=outgoing.count(v)?outgoing.at(v):empty.at(v);}
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
  Copies copies;
  for(const auto& [port,original]:result.outputs){const auto& p=transport(original,like,copies);auto rows=indices(p.rows,like.device()),strength=p.strength.clamp_min(0);
    mass=mass.index_add(0,rows,strength);value=value.index_add(0,rows,strength.unsqueeze(-1)*p.content);}
  return value/mass.clamp_min(1).unsqueeze(-1);
}
}
