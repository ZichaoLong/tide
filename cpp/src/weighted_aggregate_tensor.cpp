// Architecture §2: source-owner bank, integer incidence and centered VJP.
#include "weighted_aggregate_layout.h"
#include "tide/autograd_census.h"
#include "tide/packing_work.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <cstdint>
#include <optional>

namespace tide::weighted::columns {
namespace {
using torch::autograd::variable_list;
class AggregateVjp final:public torch::autograd::Node {
 public:
  AggregateSourceLayout layout;
  std::vector<torch::autograd::SavedVariable> originals,numeric;
  std::vector<std::pair<Index,bool>> outputs;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("aggregate tensor supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("aggregate tensor graph released");
    variable_list inputs,values;for(auto& t:originals)inputs.push_back(t.unpack());for(auto& t:numeric)values.push_back(t.unpack());
    const auto& p=layout;auto mass=values[0],value=values[1],total=values[2],mean=values[3],positive=values[4],nonnegative=values[5],incidence=values[6];
    auto gm=at::zeros(total.sizes(),total.options()),gu=at::zeros(mean.sizes(),mean.options());
    std::vector<bool> used_mass(p.groups,false),used_value(p.groups,false);
    for(size_t j=0;j<bars.size();++j)if(bars[j].defined()){const auto [g,is_value]=outputs[j];
      if(is_value){gu[g].index_copy_(0,p.publish[g],bars[j]);used_value[g]=true;}
      else {gm[g].index_copy_(0,p.publish[g],bars[j]);used_mass[g]=true;}}
    auto dm=at::matmul(incidence.t(),gm);Tensor dv;
    if(std::any_of(used_value.begin(),used_value.end(),[](bool x){return x;})){
      auto denominator=at::where(positive,total,at::ones_like(total));
      gu=at::where(positive.unsqueeze(-1),gu,at::zeros({},gu.options()));
      auto h=gu/denominator.unsqueeze(-1);
      dv=at::matmul(incidence.t(),h.flatten(1)).reshape_as(value)*mass.unsqueeze(-1);
      // One S x R x D receiver tile, no G-expanded payload. Center before
      // contracting to retain the stable mass derivative; no B x L loop.
      for(Index g=0;g<p.groups;++g)if(used_value[g]&&p.physical_counts[g]!=1){
        auto live=(incidence[g]>0).unsqueeze(-1)&(positive[g]&~p.singleton[g]).unsqueeze(0);
        auto delta=at::where(live.unsqueeze(-1),value-mean[g],at::zeros({},value.options()));
        dm=dm+incidence[g].unsqueeze(-1)*(delta*gu[g]).sum(-1)/denominator[g];
      }
    }
    auto masses=(dm*nonnegative).flatten().index_select(0,p.inverse).split_with_sizes(p.lengths);
    std::vector<Tensor> vectors;if(dv.defined())vectors=dv.flatten(0,1).index_select(0,p.inverse).split_with_sizes(p.lengths);
    variable_list result(inputs.size());
    for(Index s=0;s<p.sources;++s){bool m=false,v=false;
      for(Index g=0;g<p.groups;++g)if(p.incidence[g][s]){m|=used_mass[g]||used_value[g];v|=used_value[g];}
      if(m&&should_compute_output(2*s))result[2*s]=masses[s];
      if(v&&should_compute_output(2*s+1))result[2*s+1]=vectors[s];
    }
    return result;
  }
  void release_variables()override{originals.clear();numeric.clear();layout=AggregateSourceLayout{};outputs.clear();}
  std::string name()const override{return "tide::weighted::AggregateTensorVjp";}
};
std::vector<Received> evaluate(AggregateSourceLayout p){
  bool grad=false;if(at::GradMode::is_enabled())for(const auto& t:p.originals)grad|=t.requires_grad();
  at::NoGradGuard guard;p.materialize();variable_list masses,vectors;
  for(size_t j=0;j<p.originals.size();j+=2){masses.push_back(p.originals[j]);vectors.push_back(p.originals[j+1]);}
  masses.push_back(at::zeros({1},masses[0].options()));vectors.push_back(at::zeros({1,p.width},vectors[0].options()));
  auto raw=at::cat(masses).index_select(0,p.index.flatten()).reshape({p.sources,p.rows});
  auto mass=raw.clamp_min(0),value=at::cat(vectors).index_select(0,p.index.flatten()).reshape({p.sources,p.rows,p.width});
  auto incidence=p.counts.to(raw.scalar_type()),total=at::matmul(incidence,mass),positive=total>0;
  auto denominator=at::where(positive,total,at::ones_like(total));
  auto mean=at::matmul(incidence,(mass.unsqueeze(-1)*value).flatten(1)).reshape({p.groups,p.rows,p.width});
  mean=at::where(positive.unsqueeze(-1),mean/denominator.unsqueeze(-1),at::zeros({},mean.options()));
  std::shared_ptr<AggregateVjp> node;Tensor live;
  if(grad){node=std::make_shared<AggregateVjp>();node->set_next_edges(torch::autograd::collect_next_edges(p.originals));
    for(const auto& t:p.originals)node->originals.emplace_back(t,false);
    for(const auto& t:{mass,value,total,mean,positive,raw>=0,incidence})node->numeric.emplace_back(t,false);
    live=positive.any(-1).to(at::kCPU).contiguous();
    if(autograd_census::enabled()){autograd_census::saved(p.originals);
      autograd_census::saved({mass,value,total,mean,positive,p.index,p.inverse,p.counts,p.singleton,incidence});autograd_census::saved(p.publish);}}
  std::vector<Received> out;
  for(Index g=0;g<p.groups;++g){auto m=total[g],u=mean[g];
    if(p.publish[g].numel()!=p.rows){m=m.index_select(0,p.publish[g]);u=u.index_select(0,p.publish[g]);}
    bool dm=false,du=false;
    if(grad){for(Index s=0;s<p.sources;++s)if(p.incidence[g][s]){dm|=p.originals[2*s].requires_grad();
        du|=p.originals[2*s].requires_grad()||p.originals[2*s+1].requires_grad();}
      du=du&&live.const_data_ptr<bool>()[g];
      if(dm){node->outputs.emplace_back(g,false);torch::autograd::set_history(m,node);}
      if(du){node->outputs.emplace_back(g,true);torch::autograd::set_history(u,node);}}
    out.push_back({p.receiver_rows[g],m,u});}
  work::packing::bucket("aggregate_tensor",p.rows*p.sources,p.groups);
  work::packing::bucket("aggregate_source_bank",p.rows,p.sources);
  if(node){p.originals.clear();node->layout=std::move(p);}
  return out;
}
void validate(const AggregateTensorRequest& r){
  detail::require(r.sources.size()==r.values.size()&&!r.sources.empty(),"invalid aggregate tensor atom columns");
  Index last=-1;for(auto row:r.rows){detail::require(row>last,"aggregate tensor rows must be sorted unique nonnegative");last=row;}
  const auto& like=r.values[0];detail::require(like.dim()==2&&like.size(1)>0,"invalid aggregate tensor value shape");
  for(size_t i=0;i<r.sources.size();++i){const auto& source=*r.sources[i];const auto& z=r.values[i];const Index n=source.rows.size();
    detail::require(n>0&&source.strength.sizes()==at::IntArrayRef({n})&&z.sizes()==at::IntArrayRef({n,like.size(1)})
      &&z.device()==like.device()&&source.strength.device()==like.device()&&z.scalar_type()==like.scalar_type()
      &&source.strength.scalar_type()==like.scalar_type(),"aggregate tensor source metadata differs");
    Index previous=-1;for(auto row:source.rows){detail::require(row>previous&&std::binary_search(r.rows.begin(),r.rows.end(),row),"aggregate tensor row map invalid");previous=row;}
  }
}
struct Task {std::vector<size_t> ids;std::optional<AggregateSourceLayout> plan;};
std::uintptr_t identity(const Tensor& t){return reinterpret_cast<std::uintptr_t>(t.unsafeGetTensorImpl());}
}
std::vector<Received> aggregate_tensor_many(const std::vector<AggregateTensorRequest>& requests,Index budget,
    const std::function<Received(const AggregateTensorRequest&)>& fallback){
  detail::require(budget>0,"aggregate tensor workspace must be positive");std::vector<Received> result(requests.size());std::vector<Task> tasks;
  using Key=std::tuple<int,int,int,Index>;std::map<Key,std::vector<size_t>> groups;
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];if(r.rows.empty())continue;validate(r);const auto& like=r.values[0];
    if(like.scalar_type()!=at::kFloat&&like.scalar_type()!=at::kDouble){tasks.push_back({{i},std::nullopt});continue;}
    groups[{int(like.device().type()),like.device().index(),int(like.scalar_type()),like.size(1)}].push_back(i);}
  auto plan=[&](const std::vector<size_t>& ids){std::vector<const AggregateTensorRequest*> entries;
    for(auto i:ids)entries.push_back(&requests[i]);return aggregate_source_layout(entries);};
  for(const auto& [key,ids]:groups){std::vector<size_t> pending;std::optional<AggregateSourceLayout> current;
    auto flush=[&](){if(!pending.empty())tasks.push_back({std::move(pending),current->groups>1?std::move(current):std::nullopt});
      pending.clear();current.reset();};
    for(auto i:ids){auto proposed=pending;proposed.push_back(i);auto candidate=plan(proposed);
      if(!pending.empty()&&candidate.bytes()>budget){flush();proposed={i};candidate=plan(proposed);}
      if(candidate.bytes()>budget)tasks.push_back({{i},std::nullopt});
      else {pending=std::move(proposed);current=std::move(candidate);}}
    flush();
  }
  // Only real numerical budget boundaries publish ordered use roots. Within
  // each incidence block one unique source root receives every physical use.
  std::map<std::uintptr_t,std::pair<Tensor,std::vector<size_t>>> roots;
  std::vector<std::map<std::uintptr_t,Tensor>> mappings(tasks.size());
  for(size_t t=0;t<tasks.size();++t)for(auto i:tasks[t].ids){const auto& r=requests[i];
    for(size_t j=0;j<r.sources.size();++j)for(const auto& value:{r.sources[j]->strength,r.values[j]}){
      auto [it,inserted]=roots.emplace(identity(value),std::make_pair(value,std::vector<size_t>{}));auto& users=it->second.second;
      if(users.empty()||users.back()!=t)users.push_back(t);}}
  for(const auto& [key,root]:roots){auto uses=ordered_uses(root.first,root.second.size());
    for(size_t j=0;j<uses.size();++j)mappings[root.second[j]][key]=uses[j];}
  for(size_t t=0;t<tasks.size();++t){auto& task=tasks[t];const auto& mapped=mappings[t];
    if(task.plan){for(auto& value:task.plan->originals)value=mapped.at(identity(value));
      auto out=evaluate(std::move(*task.plan));for(size_t g=0;g<out.size();++g)result[task.ids[g]]=std::move(out[g]);}
    else {const auto i=task.ids[0];const auto& r=requests[i];auto q=r;std::vector<Message> borrowed;borrowed.reserve(r.sources.size());
      q.sources.clear();q.values.clear();
      for(size_t j=0;j<r.sources.size();++j){borrowed.push_back(*r.sources[j]);borrowed.back().strength=mapped.at(identity(r.sources[j]->strength));
        q.sources.push_back(&borrowed.back());q.values.push_back(mapped.at(identity(r.values[j])));}
      result[i]=fallback(q);}
  }
  return result;
}
}
