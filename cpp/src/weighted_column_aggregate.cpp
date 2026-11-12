// Architecture §2: exact regional caches preserve physical atom multiplicity.
#include "weighted_column_sources.h"
#include "weighted_mean.h"
#include "weighted_aggregate_tensor.h"
#include "tide/packing_work.h"
#include <set>
#include <cstdint>
namespace tide::weighted::columns {
namespace {
using PayloadKey=std::tuple<std::vector<Index>,std::uintptr_t,std::uintptr_t,std::vector<std::uintptr_t>>;
using Key=std::tuple<double,double,std::vector<PayloadKey>>;
Key key(const AggregateRequest& r){
  std::vector<PayloadKey> ids;
  for(const auto& [p,label]:r.sources){if(p->rows.empty())continue;std::vector<std::uintptr_t> parameters;
    if(r.binding->aggregate_alpha)for(const auto& t:source_parameters(*r.weights,label))parameters.push_back(reinterpret_cast<std::uintptr_t>(t.unsafeGetTensorImpl()));
    ids.emplace_back(p->rows,reinterpret_cast<std::uintptr_t>(p->strength.unsafeGetTensorImpl()),
      reinterpret_cast<std::uintptr_t>(p->content.unsafeGetTensorImpl()),parameters);
  }
  return {r.binding->aggregate_alpha,std::visit([](const auto& c){return c.epsilon;},r.binding->memory),ids};
}
Received assemble(const AggregateTensorRequest& r){
  Received out;out.rows=r.rows;if(out.rows.empty())return out;
  const auto n=static_cast<Index>(out.rows.size());std::vector<Tensor> masses,values;
  for(size_t i=0;i<r.sources.size();++i){const auto* p=r.sources[i];auto z=r.values[i];
    if(p->rows==out.rows){masses.push_back(p->strength.clamp_min(0));values.push_back(z);}
    else{std::vector<Index> positions;for(auto row:p->rows)positions.push_back(std::lower_bound(out.rows.begin(),out.rows.end(),row)-out.rows.begin());
      auto ids=indices(positions,z.device());masses.push_back(at::zeros({n},z.options()).index_copy(0,ids,p->strength.clamp_min(0)));
      values.push_back(at::zeros({n,z.size(1)},z.options()).index_copy(0,ids,z));}
  }
  auto mass=at::stack(masses,1),value=at::stack(values,1);out.mass=mass.sum(1);
  auto positive=(out.mass.detach()>0).nonzero().flatten();out.content=at::zeros({n,value.size(-1)},value.options());
  if(positive.numel()==n)out.content=weighted_mean(mass,value);
  else if(positive.numel())out.content=out.content.index_copy(0,positive,weighted_mean(mass.index_select(0,positive),value.index_select(0,positive)));
  return out;
}
}
std::vector<Received> aggregate_columns_many(const std::vector<AggregateRequest>& requests,Index budget){
  std::map<Key,size_t> lookup;std::vector<size_t> unique;std::vector<std::vector<size_t>> uses,slots;
  std::vector<SourceRequest> transforms;
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];auto k=key(r);
    auto [found,inserted]=lookup.emplace(k,unique.size());
    if(inserted){unique.push_back(i);uses.emplace_back();slots.emplace_back();
      for(const auto& [p,label]:r.sources)if(!p->rows.empty()){
        auto parameters=r.binding->aggregate_alpha?source_parameters(*r.weights,label):std::vector<Tensor>{};
        slots.back().push_back(transforms.size());transforms.push_back({p->content,parameters,r.binding->aggregate_alpha,std::get<1>(k)});}}
    uses[found->second].push_back(i);
  }
  auto transformed=source_columns(transforms,budget);std::vector<Received> result(requests.size());std::vector<AggregateTensorRequest> entries;
  for(size_t j=0;j<unique.size();++j){AggregateTensorRequest entry;std::set<Index> present;size_t k=0;
    for(const auto& [p,label]:requests[unique[j]].sources)if(!p->rows.empty()){
      present.insert(p->rows.begin(),p->rows.end());entry.sources.push_back(p);entry.values.push_back(transformed[slots[j][k++]]);}
    entry.rows={present.begin(),present.end()};entries.push_back(std::move(entry));}
  auto assembled=aggregate_tensor_many(entries,budget,assemble);
  for(size_t j=0;j<unique.size();++j){auto value=std::move(assembled[j]);
    if(value.rows.empty())continue;
    auto masses=ordered_uses(value.mass,uses[j].size()),contents=ordered_uses(value.content,uses[j].size());
    if(uses[j].size()>1)work::packing::bucket("aggregate_exact_reuse",1,uses[j].size());
    for(size_t i=0;i<uses[j].size();++i)result[uses[j][i]]={value.rows,masses[i],contents[i]};
  }
  return result;
}
}
