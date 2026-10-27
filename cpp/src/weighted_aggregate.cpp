// Architecture §2: parameter-class projections, distinct atoms, complete fibers.
#include "weighted_aggregate.h"
#include "weighted_packed.h"
#include "weighted_mean.h"

namespace tide::weighted {
std::vector<AggregateResult> aggregate_batch(const NodeWeights& w,const std::vector<AggregateInput>& requests,
    const SourceClasses& classes,double alpha,double epsilon){
  if(requests.empty())return {};
  std::vector<Tensor> atoms;std::vector<Index> offsets{0};
  std::map<Index,std::vector<Index>> labels,sizes;
  for(size_t i=0;i<requests.size();++i){
    detail::require(!requests[i].sources.empty(),"weighted Aggregate requires nonempty fibers");
    sizes[requests[i].sources.size()].push_back(i);
    for(const auto& source:requests[i].sources){
      if(alpha)labels[classes.classify(source.atom)].push_back(atoms.size());
      atoms.push_back(source.atom.value);
    }
    offsets.push_back(atoms.size());
  }
  auto payload=at::stack(atoms),mass=payload.select(1,0).clamp_min(0),original=payload.slice(1,1);
  auto content=original;auto ints=payload.options().dtype(at::kLong);
  for(const auto& [label,members]:labels){
    auto ids=at::tensor(members,ints),x=original.index_select(0,ids);
    auto prefix="aggregate_program.transforms."+std::to_string(label);
    auto hidden=at::silu(packed::project(normalize(x,epsilon),detail::weight(w,prefix+".a"))+detail::weight(w,prefix+".embedding"));
    auto values=x+alpha*packed::project(hidden,detail::weight(w,prefix+".b"));
    content=content.index_copy(0,ids,values);
  }
  std::vector<AggregateResult> result(requests.size());
  for(const auto& [length,owners]:sizes){
    std::vector<Index> flat;
    for(auto owner:owners)for(Index j=0;j<length;++j)flat.push_back(offsets[owner]+j);
    auto ids=at::tensor(flat,ints).reshape({static_cast<Index>(owners.size()),length});
    auto total=mass.index({ids}).sum(1),positive=total>0;
    auto values=weighted_mean(mass.index({ids}),content.index({ids}));
    values=at::where(positive.unsqueeze(-1),values,at::zeros_like(values));
    auto encoded=at::cat({total.unsqueeze(-1),values},-1);
    for(size_t row=0;row<owners.size();++row)result[owners[row]]={encoded[row],{}};
  }
  return result;
}
}
