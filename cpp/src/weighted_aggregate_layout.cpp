// Payload storage is source x row x field; receivers contain only incidence.
#include "weighted_aggregate_layout.h"
#include <cstdint>
#include <set>
namespace tide::weighted::columns {
AggregateSourceLayout aggregate_source_layout(const std::vector<const AggregateTensorRequest*>& requests){
  AggregateSourceLayout p;p.groups=requests.size();p.width=requests[0]->values[0].size(1);
  using Key=std::tuple<std::vector<Index>,std::uintptr_t,std::uintptr_t>;
  std::map<Key,Index> keys;std::set<Index> rows;std::vector<std::map<Index,Index>> incidence;
  for(const auto* r:requests){p.receiver_rows.push_back(r->rows);rows.insert(r->rows.begin(),r->rows.end());
    p.physical_counts.push_back(r->sources.size());std::map<Index,Index> counts;
    for(size_t j=0;j<r->sources.size();++j){const auto* source=r->sources[j];const auto& value=r->values[j];
      Key key{source->rows,reinterpret_cast<std::uintptr_t>(source->strength.unsafeGetTensorImpl()),
        reinterpret_cast<std::uintptr_t>(value.unsafeGetTensorImpl())};
      auto [slot,inserted]=keys.emplace(key,p.source_rows.size());
      if(inserted){p.source_rows.push_back(source->rows);p.lengths.push_back(source->rows.size());
        p.originals.insert(p.originals.end(),{source->strength,value});}
      ++counts[slot->second];
    }
    incidence.push_back(std::move(counts));
  }
  p.global_rows={rows.begin(),rows.end()};p.rows=p.global_rows.size();p.sources=p.source_rows.size();
  for(const auto& group:incidence){std::vector<Index> counts(p.sources,0);for(auto [s,count]:group)counts[s]=count;p.incidence.push_back(std::move(counts));}
  return p;
}
long double AggregateSourceLayout::bytes()const{
  const long double s=sources,r=rows,g=groups,d=width,size=originals[1].element_size();
  // Includes one centered receiver scratch tile, both directional fields and
  // metadata; no G x S x R x D saved or temporary tensor.
  return s*r*((18*d+18)*size+24)+g*r*(6*d+8)*size+g*s*(8+size);
}
void AggregateSourceLayout::materialize(){
  Index total=0;for(auto n:lengths)total+=n;
  std::map<Index,Index> row_index;for(Index j=0;j<rows;++j)row_index[global_rows[j]]=j;
  std::vector<Index> ids(sources*rows,total),reverse,counts_flat,present(groups*rows,0);Index offset=0;
  for(Index s=0;s<sources;++s){for(size_t j=0;j<source_rows[s].size();++j){
    const auto row=row_index.at(source_rows[s][j]),slot=s*rows+row;ids[slot]=offset+j;reverse.push_back(slot);
    for(Index g=0;g<groups;++g)if(incidence[g][s])++present[g*rows+row];}
    offset+=lengths[s];}
  const auto device=originals[0].device();index=indices(ids,device).reshape({sources,rows});inverse=indices(reverse,device);
  for(const auto& group:incidence)counts_flat.insert(counts_flat.end(),group.begin(),group.end());
  counts=indices(counts_flat,device).reshape({groups,sources});
  for(auto& count:present)count=count==1;
  singleton=indices(present,device).to(at::kBool).reshape({groups,rows});
  for(const auto& domain:receiver_rows){std::vector<Index> positions;for(auto row:domain)positions.push_back(row_index.at(row));publish.push_back(indices(positions,device));}
}
}
