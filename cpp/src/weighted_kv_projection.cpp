#include "weighted_kv_projection.h"
namespace tide::weighted::packed {
std::pair<Tensor,Tensor> kv_projections(const NodeWeights& w,const Writes& p,const PackedSequence& batch,const KVConfig& c){
  std::vector<Index> shared,independent;
  for(size_t k=0;k<p.ids.size();++k){
    auto j=p.ids[k];
    (batch.prepared.empty()||batch.prepared[j].empty()?independent:shared).push_back(k);
  }
  std::vector<Tensor> values(p.ids.size()),summaries(p.ids.size());
  auto adapter=at::matmul(detail::weight(w,"kernel.adapter_u"),detail::weight(w,"kernel.adapter_v"));
  if(shared.empty())return {project(normalize(p.input,c.epsilon),detail::weight(w,"kernel.base")+adapter),
      project(normalize(p.input,c.epsilon),detail::weight(w,"kernel.summary"))};
  if(!independent.empty()){
    auto ids=at::tensor(independent,p.input.options().dtype(at::kLong));
    auto input=p.input.index_select(0,ids),x=normalize(input,c.epsilon);
    auto v=project(x,detail::weight(w,"kernel.base")+adapter);
    auto s=project(normalize(input,c.epsilon),detail::weight(w,"kernel.summary"));
    for(size_t i=0;i<independent.size();++i){values[independent[i]]=v[i];summaries[independent[i]]=s[i];}
  }
  if(!shared.empty()){
    std::vector<Tensor> xs,bases;
    for(auto k:shared){
      const auto& row=batch.prepared[p.ids[k]];
      detail::require(row.size()==2&&row.count("normal")&&row.count("base"),"invalid KV preparation fields");
      detail::require(row.at("normal").sizes()==at::IntArrayRef({c.width})&&
          row.at("base").sizes()==at::IntArrayRef({2*c.width}),"invalid KV preparation shape");
      for(const auto& [name,t]:row)detail::require(t.scalar_type()==p.input.scalar_type()&&t.device()==p.input.device(),
          "invalid KV preparation dtype/device");
      xs.push_back(row.at("normal"));bases.push_back(row.at("base"));
    }
    auto x=at::stack(xs),v=at::stack(bases)+project(x,adapter);
    auto s=project(x,detail::weight(w,"kernel.summary"));
    for(size_t i=0;i<shared.size();++i){values[shared[i]]=v[i];summaries[shared[i]]=s[i];}
  }
  return {at::stack(values),at::stack(summaries)};
}
}
