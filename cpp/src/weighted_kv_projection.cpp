#include "weighted_kv_projection.h"
namespace tide::weighted::packed {
std::pair<Tensor,Tensor> kv_projections(const NodeWeights& w,const Writes& p,const PackedSequence& batch,const KVConfig& c){
  std::vector<Index> shared,independent;
  std::vector<Tensor> values(p.ids.size()),summaries(p.ids.size());
  for(size_t k=0;k<p.ids.size();++k){
    auto j=p.ids[k];
    if(!batch.prepared.empty()&&batch.prepared[j].size()==2&&batch.prepared[j].count("values")&&batch.prepared[j].count("summaries")){
      const auto& row=batch.prepared[j];auto v=row.at("values"),s=row.at("summaries");
      detail::require(v.sizes()==at::IntArrayRef({2*c.width})&&s.sizes()==at::IntArrayRef({c.summary_width}),
                      "invalid KV private preparation shape");
      for(const auto& t:{v,s})detail::require(t.device()==p.input.device()&&t.scalar_type()==p.input.scalar_type(),
                                             "invalid KV private preparation dtype/device");
      values[k]=v;summaries[k]=s;continue;
    }
    (batch.prepared.empty()||batch.prepared[j].empty()?independent:shared).push_back(k);
  }
  const auto& left=detail::weight(w,"kernel.adapter_u");const auto& right=detail::weight(w,"kernel.adapter_v");
  if(independent.size()==p.ids.size()){
    auto x=normalize(p.input,c.epsilon);
    return {project(x,detail::weight(w,"kernel.base"))+project(project(x,right),left),project(x,detail::weight(w,"kernel.summary"))};
  }
  if(!independent.empty()){
    auto ids=at::tensor(independent,p.input.options().dtype(at::kLong));
    auto input=p.input.index_select(0,ids),x=normalize(input,c.epsilon);
    auto v=project(x,detail::weight(w,"kernel.base"))+project(project(x,right),left);
    auto s=project(x,detail::weight(w,"kernel.summary"));
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
    auto x=at::stack(xs),v=at::stack(bases)+project(project(x,right),left);
    auto s=project(x,detail::weight(w,"kernel.summary"));
    for(size_t i=0;i<shared.size();++i){values[shared[i]]=v[i];summaries[shared[i]]=s[i];}
  }
  return {at::stack(values),at::stack(summaries)};
}
}
