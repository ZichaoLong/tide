// Regional-call-local source identities and bounded compatible operator groups.
#include "weighted_column_sources.h"
#include "tide/packing_work.h"
#include <cstdint>
namespace tide::weighted::columns {
std::vector<Tensor> source_columns_group(const std::vector<SourceRequest>&);
namespace {
using Key=std::tuple<double,double,std::vector<std::uintptr_t>>;
Key key(const SourceRequest& r){
  std::vector<std::uintptr_t> ids{reinterpret_cast<std::uintptr_t>(r.content.unsafeGetTensorImpl())};
  for(const auto& t:r.parameters)ids.push_back(reinterpret_cast<std::uintptr_t>(t.unsafeGetTensorImpl()));
  return {r.alpha,r.epsilon,ids};
}
bool compatible(const SourceRequest& a,const SourceRequest& b){
  if(a.alpha!=b.alpha||a.epsilon!=b.epsilon||a.content.sizes()!=b.content.sizes()
      ||a.content.device()!=b.content.device()||a.content.scalar_type()!=b.content.scalar_type()
      ||a.parameters.size()!=b.parameters.size())return false;
  for(size_t j=0;j<a.parameters.size();++j)if(a.parameters[j].sizes()!=b.parameters[j].sizes())return false;
  return true;
}
long double bytes(const SourceRequest& r){
  long double size=16.L*r.content.size(0)*(r.content.size(1)+r.parameters[0].size(0)+2);
  for(const auto& t:r.parameters)size+=12.L*t.numel();
  return size*std::max<int64_t>(4,r.content.element_size());
}
void validate(const SourceRequest& r){
  detail::require(r.content.dim()==2&&r.content.size(0)>0&&(r.parameters.size()==3||r.parameters.size()==7),"invalid source column shape/slots");
  const auto d=r.content.size(1),h=r.parameters[0].size(0);
  std::vector<std::vector<int64_t>> shapes{{h,d},{d,h},{h}};
  if(r.parameters.size()==7){const auto rank=r.parameters[3].size(-1);shapes.insert(shapes.end(),{{h,rank},{rank,d},{d,rank},{rank,h}});}
  for(size_t j=0;j<shapes.size();++j)detail::require(r.parameters[j].sizes()==at::IntArrayRef(shapes[j])
    &&r.parameters[j].device()==r.content.device()&&r.parameters[j].scalar_type()==r.content.scalar_type(),"source column parameter metadata mismatch");
}
Tensor fallback(const SourceRequest& r){
  auto a=r.parameters[0],b=r.parameters[1];if(r.parameters.size()==7){a=a+at::matmul(r.parameters[3],r.parameters[4]);b=b+at::matmul(r.parameters[5],r.parameters[6]);}
  work::packing::bucket("source_columns_single_fallback",r.content.size(0),1);
  return r.content+r.alpha*project(at::silu(project(normalize(r.content,r.epsilon),a)+r.parameters[2]),b);
}
}
std::vector<Tensor> source_parameters(const NodeWeights& w,Index label){
  auto prefix="aggregate_program.transforms."+std::to_string(label)+".";std::vector<Tensor> result;
  for(const auto* name:{"a","b","embedding"})result.push_back(detail::weight(w,prefix+name));
  if(w.extra.count(prefix+"a_left"))for(const auto* name:{"a_left","a_right","b_left","b_right"})result.push_back(detail::weight(w,prefix+name));
  return result;
}
std::vector<Tensor> source_columns(const std::vector<SourceRequest>& requests,Index budget){
  detail::require(budget>0,"source column workspace must be positive");
  std::map<Key,size_t> lookup;std::vector<SourceRequest> unique;std::vector<std::vector<size_t>> uses,groups;
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];
    detail::require(r.content.dim()==2&&r.content.size(0)>0&&r.content.size(1)>0
      &&std::isfinite(r.alpha)&&std::isfinite(r.epsilon)&&r.epsilon>0,"invalid source column content/scalars");
    auto k=key(r);auto [found,inserted]=lookup.emplace(k,unique.size());
    if(inserted){unique.push_back(requests[i]);uses.emplace_back();}uses[found->second].push_back(i);}
  std::vector<Tensor> computed(unique.size()),result(requests.size());
  for(size_t i=0;i<unique.size();++i){const auto& r=unique[i];if(!r.alpha){computed[i]=r.content;continue;}validate(r);
    bool found=false;for(auto& g:groups)if(compatible(unique[g.front()],r)){g.push_back(i);found=true;break;}
    if(!found)groups.push_back({i});}
  for(const auto& group:groups){std::vector<size_t> pending;long double used=0;
    auto flush=[&](){if(pending.empty())return;std::vector<SourceRequest> rs;for(auto i:pending)rs.push_back(unique[i]);
      auto values=source_columns_group(rs);for(size_t j=0;j<pending.size();++j)computed[pending[j]]=values[j];pending.clear();used=0;};
    for(auto i:group){auto size=bytes(unique[i]);if(!pending.empty()&&used+size>budget)flush();
      if(size>budget)computed[i]=fallback(unique[i]);else{pending.push_back(i);used+=size;}}
    flush();}
  for(size_t i=0;i<unique.size();++i){auto values=ordered_uses(computed[i],uses[i].size());
    if(uses[i].size()>1)work::packing::bucket("source_exact_reuse",computed[i].size(0),uses[i].size());
    for(size_t j=0;j<uses[i].size();++j)result[uses[i][j]]=values[j];}
  return result;
}
}
