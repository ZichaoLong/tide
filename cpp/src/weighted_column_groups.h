#pragma once
#include "weighted_column_queries.h"
#include "weighted_column_query_ops.h"
namespace tide::weighted::columns::groups {
inline std::pair<Index,Index> shape(const Layout& layout){
  Index width=1;for(Index b=0;b<layout.samples();++b)width=std::max(width,layout.offsets[b+1]-layout.offsets[b]);
  return {layout.samples(),width};
}
inline long double bytes(const NodeWeights& w,const KVConfig& c,const Layout& query,const Layout& bank,Index size){
  auto [b,q]=shape(query);const auto k=shape(bank).second;
  long double matrices=0;for(const auto& [name,t]:w.extra)matrices+=t.numel();
  long double pair=32+8*c.heads;bool positional=!c.read_frequencies.empty();
  for(const auto& f:c.full_frequencies)positional|=!f.empty();
  if(positional)pair+=8*(c.width+c.summary_width);
  return (4*matrices+static_cast<long double>(b)*q*(32*c.width+8*c.ffn_width+16*c.summary_width+128)
    +static_cast<long double>(b)*k*(20*c.width+8*c.summary_width)+static_cast<long double>(b)*q*k*pair)*std::max<Index>(size,4)
    +32*static_cast<long double>(b)*(q+k+static_cast<long double>(q)*k);
}
template<class Equal,class Cost> std::vector<std::vector<size_t>> plan(const std::vector<size_t>& ids,Index budget,Equal equal,Cost cost){
  detail::require(budget>0,"KV query group budget must be a positive int64 byte count");
  std::vector<std::vector<size_t>> compatible,result;
  for(auto i:ids){bool found=false;for(auto& group:compatible)if(equal(group.front(),i)){group.push_back(i);found=true;break;}
    if(!found)compatible.push_back({i});}
  for(const auto& group:compatible){std::vector<size_t> next;long double used=0;
    for(auto i:group){auto value=cost(i);if(!next.empty()&&used+value>budget){result.push_back(next);next.clear();used=0;}
      next.push_back(i);used+=value;}
    if(!next.empty())result.push_back(next);}
  return result;
}
inline bool same_payload(const Tensor& a,const Tensor& b){return a.scalar_type()==b.scalar_type()&&a.device()==b.device();}
inline std::vector<size_t> active(const std::vector<QueryInputs>& qs){
  std::vector<Tensor> flags;for(const auto& q:qs)flags.push_back((q.weights.detach()>0).any());
  auto host=at::stack(flags).to(at::kCPU);std::vector<size_t> ids;
  for(size_t i=0;i<qs.size();++i)if(host[i].item<bool>())ids.push_back(i);return ids;
}
}
