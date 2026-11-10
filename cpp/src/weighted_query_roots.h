#pragma once
// Compose only a complete disjoint contiguous partition's linear transpose.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include "tide/packing_work.h"
#include <algorithm>
#include <map>
#include <set>

namespace tide::weighted::query {
inline void coalesce_roots(std::vector<at::Tensor>& roots,std::vector<at::Tensor>& bars){
  std::map<const void*,std::vector<size_t>> groups;
  for(size_t i=0;i<roots.size();++i){
    const auto& t=roots[i];if(!t.is_view())continue;
    const auto& base=t._base();
    if(base.requires_grad()&&base.is_contiguous()&&t.is_contiguous()&&t.numel()
       &&!t.is_conj()&&!t.is_neg()&&t.scalar_type()==base.scalar_type())
      groups[base.unsafeGetTensorImpl()].push_back(i);
  }
  std::map<size_t,std::pair<at::Tensor,at::Tensor>> replacements;std::set<size_t> removed;
  for(const auto& [_,ids]:groups){
    if(ids.size()<2)continue;
    const auto base=at::Tensor(roots[ids[0]]._base());auto ordered=ids;
    std::sort(ordered.begin(),ordered.end(),[&](size_t a,size_t b){return roots[a].storage_offset()<roots[b].storage_offset();});
    int64_t cursor=base.storage_offset();bool complete=true;
    for(auto i:ordered){if(roots[i].storage_offset()!=cursor){complete=false;break;}cursor+=roots[i].numel();}
    if(!complete||cursor!=base.storage_offset()+base.numel())continue;
    std::vector<at::Tensor> parts;for(auto i:ordered)parts.push_back(bars[i].reshape({-1}));
    replacements.emplace(ids[0],std::make_pair(base,at::cat(parts).reshape_as(base)));
    removed.insert(ids.begin()+1,ids.end());
    work::packing::projection("query_vjp_root_partition",ids.size());
  }
  std::vector<at::Tensor> next_roots,next_bars;
  for(size_t i=0;i<roots.size();++i)if(!removed.count(i)){
    auto found=replacements.find(i);
    next_roots.push_back(found==replacements.end()?roots[i]:found->second.first);
    next_bars.push_back(found==replacements.end()?bars[i]:found->second.second);
  }
  roots=std::move(next_roots);bars=std::move(next_bars);
}
}
