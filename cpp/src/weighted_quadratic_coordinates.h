#pragma once
// Allocation §6/S.2: cap-weighted score quantile, root in [pivot-tau*max(mu),pivot].
#include "tide/types.h"
namespace tide::weighted::detail {
inline Tensor quad_pivot(const Tensor& scores,const Tensor& caps,const Tensor& target){
  auto order=std::get<1>(at::sort(scores,true,-1,true));
  auto prefix=caps.gather(-1,order).cumsum(-1);
  auto index=(prefix<target.unsqueeze(-1)).sum(-1).clamp_max(scores.size(-1)-1);
  return scores.gather(-1,order.gather(-1,index.unsqueeze(-1))).squeeze(-1);
}
}
