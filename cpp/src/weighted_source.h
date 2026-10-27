#pragma once
// Architecture §2.2: shared base matrices and optional private low-rank factors.
#include "weighted_detail.h"
namespace tide::weighted::detail {
inline std::pair<Tensor,Tensor> source_matrices(const NodeWeights& w,const std::string& p){
  auto a=weight(w,p+".a"),b=weight(w,p+".b");
  Index count=0;for(const auto& name:{"a_left","a_right","b_left","b_right"})count+=w.extra.count(p+"."+name);
  require(count==0||count==4,"partial source adaptation factors");
  if(count){a=a+at::matmul(weight(w,p+".a_left"),weight(w,p+".a_right"));b=b+at::matmul(weight(w,p+".b_left"),weight(w,p+".b_right"));}
  return {a,b};
}
inline void validate_source(const NodeWeights& w,const std::string& p,Index d){
  const auto& a=weight(w,p+".a");require(a.dim()==2&&a.size(1)==d,"invalid source A shape");
  const auto h=a.size(0);require(h>0,"invalid source hidden width");shape(w,p+".b",{d,h});shape(w,p+".embedding",{h});
  Index count=0;for(const auto& name:{"a_left","a_right","b_left","b_right"})count+=w.extra.count(p+"."+name);
  require(count==0||count==4,"partial source adaptation factors");
  if(count){
    const auto& l=weight(w,p+".a_left");require(l.dim()==2&&l.size(0)==h&&l.size(1)>0,"invalid source adaptation rank/shape");
    const auto r=l.size(1);shape(w,p+".a_right",{r,d});shape(w,p+".b_left",{d,r});shape(w,p+".b_right",{r,h});
  }
}
}
