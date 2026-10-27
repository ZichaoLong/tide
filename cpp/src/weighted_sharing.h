#pragma once
// Architecture §2.2 / FC §6.2: same ordered class/input and parameter identities.
#include "weighted_detail.h"
#include "tide/aggregate.h"
#include <cstdint>
namespace tide::weighted::detail {
inline std::string aggregate_key(const NodeWeights& w,const AggregateInput& r,const SourceClasses& classes,double alpha,double epsilon){
  std::string key;
  auto append=[&](const auto& v){key.append(reinterpret_cast<const char*>(&v),sizeof(v));};
  auto tensor=[&](const Tensor& t){const auto p=reinterpret_cast<std::uintptr_t>(t.unsafeGetTensorImpl());append(p);};
  append(alpha);append(epsilon);append(r.sources.size());
  for(const auto& source:r.sources){
    auto label=classes.classify(source.atom);append(label);tensor(source.atom.value);
    if(alpha){
      auto p="aggregate_program.transforms."+std::to_string(label)+".";
      for(const auto& name:{"a","b","embedding"})tensor(weight(w,p+name));
      bool adapted=w.extra.count(p+"a_left");append(adapted);
      if(adapted)for(const auto& name:{"a_left","a_right","b_left","b_right"})tensor(weight(w,p+name));
    }
  }
  return key;
}
}
