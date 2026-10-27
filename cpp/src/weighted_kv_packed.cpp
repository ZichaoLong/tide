// KV §2/§5: one projection batch, actual raw caches at every positive prefix.
#include "weighted_packed.h"

namespace tide::weighted::packed {
using detail::weight;
PackedStates kv(const NodeWeights& w,const std::vector<State>& old,const PackedSequence& batch,const KVConfig& c){
  Writes plan(old,batch);if(plan.ids.empty())return plan.decode(old,batch,{},0);
  auto x=normalize(plan.input,c.epsilon);
  auto matrix=weight(w,"kernel.base")+at::matmul(weight(w,"kernel.adapter_u"),weight(w,"kernel.adapter_v"));
  auto values=project(x,matrix);
  std::map<std::string,Tensor> entries{{"times",plan.times},{"keys",values.slice(1,0,c.width)},
    {"values",values.slice(1,c.width)},{"summaries",project(normalize(plan.input,c.epsilon),weight(w,"kernel.summary"))},{"weights",plan.receive}};
  std::vector<std::map<std::string,Tensor>> slots;
  for(size_t i=0;i<old.size();++i){auto a=plan.offsets[i],b=plan.offsets[i+1];if(a==b)continue;
    std::map<std::string,Tensor> joined;
    for(const auto& [name,v]:entries)joined[name]=at::cat({old[i].slots.at(name),v.slice(0,a,b)});
    for(Index j=a;j<b;++j){
      auto age=plan.times[j]-joined.at("times"),visible=age>=0;if(c.window)visible=visible&(age<c.window);
      std::map<std::string,Tensor> row;
      for(const auto& [name,v]:joined)row[name]=v.index({visible});
      slots.push_back(std::move(row));
    }
  }
  return plan.decode(old,batch,slots,1);
}
}
