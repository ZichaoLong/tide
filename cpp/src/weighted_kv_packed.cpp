#include "tide/operator_profile.h"
#include <ATen/record_function.h>
// KV §2/§5: one projection batch, actual raw caches at every positive prefix.
#include "weighted_kv_projection.h"
#include <algorithm>

namespace tide::weighted::packed {
using detail::weight;
PackedStates kv(const NodeWeights& w,const std::vector<State>& old,const PackedSequence& batch,const KVConfig& c){
  RECORD_FUNCTION("tide::stage::state_snapshots", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::StateOther);
  Writes plan(old,batch);if(plan.ids.empty())return plan.decode(old,batch,{},0);
  auto [values,summaries]=kv_projections(w,plan,batch,c);
  std::map<std::string,Tensor> entries{{"times",plan.times},{"keys",values.slice(1,0,c.width)},
    {"values",values.slice(1,c.width)},{"summaries",summaries},{"weights",plan.receive}};
  std::vector<size_t> needs;std::vector<Tensor> stored;
  for(size_t i=0;i<old.size();++i)if(plan.offsets[i]<plan.offsets[i+1]&&c.window
      &&batch.times[plan.ids[plan.offsets[i+1]-1]]>=c.window){
    needs.push_back(i);stored.push_back(old[i].slots.at("times"));
  }
  std::map<size_t,std::vector<Index>> stamps;
  if(!stored.empty()){
    auto host=at::cat(stored).to(at::kCPU).contiguous();const auto* data=host.data_ptr<Index>();Index offset=0;
    for(auto i:needs){auto n=old[i].slots.at("times").numel();
      stamps[i]=n?std::vector<Index>(data+offset,data+offset+n):std::vector<Index>{};offset+=n;}
  }
  std::vector<std::map<std::string,Tensor>> slots;
  for(size_t i=0;i<old.size();++i){auto a=plan.offsets[i],b=plan.offsets[i+1];if(a==b)continue;
    std::map<std::string,Tensor> joined;
    for(const auto& [name,v]:entries)joined[name]=at::cat({old[i].slots.at(name),v.slice(0,a,b)});
    const auto n=old[i].slots.at("times").numel();auto found=stamps.find(i);
    if(found!=stamps.end())for(Index j=a;j<b;++j)found->second.push_back(batch.times[plan.ids[j]]);
    for(Index j=a;j<b;++j){
      const auto end=n+j-a+1;Index start=0;
      if(found!=stamps.end()){
        const auto& clock=found->second;
        start=std::lower_bound(clock.begin(),clock.begin()+end,batch.times[plan.ids[j]]-(c.window-1))-clock.begin();
      }
      std::map<std::string,Tensor> row;
      for(const auto& [name,v]:joined)row[name]=v.slice(0,start,end);
      slots.push_back(std::move(row));
    }
  }
  return plan.decode(old,batch,slots,1);
}
}
