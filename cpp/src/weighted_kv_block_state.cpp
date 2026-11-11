#include "weighted_kv_block.h"
#include <set>

namespace tide::weighted::block {
bool View::valid(const State& state)const{
  if(!state.value.is_same(snapshot.value)||state.slots.size()!=snapshot.slots.size())return false;
  for(const auto& [name,t]:snapshot.slots){auto it=state.slots.find(name);
    if(it==state.slots.end()||!it->second.is_same(t))return false;}
  return true;
}
std::shared_ptr<const View> view(const State* state){
  if(!state)return {};
  auto v=std::dynamic_pointer_cast<const View>(state->execution_view);
  return v&&v->valid(*state)?v:nullptr;
}
void attach(const NodeWeights& w,const std::vector<State>& old,const PackedSequence& p,std::vector<State>& states){
  if(p.prepared.empty())return;
  for(size_t b=0;b<old.size();++b){
    std::vector<Index> ids;
    for(Index j=p.offsets[b];j<p.offsets[b+1];++j)if(states[j].last_time==p.times[j])ids.push_back(j);
    if(ids.size()<2)continue;
    bool valid=true;
    for(auto j:ids)valid&=p.prepared[j].size()==2&&p.prepared[j].count("values")&&p.prepared[j].count("summaries");
    if(!valid)continue;
    auto bank=std::make_shared<Bank>();bank->old=old[b];
    auto stamps=old[b].slots.at("times").to(at::kCPU).contiguous();
    const auto* ptr=stamps.const_data_ptr<Index>();
    if(stamps.numel())bank->times.assign(ptr,ptr+stamps.numel());
    for(auto j:ids){bank->times.push_back(p.times[j]);bank->values.push_back(p.prepared[j].at("values"));
      bank->summaries.push_back(p.prepared[j].at("summaries"));bank->contents.push_back(p.views[j].value);}
    bank->guards={w.decay,w.weight,w.bias,w.read,old[b].value};
    for(const auto& [name,t]:w.extra)bank->guards.push_back(t);
    for(const auto& [name,t]:old[b].slots)bank->guards.push_back(t);
    Index count=stamps.numel();std::set<Index> positive(ids.begin(),ids.end());
    for(Index j=p.offsets[b];j<ids.back();++j){
      count+=positive.count(j);auto& state=states[j];
      if(state.last_time==old[b].last_time||state.slots.at("times").numel()!=count)continue;
      auto metadata=std::make_shared<View>();metadata->bank=bank;metadata->end=count;
      metadata->snapshot=state;metadata->snapshot.execution_view.reset();state.execution_view=metadata;
    }
  }
}
}
