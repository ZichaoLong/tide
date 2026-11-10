#include "tide/operator_profile.h"
#include <ATen/record_function.h>
#include "tide/state_preparation.h"
#include "tide/kernel.h"
#include <typeinfo>
#include <stdexcept>
#include <algorithm>
namespace tide {
void prepare_shared_state(const Model& m,std::vector<Event>& events,std::map<std::string,Index>& stats){
  RECORD_FUNCTION("tide::stage::state_prepare", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::Preparation);
  std::map<std::string,std::vector<size_t>> blocks;
  std::vector<bool> handled(events.size());
  for(size_t i=0;i<events.size();++i){events[i].state_prepared.clear();const auto& k=m.nodes[events[i].node].kernel;
    if(k->block_preparation())blocks[typeid(*k).name()].push_back(i);}
  for(const auto& [key,ids]:blocks){
    std::vector<const NodeWeights*> rows;std::vector<Tensor> contents;
    for(auto i:ids){rows.push_back(&m.nodes[events[i].node]);contents.push_back(events[i].content);}
    auto values=rows[0]->kernel->prepare_block_batch(rows,contents,stats);
    if(values.empty())continue;
    if(values.size()!=ids.size())throw std::invalid_argument("block preparation row count mismatch");
    for(size_t j=0;j<ids.size();++j){events[ids[j]].state_prepared=std::move(values[j]);handled[ids[j]]=true;}
  }
  std::map<std::pair<std::string,std::string>,std::vector<size_t>> groups;
  for(size_t i=0;i<events.size();++i){
    if(handled[i])continue;
    const auto& e=events[i];const auto& w=m.nodes[e.node];
    auto key=w.kernel->preparation_key(w,e.content);
    if(!key.empty())groups[{typeid(*w.kernel).name(),key}].push_back(i);
  }
  // Keys establish equality and may contain TensorImpl addresses. They must
  // not order autograd node creation: restoring equal values into new owners
  // would otherwise change the reduction order of their shared gradients.
  std::vector<std::vector<size_t>> ordered;
  for(auto& [key,ids]:groups)ordered.push_back(std::move(ids));
  std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.front()<b.front();});
  for(const auto& ids:ordered){
    if(ids.size()<2)continue;
    const auto& first=events[ids[0]];const auto& w=m.nodes[first.node];
    auto values=w.kernel->prepare_shared(w,first.content);
    std::vector<const NodeWeights*> rows;for(auto i:ids)rows.push_back(&m.nodes[events[i].node]);
    auto private_values=w.kernel->prepare_private_batch(rows,values,stats);
    if(private_values.size()!=ids.size())throw std::invalid_argument("private preparation row count mismatch");
    for(size_t j=0;j<ids.size();++j)events[ids[j]].state_prepared=std::move(private_values[j]);
    ++stats["state_preparation_calls"];
    stats["state_preparation_events"]+=ids.size();
    stats["state_preparation_reused_events"]+=ids.size()-1;
  }
  std::map<std::string,std::vector<size_t>> independent;
  for(size_t i=0;i<events.size();++i){
    if(handled[i])continue;
    const auto& kernel=m.nodes[events[i].node].kernel;
    if(events[i].state_prepared.empty()&&kernel->independent_preparation())
      independent[typeid(*kernel).name()].push_back(i);
  }
  ordered.clear();for(auto& [key,ids]:independent)ordered.push_back(std::move(ids));
  std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.front()<b.front();});
  for(const auto& ids:ordered){
    if(ids.size()<2)continue;
    std::vector<const NodeWeights*> rows;std::vector<Tensor> contents;
    for(auto i:ids){rows.push_back(&m.nodes[events[i].node]);contents.push_back(events[i].content);}
    auto values=rows[0]->kernel->prepare_independent_batch(rows,contents,stats);
    if(values.size()!=ids.size())throw std::invalid_argument("independent preparation row count mismatch");
    for(size_t j=0;j<ids.size();++j)events[ids[j]].state_prepared=std::move(values[j]);
  }
}
}
