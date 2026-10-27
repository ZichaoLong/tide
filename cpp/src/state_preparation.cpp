#include "tide/state_preparation.h"
#include "tide/kernel.h"
#include <typeinfo>
namespace tide {
void prepare_shared_state(const Model& m,std::vector<Event>& events,std::map<std::string,Index>& stats){
  std::map<std::pair<std::string,std::string>,std::vector<size_t>> groups;
  for(size_t i=0;i<events.size();++i){
    events[i].state_prepared.clear();
    const auto& e=events[i];const auto& w=m.nodes[e.node];
    auto key=w.kernel->preparation_key(w,e.content);
    if(!key.empty())groups[{typeid(*w.kernel).name(),key}].push_back(i);
  }
  for(const auto& [key,ids]:groups){
    if(ids.size()<2)continue;
    const auto& first=events[ids[0]];const auto& w=m.nodes[first.node];
    auto values=w.kernel->prepare_shared(w,first.content);
    for(auto i:ids)events[i].state_prepared=values;
    ++stats["state_preparation_calls"];
    stats["state_preparation_events"]+=ids.size();
    stats["state_preparation_reused_events"]+=ids.size()-1;
  }
}
}
