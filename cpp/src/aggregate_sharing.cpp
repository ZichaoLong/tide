// Optional pure-function reuse; physical atoms and private states remain distinct.
#include "tide/aggregate_sharing.h"
#include <typeinfo>
namespace tide {
SharedAggregates plan_aggregate_sharing(const Graph& g,const Model& m,const std::vector<Event>& events,
    const std::map<Index,std::vector<size_t>>& by_node,bool packed,std::map<std::string,Index>& stats){
  SharedAggregates result;std::map<std::pair<std::string,std::string>,size_t> cache;
  for(const auto& [node,ids]:by_node){
    const auto& w=m.nodes[node];const auto& kernel=*w.aggregate_kernel;
    for(auto id:ids){
      auto key=packed?kernel.sharing_key(w,aggregate_request(g,m,events[id])):std::string{};
      auto tagged=std::make_pair(std::string(typeid(kernel).name()),key);
      auto previous=cache.find(tagged);
      if(!key.empty()&&previous!=cache.end()){
        result.copies.emplace_back(id,previous->second);++stats["aggregate_reused_events"];
      }else{
        result.compute[node].push_back(id);++stats["aggregate_computed_events"];
        if(!key.empty())cache.emplace(std::move(tagged),id);
      }
    }
  }
  return result;
}
void decode_shared_aggregates(const Graph& g,const Model& m,std::vector<Event>& events,const SharedAggregates& plan){
  for(const auto& [to,from]:plan.copies){
    auto& target=events[to];const auto& source=events[from];
    target.content=source.content;target.contributions=source.contributions;
    target.sources=aggregate_request(g,m,target).sources;
    // The optional source transport is local to its original physical layout.
    // Keep this event's actual sources; never reuse another node's source slots.
    target.source_batch.reset();target.source_row=-1;
  }
}
}
