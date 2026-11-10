#include "tide/greedy.h"
#include "tide/delivery.h"
#include "tide/ops.h"
#include "component_plan.h"
#include "stream_support.h"
#include <algorithm>

namespace tide {
Result Greedy::run_window(const Continuation& initial,const std::vector<External>& external,Index stop,Index seal){
  Result result;auto& q=result.continuation;q=initial;
  auto atoms=validate_window(graph_,model_,q,external,stop,seal);
  atoms.insert(atoms.end(),q.pending.begin(),q.pending.end());
  const auto& plan=*components_;std::vector<Fibers> partitions(plan.groups.size());Index live=0;
  auto insert=[&](const Atom& atom){auto& fibers=partitions[plan.owner[graph_.nodes[atom.node].region]];
    auto [it,created]=fibers.try_emplace({atom.batch,atom.node,atom.time});live+=created;it->second.push_back(atom);};
  for(const auto& atom:atoms)insert(atom);
  auto& stats=result.stats;
  stats={{"greedy_stages",0},{"greedy_relaxed_edges",0},{"region_blocks",0},{"candidate_events",0},
         {"visited_edges",0},{"max_live_fibers",live},{"window_components",Index(plan.groups.size())},
         {"window_cyclic_components",Index(std::count(plan.cyclic.begin(),plan.cyclic.end(),true))},{"window_causal_blocks",0}};
  if(live>options_.max_events)throw std::invalid_argument("window live-fiber capacity exceeded");
  for(size_t c=0;c<partitions.size();++c){
    auto& fibers=partitions[c];
    while(true){
      std::map<Index,std::vector<Frame>> blocks;
      if(plan.cyclic[c]){
        blocks=plan.closures[c]->ready(graph_,fibers,stop,options_.max_events,stats["greedy_relaxed_edges"]);
        if(!blocks.empty())++stats["window_causal_blocks"];
      }else{
        std::map<Owner,std::set<Index>> frames;
        for(const auto& [key,values]:fibers){auto [batch,node,time]=key;if(!values.empty()&&time<stop)frames[{time,batch}].insert(node);}
        if(!frames.empty()){auto region=plan.groups[c][0];for(const auto& [key,nodes]:frames)
          blocks[region].push_back({key.second,region,key.first,nodes,{}});}
      }
      if(blocks.empty())break;++stats["greedy_stages"];
      for(const auto& [region,frames]:blocks){
        auto events=evaluate_block(graph_,model_,q,frames,fibers,options_,pool_,stats);
        for(const auto& frame:frames)for(auto node:frame.nodes)live-=fibers.erase({frame.batch,node,frame.time});
        ++stats["region_blocks"];stats["candidate_events"]+=events.size();
        stats["max_greedy_frames"]=std::max<Index>(stats["max_greedy_frames"],frames.size());
        deliver_batch(graph_,model_,events,options_.packed,stats,[&](const Atom& atom){
          ++stats["visited_edges"];insert(atom);if(options_.trace)result.messages.push_back(atom);
        },[&](const Output& output){result.outputs.push_back(output);});
        for(auto& event:events)if(options_.trace)result.trace.push_back(std::move(event));
        if(options_.compact_events&&!options_.trace)release_stream_events(events,pool_,options_.workers);
        stats["max_live_fibers"]=std::max(stats["max_live_fibers"],live);
        if(live>options_.max_events)throw std::invalid_argument("window live-fiber capacity exceeded");
      }
    }
  }
  q.pending.clear();for(const auto& fibers:partitions)for(const auto& [key,bucket]:fibers)
    for(const auto& atom:bucket)if(atom.kind==1)q.pending.push_back(atom);
  q.cut=stop;
  std::sort(q.pending.begin(),q.pending.end(),[](const auto& a,const auto& b){return a.key()<b.key();});
  std::sort(result.trace.begin(),result.trace.end(),[](const auto& a,const auto& b){return std::tie(a.time,a.batch,a.node)<std::tie(b.time,b.batch,b.node);});
  std::sort(result.messages.begin(),result.messages.end(),[&](const auto& a,const auto& b){
    return std::tie(a.position,a.batch,graph_.edges[a.source].source,a.source)<std::tie(b.position,b.batch,graph_.edges[b.source].source,b.source);});
  std::sort(result.outputs.begin(),result.outputs.end(),[&](const auto& a,const auto& b){
    return std::tie(a.time,a.batch,graph_.outputs[a.port],a.port)<std::tie(b.time,b.batch,graph_.outputs[b.port],b.port);});
  return result;
}
}
