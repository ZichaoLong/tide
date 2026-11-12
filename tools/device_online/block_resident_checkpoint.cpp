#include "block_resident_internal.h"
#include "tide/weighted_column_checkpoint.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
BlockSettleCheckpoint import_block_checkpoint(const BlockCheckpoint& c){
  if(c.optimizer!=ResidentOptimizerKind::sgd&&c.optimizer!=ResidentOptimizerKind::adamw)throw std::invalid_argument("unknown block checkpoint optimizer");
  BlockSettleCheckpoint out;out.schema=c.schema;out.generation=c.generation;out.next_token=c.next_token;out.identity=c.identity;
  out.aliases=c.aliases;out.parameters=c.parameters;out.optimizer=c.optimizer==ResidentOptimizerKind::sgd?DeviceOptimizerKind::sgd:DeviceOptimizerKind::adamw;
  out.groups=c.groups;out.slots=c.slots;out.last_positions=c.last_positions;
  out.boundary={c.node_present,c.region_times,c.input_counts,c.input_times};
  for(const auto& s:c.state)out.state.push_back({s.floating,s.integer});for(const auto& s:c.bound_initial)out.bound_initial.push_back({s.floating,s.integer});return out;
}
BlockCheckpoint export_block_checkpoint(const BlockSettleCheckpoint& c){
  BlockCheckpoint out;out.schema=c.schema;out.generation=c.generation;out.next_token=c.next_token;out.identity=c.identity;
  out.aliases=c.aliases;out.parameters=c.parameters;out.optimizer=c.optimizer==DeviceOptimizerKind::sgd?ResidentOptimizerKind::sgd:ResidentOptimizerKind::adamw;
  out.groups=c.groups;out.slots=c.slots;out.last_positions=c.last_positions;
  out.node_present=c.boundary.node_present;out.region_times=c.boundary.region_times;out.input_counts=c.boundary.input_counts;out.input_times=c.boundary.input_times;
  for(const auto& s:c.state)out.state.push_back({s.floating,s.integer});for(const auto& s:c.bound_initial)out.bound_initial.push_back({s.floating,s.integer});return out;
}
BlockCheckpoint BlockResidentSession::checkpoint()const{impl_->check();return export_block_checkpoint(impl_->owner->checkpoint());}
Continuation BlockResidentSession::continuation()const{
  const auto& s=*impl_;s.check();const auto actual=s.owner->checkpoint();
  Continuation out;out.identity=s.graph.graph().identity;out.batch_size=s.batch;out.cut=s.cut;
  const auto& graph=s.graph.graph();
  for(size_t r=0;r<s.binding.bindings.size();++r){std::vector<Index> nodes;
    for(size_t v=0;v<graph.nodes.size();++v)if(graph.nodes[v].region==int64_t(r))nodes.push_back(v);
    const WeightedRawLayout layout(s.binding.bindings[r].memory,actual.state[r].integer.size(1)-4);
    const auto state=decode_weighted_raw_state(layout,actual.state[r]);
    for(Index b=0;b<s.batch;++b)for(size_t n=0;n<nodes.size();++n)if(actual.boundary.node_present[b][nodes[n]].item<bool>())out.states[{b,nodes[n]}]=state[b*nodes.size()+n];
    for(Index b=0;b<s.batch;++b){const auto time=actual.boundary.region_times[b][r].item<Index>();if(time>=0)out.history[{b,Index(r)}].last_time=time;}
  }
  for(Index b=0;b<s.batch;++b)for(size_t p=0;p<graph.inputs.size();++p){const auto count=actual.boundary.input_counts[b][p].item<Index>();
    if(count>=0)out.ledger[{b,Index(p)}]={count,actual.boundary.input_times[b][p].item<Index>()};}
  // Validation only: no reference recurrence or synthetic missing state.
  weighted::columns::validate_owner(s.graph,s.model,out,s.batch);return out;
}
void save_block_checkpoint(const BlockCheckpoint& c,const std::string& path,int64_t max_bytes){save_block_settle_checkpoint(import_block_checkpoint(c),path,max_bytes);}
BlockCheckpoint load_block_checkpoint(const std::string& path,int64_t max_bytes){return export_block_checkpoint(load_block_settle_checkpoint(path,max_bytes));}
}
