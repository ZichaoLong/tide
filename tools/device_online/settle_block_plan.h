#pragma once
#include "block_task_plan.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
// One task represents a whole certified B x L tensor window for that owner.
// Group identities describe numerical roles, never benchmark names or results.
enum class SettleBlockRole : int64_t { Local=0, Region=1, Gate=2, Full=3 };
struct SettleBlockPlan {
  BlockDispatchPlan dispatch;
  std::vector<int64_t> local,region,gate,full,input_tasks,edge_dependencies;
  std::vector<int64_t> member_offsets,members;
};

// Metadata-only lowering. Every potential edge is retained, including parallel
// edges. No inputs, c values, readiness masks or selected sequence are accepted.
inline SettleBlockPlan make_settle_block_plan(
    const std::vector<int64_t>& node_region,const std::vector<int64_t>& ranks,
    const std::vector<std::array<int64_t,3>>& edges,const std::vector<int64_t>& inputs,
    int64_t first_position,int64_t stride,int64_t batch_capacity,int64_t tensor_bytes){
  using I=int64_t;const I nodes=node_region.size(),regions=ranks.size();
  if(nodes<1||regions<1||first_position<0||stride<1||batch_capacity<1||tensor_bytes<1||
     nodes>(std::numeric_limits<I>::max()-regions)/3)
    throw std::invalid_argument("invalid Settle block layout");
  std::vector<I> count(regions,0);
  for(auto r:node_region){if(r<0||r>=regions)throw std::invalid_argument("invalid Settle node region");++count[r];}
  for(auto n:count)if(!n)throw std::invalid_argument("empty Settle block region");
  for(auto rank:ranks)if(rank<0||first_position>(std::numeric_limits<I>::max()-rank)/stride)
    throw std::invalid_argument("Settle block logical time overflow");
  for(const auto& e:edges){
    if(e[0]<0||e[0]>=nodes||e[1]<0||e[1]>=nodes||e[2]<1)
      throw std::invalid_argument("invalid Settle block edge");
    if(ranks[node_region[e[1]]]-ranks[node_region[e[0]]]!=e[2])
      throw std::invalid_argument("Settle block lowering requires exact rank-delay alignment");
  }
  for(auto v:inputs)if(v<0||v>=nodes)throw std::invalid_argument("invalid Settle input target");
  const I tasks=3*nodes+regions;
  const long double dependencies=3.L*nodes+edges.size();
  const long double bytes=90.L*tasks+24.L*dependencies+9.L*batch_capacity+24.L*tasks+80;
  if(bytes>tensor_bytes)throw std::invalid_argument("Settle block metadata budget exceeded before lowering");
  SettleBlockPlan out;auto& d=out.dispatch;d.groups=4;d.batch_capacity=batch_capacity;
  d.journal_capacity=tasks;d.tensor_bytes=tensor_bytes;d.tasks.resize(tasks);
  for(I v=0;v<nodes;++v){
    const I time=first_position*stride+ranks[node_region[v]];
    out.local.push_back(v);out.gate.push_back(nodes+regions+v);out.full.push_back(2*nodes+regions+v);
    d.tasks[v]={0,0,v,time,0};d.tasks[out.gate.back()]={2,0,v,time,0};d.tasks[out.full.back()]={3,0,v,time,0};
  }
  for(I r=0;r<regions;++r){out.region.push_back(nodes+r);d.tasks[nodes+r]={1,0,r,first_position*stride+ranks[r],0};}
  for(I v=0;v<nodes;++v){
    d.dependencies.push_back({out.local[v],out.region[node_region[v]],1});
    d.dependencies.push_back({out.region[node_region[v]],out.gate[v],1});
    // The device Gate reads this node's controls, not any(c>0) for its region.
    d.dependencies.push_back({out.gate[v],out.full[v],1});
  }
  for(const auto& e:edges){out.edge_dependencies.push_back(d.dependencies.size());d.dependencies.push_back({out.full[e[0]],out.local[e[1]],1});}
  for(auto v:inputs)out.input_tasks.push_back(out.local[v]);
  out.member_offsets.push_back(0);
  for(I r=0;r<regions;++r){for(I v=0;v<nodes;++v)if(node_region[v]==r)out.members.push_back(v);out.member_offsets.push_back(out.members.size());}
  return out;
}
}
