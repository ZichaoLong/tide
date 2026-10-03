#pragma once
#include "consumer.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <set>

namespace tide_flow::eager_capacity {
inline Index sum(std::initializer_list<Index> values) {
  Index result=0;for(auto v:values) {
    if(v<0||v>std::numeric_limits<Index>::max()-result)throw std::invalid_argument("consumer traffic extent overflow");
    result+=v;
  }return result;
}
inline Index product(std::initializer_list<Index> values) {
  Index result=1;for(auto v:values) {
    if(v<0||(v&&result>std::numeric_limits<Index>::max()/v))throw std::invalid_argument("consumer traffic extent overflow");
    result*=v;
  }return result;
}
inline void array(std::ostream& out,const std::vector<Index>& x) {
  out<<'[';for(size_t i=0;i<x.size();++i){if(i)out<<',';out<<x[i];}out<<']';
}
inline void fields(std::ostream& out,const std::map<std::string,Index>& values) {
  out<<'{';bool first=true;for(const auto& [k,v]:values){if(!first)out<<',';first=false;out<<quoted(k)<<':'<<v;}out<<'}';
}
struct Traffic {
  std::vector<Index> low,high,region_frames,node_events,node_atoms,owner_atoms,owner_events,owner_emissions;
  Index output_atoms=0,output_frames=0,maximum_offset=0;
};
inline Traffic traffic_bounds(const Packet& p,const std::vector<Index>& owners) {
  const auto& g=p.body;const Index n=g.nodes.size(),regions=g.regions.size();
  std::set<Index> used(owners.begin(),owners.end());
  if(Index(owners.size())!=n+2||owners[n]!=0||owners[n+1]!=0||*used.begin()!=0||*used.rbegin()!=Index(used.size())-1)
    throw std::invalid_argument("traffic owners must cover encoded nodes with owner-zero boundaries");
  const Index devices=used.size();Traffic result;auto& q=result;
  std::vector<std::vector<Index>> members(regions);std::vector<Index> indegree(n),external(n),readout(n);
  for(Index i=0;i<n;++i)members[g.nodes[i].region].push_back(i);
  for(const auto& row:members)if(p.budget<1||Index(row.size())<p.budget)throw std::invalid_argument("invalid traffic region budget");
  for(const auto& e:g.edges)++indegree[e.target];
  for(auto i:g.inputs)++external[i];for(auto i:g.outputs)++readout[i];
  q.low.assign(n,std::numeric_limits<Index>::max());q.high.assign(n,-1);
  const bool aligned=p.topology=="ranked-local";
  for(auto i:g.inputs)q.low[i]=q.high[i]=aligned?p.ranks[g.nodes[i].region]:1;
  std::deque<Index> ready;for(Index i=0;i<n;++i)if(!indegree[i])ready.push_back(i);
  Index visited=0;
  while(!ready.empty()) {
    const auto i=ready.front();ready.pop_front();++visited;
    if(q.high[i]<0)throw std::invalid_argument("traffic graph has an input-unreachable node");
    for(Index j=g.csr.offsets[i];j<g.csr.offsets[i+1];++j) {
      const auto& e=g.edges[g.csr.edges[j]];
      q.low[e.target]=std::min(q.low[e.target],sum({q.low[i],e.delay}));
      q.high[e.target]=std::max(q.high[e.target],sum({q.high[i],e.delay}));
      if(!--indegree[e.target])ready.push_back(e.target);
    }
  }
  if(visited!=n)throw std::invalid_argument("finite DAG traffic accounting cannot admit a feedback packet");
  q.node_events.resize(n);q.node_atoms=external;
  for(Index i=0;i<n;++i) {
    if(aligned&&(q.low[i]!=q.high[i]||q.low[i]!=p.ranks[g.nodes[i].region]))throw std::invalid_argument("rank-aligned traffic packet has incompatible delays");
    q.node_events[i]=q.high[i]-q.low[i]+1;
  }
  const Index readout_rank=sum({*std::max_element(p.ranks.begin(),p.ranks.end()),1});
  Index output_low=std::numeric_limits<Index>::max();
  for(auto i:g.outputs) {
    output_low=std::min(output_low,aligned?readout_rank:sum({q.low[i],1}));
    q.maximum_offset=std::max(q.maximum_offset,aligned?readout_rank:sum({q.high[i],1}));
  }
  if(*std::min_element(q.low.begin(),q.low.end())<1||std::max(*std::max_element(q.high.begin(),q.high.end()),q.maximum_offset)>=p.stride)
    throw std::invalid_argument("traffic position is not sealed before the next injection");
  q.region_frames.resize(regions);
  for(Index r=0;r<regions;++r) {
    Index lo=std::numeric_limits<Index>::max(),hi=0;
    for(auto i:members[r]){lo=std::min(lo,q.low[i]);hi=std::max(hi,q.high[i]);}
    q.region_frames[r]=hi-lo+1;
  }
  auto top=[&](std::vector<Index> values) {
    std::sort(values.begin(),values.end(),std::greater<Index>());Index count=0;
    for(Index i=0;i<std::min<Index>(values.size(),p.budget);++i)count=sum({count,values[i]});return count;
  };
  for(Index i=0;i<n;++i) {
    std::map<Index,std::map<Index,Index>> counts;
    for(Index j=g.csc.offsets[i];j<g.csc.offsets[i+1];++j){auto a=g.edges[g.csc.edges[j]].source;++counts[g.nodes[a].region][a];}
    for(const auto& [r,values]:counts){std::vector<Index> costs;for(const auto& [_,v]:values)costs.push_back(v);
      q.node_atoms[i]=sum({q.node_atoms[i],product({q.region_frames[r],top(costs)})});}
    q.node_events[i]=std::min(q.node_events[i],q.node_atoms[i]);
  }
  q.owner_atoms.assign(devices,0);q.owner_events.assign(devices,0);q.owner_emissions.assign(devices,0);
  std::vector<std::vector<Index>> targets(n,std::vector<Index>(devices));
  for(Index i=0;i<n;++i)q.owner_atoms[owners[i]]+=external[i];
  for(const auto& e:g.edges)++targets[e.source][owners[e.target]];
  for(Index r=0;r<regions;++r) {
    std::vector<Index> outs;for(auto i:members[r])outs.push_back(readout[i]);
    q.output_atoms=sum({q.output_atoms,product({q.region_frames[r],top(outs)})});
    for(Index d=0;d<devices;++d) {
      std::vector<Index> incoming,emitted;
      for(auto i:members[r]){incoming.push_back(targets[i][d]);if(owners[i]==d)emitted.push_back(g.csr.offsets[i+1]-g.csr.offsets[i]);}
      q.owner_atoms[d]=sum({q.owner_atoms[d],product({q.region_frames[r],top(incoming)})});
      q.owner_emissions[d]=sum({q.owner_emissions[d],product({q.region_frames[r],top(emitted)})});
    }
  }
  for(Index i=0;i<n;++i)q.owner_events[owners[i]]=sum({q.owner_events[owners[i]],q.node_events[i]});
  for(Index d=0;d<devices;++d)q.owner_events[d]=std::min(q.owner_events[d],q.owner_atoms[d]);
  q.output_frames=std::min(q.output_atoms,q.maximum_offset-output_low+1);return q;
}
inline void record(std::ostream& out,const Traffic& q) {
  out<<"{\"schema\":\"tide-consumer-traffic-bound-v1\",\"scope\":\"per sample/input position; validated DAG packet, empty initial state; all legal selections; not actual events\",\"node_offset_intervals\":[";
  for(size_t i=0;i<q.low.size();++i){if(i)out<<',';out<<'['<<q.low[i]<<','<<q.high[i]<<']';}out<<']';
  for(const auto& [key,values]:std::map<std::string,std::vector<Index>>{{"region_frame_factors",q.region_frames},{"node_event_factors",q.node_events},{"node_atom_factors",q.node_atoms},
      {"owner_body_atom_factors",q.owner_atoms},{"owner_body_event_factors",q.owner_events},{"owner_body_emission_factors",q.owner_emissions}}) {
    out<<','<<quoted(key)<<':';array(out,values);
  }
  out<<",\"output_atom_factor\":"<<q.output_atoms<<",\"output_frame_factor\":"<<q.output_frames<<",\"maximum_position_offset\":"<<q.maximum_offset<<",\"positions_are_sealed\":true}";
}
} // namespace tide_flow::eager_capacity
