#pragma once
#include "consumer.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>

namespace tide_flow {
struct EagerPlacement {std::vector<Index> owners,elements;};
inline EagerPlacement eager_placement(const Packet& p,const Config& c) {
  const auto n=Index(p.body.nodes.size()),d=p.width;
  if(c.devices<1||c.devices>std::min<Index>(16,n))throw std::invalid_argument("eager devices must be in 1..min(16, body nodes)");
  if(c.owner_policy!="memory"&&c.owner_policy!="locality")throw std::invalid_argument("invalid eager owner policy");
  std::vector<Index> costs(n),members(c.devices),loads(c.devices);
  std::vector<std::vector<Index>> adjacent(n+2);
  for(Index i=0;i<n;++i)costs[i]=(p.body.csr.offsets[i+1]-p.body.csr.offsets[i])*d*d+d+p.body.source_counts[i]+(p.memory=="attention"?4*d*d:0);
  for(const auto& e:p.body.edges){adjacent[e.source].push_back(e.target);adjacent[e.target].push_back(e.source);}
  for(auto v:p.body.inputs)adjacent[v].push_back(n);
  for(auto v:p.body.outputs)adjacent[v].push_back(n+1);
  EagerPlacement result;auto& owners=result.owners;owners=c.owner_map;
  if(!owners.empty()) {
    std::set<Index> used(owners.begin(),owners.end());
    if(Index(owners.size())!=n+2||owners[0]!=0||owners[n]!=0||owners[n+1]!=0||*used.begin()<0||*used.rbegin()>=c.devices||Index(used.size())!=c.devices)
      throw std::invalid_argument("eager owner map must cover encoded nodes, use every device, and keep node zero/boundaries on owner zero");
  } else {
    owners.assign(n+2,-1);owners[0]=owners[n]=owners[n+1]=0;
    loads[0]=2*p.vocab*d+costs[0];members[0]=1;std::vector<Index> order;
    for(Index i=1;i<n;++i)order.push_back(i);
    std::stable_sort(order.begin(),order.end(),[&](Index a,Index b){return costs[a]>costs[b];});
    for(auto node:order) {
      const bool empty=std::find(members.begin(),members.end(),0)!=members.end();
      auto score=[&](Index owner) {
        Index affinity=0;for(auto v:adjacent[node])affinity+=owners[v]==owner;
        if(c.owner_policy=="memory")return std::make_tuple(loads[owner],Index(0),Index(0),owner);
        return std::make_tuple(std::max(*std::max_element(loads.begin(),loads.end()),loads[owner]+costs[node]),-affinity,loads[owner],owner);
      };
      Index best=-1;for(Index owner=0;owner<c.devices;++owner)
        if((!empty||members[owner]==0)&&(best<0||score(owner)<score(best)))best=owner;
      owners[node]=best;loads[best]+=costs[node];++members[best];
    }
  }
  result.elements.assign(c.devices,0);result.elements[0]=2*p.vocab*d;
  for(Index i=0;i<n;++i)result.elements[owners[i]]+=costs[i];
  Index total=0;for(auto x:result.elements)total+=x;
  if(total!=p.parameters())throw std::logic_error("eager parameter accounting disagrees with packet");
  return result;
}
inline std::vector<at::Device> eager_devices(at::Device first,Index count) {
  if(first.is_cpu()) {
    if(count!=1)throw std::invalid_argument("multiple eager devices require an accelerator");
    return {first};
  }
  c10::impl::VirtualGuardImpl implementation(first.type());
  if(first.index()<0||first.index()+count>implementation.deviceCount())throw std::invalid_argument("eager logical device is unavailable");
  std::vector<at::Device> result;for(Index i=0;i<count;++i)result.emplace_back(first.type(),first.index()+i);
  return result;
}
inline std::string eager_placement_json(const Config& c,const EagerPlacement& p,const std::vector<at::Device>& devices) {
  std::ostringstream out;out<<"{\"devices\":[";
  for(size_t i=0;i<devices.size();++i){if(i)out<<',';out<<quoted(devices[i].str());}
  out<<"],\"node_owners\":[";for(size_t i=0;i<p.owners.size();++i){if(i)out<<',';out<<p.owners[i];}
  out<<"],\"parameter_elements\":[";for(size_t i=0;i<p.elements.size();++i){if(i)out<<',';out<<p.elements[i];}
  out<<"],\"policy\":"<<quoted(c.owner_policy)<<",\"owner_selection\":"<<quoted(c.owner_map.empty()?"automatic":"explicit")
     <<",\"scope\":\"static learned-parameter balance; not total peak-memory admission\"}";
  return out.str();
}
} // namespace tide_flow
