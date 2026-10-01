#include "consumer.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace tide_flow {
Index Packet::parameters() const {
  const long double d=width,n=body.nodes.size(),e=body.edges.size();
  const long double count=e*d*d+n*d+e+body.inputs.size()+2.L*vocab*d+(memory=="attention"?4*n*d*d:0);
  if(count>std::numeric_limits<Index>::max())throw std::invalid_argument("parameter count overflow");
  return Index(count);
}
Packet read_packet(const std::string& path) {
  std::ifstream input(path);Packet p;std::string magic;
  Index n=0,r=0,e=0,ni=0,no=0,clear=0;
  input>>magic>>p.sha>>p.width>>p.batch>>p.tokens>>p.vocab>>p.seed>>p.memory>>p.topology
       >>n>>r>>e>>ni>>no>>p.stride>>p.budget>>clear;
  if(!input||magic!="TIDE_COMPLETE_FLOW_2"||p.sha.size()!=64
     ||p.sha.find_first_not_of("0123456789abcdef")!=std::string::npos
     ||n<1||n>200000||r<1||r>n||e<0||e>10000000||ni<1||ni>n||no<1||no>n
     ||p.width<4||p.width%4||p.width>1000000||p.batch<1||p.batch>512||p.tokens<1||p.tokens>32
     ||p.vocab<2||p.seed<0||p.stride<3||p.stride>std::numeric_limits<Index>::max()/p.tokens
     ||p.budget<1||(clear!=0&&clear!=1)||(p.memory!="add"&&p.memory!="attention")
     ||(p.topology!="ranked-local"&&p.topology!="timed-local"))
    throw std::invalid_argument("invalid continuous v2 packet header");
  p.clear=clear;p.body.nodes.resize(n);p.ranks.resize(r);p.body.inputs.resize(ni);p.body.outputs.resize(no);
  p.body.regions.assign(r,tide::Region{p.budget});
  for(auto& node:p.body.nodes) {
    input>>node.region;node.clear=p.clear;node.memory=p.memory=="add"?"lh-add-repeat-v1":"lh-fiber-attention-all-softmax-repeat-v1";
    node.aggregation=p.memory=="add"?"all_softmax":"sum";node.full="lh-silu-rms-v1";
    node.emission="slot_affine";node.readout="norm-fp32-v1";node.query_heads=node.kv_heads=4;
  }
  for(auto& rank:p.ranks)input>>rank;
  for(auto& v:p.body.inputs)input>>v;
  for(auto& v:p.body.outputs)input>>v;
  p.body.edges.resize(e);for(auto& edge:p.body.edges)input>>edge.source>>edge.target>>edge.delay;
  std::string extra;if(!input||(input>>extra))throw std::invalid_argument("truncated packet or trailing tokens");
  p.body.compile();const auto order=p.body.topological_order();
  std::set<Index> ranks(p.ranks.begin(),p.ranks.end());
  if(ranks.size()!=size_t(r)||*ranks.begin()<1||*ranks.rbegin()>std::numeric_limits<Index>::max()-2)
    throw std::invalid_argument("invalid packet region ranks");
  for(Index region=0;region<r;++region)if(p.body.region_index.offsets[region+1]-p.body.region_index.offsets[region]<p.budget)
    throw std::invalid_argument("packet budget exceeds region");
  std::vector<bool> reached(n),useful(n);std::vector<Index> distance(n);
  for(auto v:p.body.inputs)reached[v]=true;
  for(auto v:p.body.outputs)useful[v]=true;
  for(auto v:order)for(Index i=p.body.csr.offsets[v];i<p.body.csr.offsets[v+1];++i) {
    const auto& edge=p.body.edges[p.body.csr.edges[i]];reached[edge.target]=reached[edge.target]||reached[v];
    if(edge.delay>std::numeric_limits<Index>::max()-distance[v])throw std::invalid_argument("path delay overflow");
    distance[edge.target]=std::max(distance[edge.target],distance[v]+edge.delay);
  }
  for(auto it=order.rbegin();it!=order.rend();++it)for(Index i=p.body.csc.offsets[*it];i<p.body.csc.offsets[*it+1];++i)
    useful[p.body.edges[p.body.csc.edges[i]].source]=useful[p.body.edges[p.body.csc.edges[i]].source]||useful[*it];
  if(std::find(reached.begin(),reached.end(),false)!=reached.end()||std::find(useful.begin(),useful.end(),false)!=useful.end())
    throw std::invalid_argument("packet contains unreachable or unobservable nodes");
  if(p.topology=="ranked-local") {
    tide::SettleGraph check(p.body,p.ranks);
    if(check.stride()!=p.stride)throw std::invalid_argument("ranked packet stride mismatch");
  } else if(*std::max_element(distance.begin(),distance.end())>std::numeric_limits<Index>::max()-3
            ||p.stride!=*std::max_element(distance.begin(),distance.end())+3)
    throw std::invalid_argument("delayed packet stride mismatch");
  p.parameters();return p;
}
} // namespace tide_flow
