#include "device_backend.h"
#include "full_placement.h"
#include "packed_lh_full.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>

namespace tide::device_online {
void validate_full_placement(const FullPlacement& p,int64_t nodes,at::Device coordinator) {
  if(nodes<1||p.devices.empty()||p.devices.size()>size_t(nodes)||p.owners.size()!=size_t(nodes))
    throw std::invalid_argument("Full placement requires one owner per node and nonempty shards");
  std::set<int> indices;std::vector<int64_t> members(p.devices.size());
  for(const auto d:p.devices)
    if(d.type()!=tide::device_online::resident_device_type||d.type()!=coordinator.type()||d.index()<0||!indices.insert(d.index()).second)
      throw std::invalid_argument("Full placement requires distinct explicit logical resident devices");
  for(const auto owner:p.owners) {
    if(owner<0||owner>=int64_t(members.size()))throw std::invalid_argument("invalid Full shard owner");
    ++members[owner];
  }
  if(std::find(members.begin(),members.end(),0)!=members.end())throw std::invalid_argument("empty Full shard");
}
FullPlacement place_full(const Graph& graph,const Model& model,std::vector<at::Device> devices,const std::string& policy) {
  const int64_t nodes=graph.nodes.size(),count=devices.size();
  if(nodes<1||count<1||count>nodes||model.nodes.size()!=size_t(nodes)||(policy!="memory"&&policy!="locality"))
    throw std::invalid_argument("invalid Full partition request");
  std::vector<int64_t> bytes(nodes,1),order(nodes),loads(count),members(count);
  int64_t total=0;
  for(int64_t n=0;n<nodes;++n) {
    const auto& node=graph.nodes[n];const auto& w=model.nodes[n];long double size=1;
    if(!node.identity) {
      if(node.full=="tanh")size+=w.weight.nbytes()+w.bias.nbytes();
      else if(node.full=="swiglu")for(const auto name:{"ffn_gate","ffn_up","ffn_down"})size+=w.extra.at(name).nbytes();
      else if(const auto kind=lh_full_kind(node.full);kind) {
        if((kind-1)%3)size+=w.extra.at("lh_norm_weight").nbytes();
        if((kind-1)%3==2)size+=w.extra.at("lh_norm_bias").nbytes();
      }
      if(node.emission=="slot_affine")for(int64_t j=0;j<graph.outgoing_ports.offsets.at(n+1)-graph.outgoing_ports.offsets.at(n);++j)
        size+=w.extra.at("emit_w_"+std::to_string(j)).nbytes()+w.extra.at("emit_b_"+std::to_string(j)).nbytes();
    }
    if(size>std::numeric_limits<int64_t>::max()-total)throw std::overflow_error("Full placement size overflow");
    total+=bytes[n]=int64_t(size);
  }
  FullPlacement out{std::move(devices),std::vector<int64_t>(nodes)};
  std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return bytes[a]>bytes[b];});
  for(const auto n:order) {
    const auto shard=std::min_element(loads.begin(),loads.end())-loads.begin();
    out.owners[n]=shard;loads[shard]+=bytes[n];++members[shard];
  }
  // A bounded generic refinement. Physical parallel edges contribute separately;
  // stable strict improvements never exceed the initial peak parameter load.
  if(policy=="locality") {
    std::vector<std::vector<int64_t>> adjacent(nodes);
    for(const auto& e:graph.edges)if(e.source!=e.target) {
      adjacent.at(e.source).push_back(e.target);adjacent.at(e.target).push_back(e.source);
    }
    const auto limit=*std::max_element(loads.begin(),loads.end());
    for(int pass=0;pass<2;++pass)for(const auto n:order) {
      const auto from=out.owners[n];if(members[from]<=1)continue;
      std::vector<int64_t> affinity(count);
      for(const auto other:adjacent[n])++affinity[out.owners[other]];
      auto to=from;
      for(int64_t s=0;s<count;++s)
        if(loads[s]<=limit-bytes[n]&&affinity[s]>affinity[to])to=s;
      if(to!=from){loads[from]-=bytes[n];loads[to]+=bytes[n];--members[from];++members[to];out.owners[n]=to;}
    }
    // Equal-size nodes often cannot move under a balanced memory cap. Swaps
    // retain that cap and accept only a strict physical-edge cut reduction.
    for(int pass=0;pass<2;++pass)for(const auto n:order) {
      const auto from=out.owners[n];std::vector<int64_t> affinity(count);
      for(const auto other:adjacent[n])++affinity[out.owners[other]];
      int64_t best=-1,gain=0;
      for(int64_t m=0;m<nodes;++m) {
        const auto to=out.owners[m];if(to==from)continue;
        if(loads[from]-bytes[n]>limit-bytes[m]||loads[to]-bytes[m]>limit-bytes[n])continue;
        int64_t back=0,stay=0,shared=0;
        for(const auto other:adjacent[m]){back+=out.owners[other]==from;stay+=out.owners[other]==to;shared+=other==n;}
        const auto delta=affinity[to]-affinity[from]+back-stay-2*shared;
        if(delta>gain){gain=delta;best=m;}
      }
      if(best>=0) {
        const auto to=out.owners[best];loads[from]+=bytes[best]-bytes[n];loads[to]+=bytes[n]-bytes[best];
        std::swap(out.owners[n],out.owners[best]);
      }
    }
  }
  validate_full_placement(out,nodes,out.devices.front());return out;
}
} // namespace tide::device_online
