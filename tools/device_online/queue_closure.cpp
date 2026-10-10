#include "device_backend.h"
#include "queue_closure.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>

namespace tide::device_online {
QueueClosure::QueueClosure(const std::vector<int64_t>& owners, int64_t regions,
    const std::vector<Wire>& wires,int64_t samples,at::Device device,int64_t budget)
    : regions_(regions),samples_(samples),nodes_(owners.size()) {
  constexpr auto inf=std::numeric_limits<int64_t>::max();
  if(regions<1 || samples<1 || nodes_<1 || budget<1 || regions>inf/regions/32
      || samples>inf/regions/32) throw std::invalid_argument("invalid closure dimensions/budget");
  for(auto r:owners) if(r<0 || r>=regions) throw std::invalid_argument("invalid region owner");
  // Explicit approximate temporary budget for closure, separate from queue,
  // model and autograd memory. Chunk samples without cutting a logical fiber.
  int64_t fixed=0;
  if(nodes_>inf/8)throw std::invalid_argument("closure node metadata size overflow");
  for(auto part:{regions*regions*8,samples*regions*24,nodes_*8}) {
    if(part>budget-fixed)throw std::invalid_argument("closure persistent metadata exceeds budget");
    fixed+=part;
  }
  if(budget<=fixed || (budget-fixed)/(regions*regions*24)<1)
    throw std::invalid_argument("closure workspace budget cannot hold one sample");
  sample_chunk_=std::min(samples,(budget-fixed)/(regions*regions*24));
  std::vector<std::vector<std::pair<int64_t,int64_t>>> outgoing(regions);
  for(auto [s,t,d]:wires) {
    if(s<0 || t<0 || s>=nodes_ || t>=nodes_ || d<1) throw std::invalid_argument("invalid closure wire");
    outgoing[owners[s]].push_back({owners[t],d});
  }
  std::vector<int64_t> distances(regions*regions,inf);
  for(int64_t source=0;source<regions;++source) {
    // Nonempty paths, including a positive self-return distance on feedback.
    auto begin=distances.begin()+source*regions;
    using Entry=std::pair<int64_t,int64_t>;
    std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> todo;
    for(auto [r,d]:outgoing[source]) if(d<begin[r]) {begin[r]=d;todo.push({d,r});}
    while(!todo.empty()) {
      auto [d,r]=todo.top();todo.pop();if(d!=begin[r])continue;
      for(auto [target,delay]:outgoing[r]) {
        auto next=delay>=inf-d?inf:d+delay;
        if(next<begin[target]) {begin[target]=next;todo.push({next,target});}
      }
    }
  }
  owner_=at::tensor(owners,at::kLong).to(device);
  distance_=at::tensor(distances,at::kLong).reshape({regions,regions}).to(device);
}
at::Tensor QueueClosure::valid_coordinates(const AtomBatch& q) const {
  const auto& c=q.coordinates;
  return ((c>=0).all(1)&(c.select(1,0)<samples_)&(c.select(1,1)<nodes_)
          &(c.select(1,3)<=1))|~q.valid;
}
at::Tensor QueueClosure::ready(const AtomBatch& q,const at::Tensor& stop,bool prefill) const {
  if(owner_.device().type()==tide::device_online::resident_device_type)
    throw std::invalid_argument("tensor closure uses unsupported NPU scatter_reduce; use explicit Ascend C closure");
  if(stop.sizes()!=at::IntArrayRef{1} || stop.scalar_type()!=at::kLong || stop.device()!=owner_.device())
    throw std::invalid_argument("closure stop must be device int64 [1]");
  const auto& c=q.coordinates;
  auto valid=valid_coordinates(q)&q.valid;
  auto batch=c.select(1,0).clamp(0,samples_-1);
  auto region=owner_.index_select(0,c.select(1,1).clamp(0,nodes_-1));
  auto time=at::where(valid,c.select(1,2),stop);
  auto pending=valid&(time<stop);
  auto seeds=stop.expand({samples_*regions_}).clone();
  seeds.scatter_reduce_(0,batch*regions_+region,at::where(pending,time,stop),"amin",true);
  seeds=seeds.reshape({samples_,regions_});
  at::Tensor cutoff;
  if(prefill) {
    cutoff=at::empty_like(seeds);
    for(int64_t first=0;first<samples_;first+=sample_chunk_) {
      auto last=std::min(samples_,first+sample_chunk_);
      auto start=seeds.slice(0,first,last).unsqueeze(2);
      // Saturate BEFORE addition: neither overflow nor FP64 rounding of clocks.
      auto arrival=start+at::minimum(distance_.unsqueeze(0),stop-start);
      cutoff.slice(0,first,last).copy_(std::get<0>(arrival.min(1)));
    }
    return pending & (time<cutoff.reshape({-1}).index_select(0,batch*regions_+region));
  }
  auto earliest=std::get<0>(seeds.min(1));
  return pending & (time==earliest.index_select(0,batch));
}
}  // namespace tide::device_online
