// Complete-cut composition: PDG finite-cut note and resident-weighted-retained.md.
#include "weighted_window_bridge.h"
namespace tide::device_online {
void append_weighted_window_bridge(DeviceProgram& p,const WeightedWindowSnapshot& a,
    WeightedWindowRoots& roots,const WeightedWindowSnapshot& b,const WeightedWindowAdjoint& later,
    const at::Tensor& error,int64_t budget){
  a.validate();b.validate();
  if(a.graph.identity!=b.graph.identity||a.device!=b.device||a.samples!=b.samples
      ||a.layouts.size()!=b.layouts.size()||roots.nodes.size()!=a.layouts.size()
      ||later.states().size()!=b.layouts.size()||a.boundary.cut!=b.start||budget<1)
    throw std::invalid_argument("incompatible retained weighted window bridge");
  const auto& boundary=later.boundary();const auto capacity=b.limits.queue;
  const auto width=a.model.width(),nodes=int64_t(a.layouts.size());
  const auto options=error.options();const auto before=p.retained_tensor_bytes();
  const long double required=static_cast<long double>(nodes)*a.samples*(13.L+4.L*width)+16.L*capacity+1024;
  if(required>budget)throw std::invalid_argument("retained weighted bridge tensor budget exceeded");
  for(size_t v=0;v<a.layouts.size();++v){
    if(a.layouts[v].floating_width()!=b.layouts[v].floating_width()||a.layouts[v].capacity()!=b.layouts[v].capacity())
      throw std::invalid_argument("retained weighted raw boundary layout changed");
    weighted_reverse::add(p,a.layouts[v],roots.nodes[v].final,later.states()[v],error);
  }
  WeightedReverseEvents empty{at::zeros({1,nodes,a.samples},options.dtype(at::kLong)),at::zeros({1,nodes,a.samples},options.dtype(at::kBool)),
    at::zeros({1,nodes,a.samples,width},options.dtype(at::kFloat)),at::zeros({1,nodes,a.samples},options.dtype(at::kBool))};
  std::vector<int64_t> sources;for(const auto& e:a.graph.edges)sources.push_back(e.source);
  auto edges=at::tensor(sources,at::kLong).to(a.device),stage=at::zeros({1},options.dtype(at::kLong));
  AtomBatch left{boundary.coordinates.narrow(0,0,capacity),boundary.values.narrow(0,0,capacity),boundary.present.narrow(0,0,capacity)};
  WeightedReverseBoundary right{a.pending.coordinates,a.pending.valid,roots.pending.value,roots.pending.connected};
  weighted_window::route(p,left,{boundary.values.narrow(0,0,capacity),boundary.connected.narrow(0,0,capacity)},
    edges,empty,right,stage,b.journal->boundary("begin"),error);
  if(p.retained_tensor_bytes()-before>budget)throw std::invalid_argument("retained weighted bridge tensor budget exceeded");
}
ParameterVjp bind_weighted_window_parameters(const ParameterVjp& partial,const ParameterRegistry& registry){
  const auto& owners=registry.owners();
  if(partial.owners.size()!=owners.size())throw std::invalid_argument("saved parameter owner count changed");
  for(size_t i=0;i<owners.size();++i){const auto& a=partial.owners[i];const auto& b=owners[i];
    if(a.canonical!=b.canonical||a.aliases!=b.aliases||a.value.sizes()!=b.value.sizes()||a.value.scalar_type()!=b.value.scalar_type())
      throw std::invalid_argument("saved parameter alias/shape/dtype changed");
  }
  auto out=partial;out.owners=owners;return out;
}
}
