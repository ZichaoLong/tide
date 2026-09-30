#include "content_flow_internal.h"
#include "tide/ops.h"
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
ValidatedInput prepare_external(const Graph& g,const Model& m,const Continuation& q,
    const std::vector<External>& external,Index until,at::Device device,const AtomBatch& out) {
  if(external.size()>size_t(out.valid.numel()))
    throw std::invalid_argument("external input buffer capacity exceeded");
  auto sorted=external;
  std::sort(sorted.begin(),sorted.end(),[](const External& a,const External& b){
    return std::tie(a.batch,a.port,a.position)<std::tie(b.batch,b.port,b.position);
  });
  if(q.cut<0||until<q.cut)throw std::invalid_argument("resident window precedes cut");
  ValidatedInput checked;std::vector<Tensor> values;
  for(const auto& x:sorted) {
    if(x.batch<0||x.batch>=q.batch_size||x.port<0||x.port>=Index(g.inputs.size())
        ||x.position<0||x.time<q.cut||x.time>=until)
      throw std::invalid_argument("invalid external coordinate");
    const Owner owner{x.batch,x.port};
    const auto changed=checked.ledger_updates.find(owner);
    const auto stored=q.ledger.find(owner);
    const auto last=changed!=checked.ledger_updates.end()?changed->second:
      stored!=q.ledger.end()?stored->second:Owner{-1,-1};
    if((x.position==0?last.first!=-1:x.position-1!=last.first)||x.time<=last.second)
      throw std::invalid_argument("noncontiguous/nonmonotonic port history");
    const auto& v=x.value;
    if(!v.defined()||v.scalar_type()!=at::kFloat||v.sizes()!=at::IntArrayRef{m.width()}
        ||(!v.device().is_cpu()&&v.device()!=device)
        ||(!values.empty()&&v.device()!=values[0].device()))
      throw std::invalid_argument("resident input requires FP32 [width] on one CPU or session NPU device");
    values.push_back(v);
    checked.ledger_updates[owner]={x.position,x.time};
    checked.atoms.push_back({x.batch,g.inputs[x.port],x.time,0,x.port,x.position,v});
  }
  Tensor packed;
  if(!values.empty()) {
    packed=at::stack(values);
    if(!at::isfinite(packed).all().item<bool>())throw std::invalid_argument("nonfinite resident input");
  }
  // Validation is complete before touching the reusable input buffer. Recursively
  // produced messages do not use this host boundary adapter.
  out.valid.zero_();
  if(!values.empty()) {
    std::vector<int64_t> coordinates;
    for(const auto& a:checked.atoms)
      coordinates.insert(coordinates.end(),{a.batch,a.node,a.time,a.kind,a.source,a.position});
    out.coordinates.narrow(0,0,values.size()).copy_(at::tensor(coordinates,at::kLong).reshape({-1,6}));
    out.values.narrow(0,0,values.size()).copy_(packed);
    out.valid.narrow(0,0,values.size()).fill_(true);
  }
  return checked;
}
} // namespace tide::device_online
