#include "tide/resident.h"
#include "tide/parameters.h"
#include "content_flow.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide {
namespace {
Tensor cpu(const Tensor& x) {return x.detach().to(at::kCPU).clone();}
Model freeze(Model model) {
  // ContentProfile validates declarations, and rejects custom handles before
  // configuring the independent built-ins. Do not strip arbitrary programs.
  for(auto& w:model.nodes) {
    w.decay=cpu(w.decay);w.weight=cpu(w.weight);w.bias=cpu(w.bias);w.read=cpu(w.read);
    for(auto& [_,v]:w.extra)v=cpu(v);
  }
  for(auto& w:model.regions)for(auto& [_,v]:w.extra)v=cpu(v);
  for(auto group:{&model.input_scale,&model.agg_scale,&model.edge_scale,&model.output_scale})
    for(auto& value:*group)value=cpu(value);
  return model;
}
Continuation freeze(Continuation q) {
  for(auto& [_,s]:q.states){s.value=cpu(s.value);for(auto& [name,v]:s.slots)v=cpu(v);}
  for(auto& [_,h]:q.history)for(auto& [name,v]:h.tensors)v=cpu(v);
  for(auto& a:q.pending)a.value=cpu(a.value);
  return q;
}
}
struct ResidentSession::Impl {
  struct Version {Tensor value;int64_t version;const void* data;};
  std::vector<Version> parameters;
  std::unique_ptr<device_online::ContentFlow> flow;
  Index cut;
  Impl(Graph graph,Model model,const Continuation& q,at::Device device,ResidentLimits limits):cut(q.cut) {
    if(at::GradMode::is_enabled())throw std::invalid_argument("resident inference requires explicit no-grad");
    if(device.type()!=c10::DeviceType::PrivateUse1||device.index()<0)
      throw std::invalid_argument("resident inference requires an explicit logical NPU index");
    for(const auto& owner:model.parameters(false).owners()) {
      const auto& v=owner.value;
      if(v.scalar_type()!=at::kFloat||(!v.device().is_cpu()&&v.device()!=device))
        throw std::invalid_argument("resident parameters require FP32 on CPU or the session NPU");
      parameters.push_back({v,v._version(),v.const_data_ptr()});
    }
    flow=std::make_unique<device_online::ContentFlow>(std::move(graph),freeze(model),freeze(q),device,limits);
  }
  void check() const {
    if(!flow)throw std::logic_error("resident session is closed");
    for(const auto& p:parameters)
      if(p.value._version()!=p.version||p.value.const_data_ptr()!=p.data)
        throw std::logic_error("resident parameters changed; create a new session from an explicit complete cut");
  }
};
ResidentSession::ResidentSession(Graph g,Model m,const Continuation& q,at::Device d,ResidentLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,l)) {}
ResidentSession::~ResidentSession()=default;
ResidentWindow ResidentSession::advance(const std::vector<External>& input,Index stop,Index seal) {
  impl_->check();
  if(seal<stop)throw std::invalid_argument("resident window is unsealed");
  const auto window=impl_->flow->advance_device(input,stop);impl_->cut=stop;
  return {window.outputs.coordinates,window.outputs.values,window.outputs.valid,
          window.output_stats,window.pending_stats,window.stages,window.events,window.full_chunks,window.emission_chunks};
}
Continuation ResidentSession::snapshot() const {
  if(!impl_->flow)throw std::logic_error("resident session is closed");
  return impl_->flow->snapshot();
}
Result ResidentSession::result() const {
  if(!impl_->flow)throw std::logic_error("resident session is closed");
  return impl_->flow->result();
}
Index ResidentSession::cut() const {return impl_->cut;}
void ResidentSession::close() {
  if(impl_->flow){impl_->flow->close();impl_->flow.reset();}
}
} // namespace tide
