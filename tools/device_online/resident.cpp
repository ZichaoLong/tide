#include "tide/resident.h"
#include "tide/parameters.h"
#include "content_flow.h"
#include "sharded_state.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide {
namespace {
Tensor cpu(const Tensor& x) {return x.detach().to(at::kCPU).clone();}
Model freeze(Model model) {
  std::map<const c10::TensorImpl*,Tensor> copies;
  auto frozen=[&](const Tensor& value) {
    auto& copy=copies[value.unsafeGetTensorImpl()];
    if(!copy.defined())copy=cpu(value);
    return copy;
  };
  // ContentProfile validates declarations, and rejects custom handles before
  // configuring the independent built-ins. Do not strip arbitrary programs.
  for(auto& w:model.nodes) {
    w.decay=frozen(w.decay);w.weight=frozen(w.weight);w.bias=frozen(w.bias);w.read=frozen(w.read);
    for(auto& [_,v]:w.extra)v=frozen(v);
  }
  for(auto& w:model.regions)for(auto& [_,v]:w.extra)v=frozen(v);
  for(auto group:{&model.input_scale,&model.agg_scale,&model.edge_scale,&model.output_scale})
    for(auto& value:*group)value=frozen(value);
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
  ResidentPlacement placement;
  Index cut;
  Impl(Graph graph,Model model,const Continuation& q,at::Device device,ResidentLimits limits,ResidentPlacement p)
      :placement(std::move(p)),cut(q.cut) {
    if(at::GradMode::is_enabled())throw std::invalid_argument("resident inference requires explicit no-grad");
    if(device.type()!=c10::DeviceType::PrivateUse1||device.index()<0)
      throw std::invalid_argument("resident inference requires an explicit logical NPU index");
    for(const auto& owner:model.parameters(false).owners()) {
      const auto& v=owner.value;
      if((v.scalar_type()!=at::kFloat&&v.scalar_type()!=at::kHalf)||(!v.device().is_cpu()&&v.device()!=device))
        throw std::invalid_argument("resident parameters require FP32/FP16 on CPU or the session NPU");
      parameters.push_back({v,v._version(),v.const_data_ptr()});
    }
    if(placement.policy!="memory"&&placement.policy!="locality")
      throw std::invalid_argument("resident placement policy must be memory or locality");
    if(placement.devices.empty()) {
      if(!placement.full_owners.empty()||!placement.state_owners.empty())
        throw std::invalid_argument("resident owner maps require an explicit device list");
      placement.devices={device};placement.full_owners.assign(graph.nodes.size(),0);
      placement.state_owners=placement.full_owners;
      flow=std::make_unique<device_online::ContentFlow>(std::move(graph),freeze(model),freeze(q),device,limits);
    } else {
      if(placement.devices.front()!=device||placement.devices.size()>16)
        throw std::invalid_argument("resident devices must start with the explicit coordinator");
      auto full=device_online::place_full(graph,model,placement.devices,placement.policy),state=full;
      if(!placement.full_owners.empty())full.owners=placement.full_owners;
      if(!placement.state_owners.empty())state.owners=placement.state_owners;
      device_online::validate_full_placement(full,graph.nodes.size(),device);
      device_online::validate_full_placement(state,graph.nodes.size(),device);
      placement.full_owners=full.owners;placement.state_owners=state.owners;
      flow=std::make_unique<device_online::ContentFlow>(std::move(graph),freeze(model),freeze(q),device,limits,
          device_online::ModelPlacement{std::move(full),std::move(state)});
    }
  }
  void check() const {
    if(!flow)throw std::logic_error("resident session is closed");
    for(const auto& p:parameters)
      if(p.value._version()!=p.version||p.value.const_data_ptr()!=p.data)
        throw std::logic_error("resident parameters changed; create a new session from an explicit complete cut");
  }
};
ResidentSession::ResidentSession(Graph g,Model m,const Continuation& q,at::Device d,ResidentLimits l)
    :ResidentSession(std::move(g),std::move(m),q,d,l,{}) {}
ResidentSession::ResidentSession(Graph g,Model m,const Continuation& q,at::Device d,ResidentLimits l,ResidentPlacement p)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,l,std::move(p))) {}
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
ResidentContinuation ResidentSession::snapshot_device(Index max_bytes) const {
  impl_->check();return impl_->flow->snapshot_device(max_bytes);
}
void ResidentSession::restore_device(const ResidentContinuation& saved) {
  impl_->check();impl_->flow->restore_device(saved);impl_->cut=saved.cut();
}
Result ResidentSession::result() const {
  if(!impl_->flow)throw std::logic_error("resident session is closed");
  return impl_->flow->result();
}
Index ResidentSession::cut() const {return impl_->cut;}
ResidentPlacement ResidentSession::placement() const {return impl_->placement;}
void ResidentSession::close() {
  if(impl_->flow){impl_->flow->close();impl_->flow.reset();}
}
} // namespace tide
