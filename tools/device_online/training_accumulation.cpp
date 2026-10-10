#include "training_internal.h"
#include "sharded_training_internal.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
#include <stdexcept>

namespace tide::training_detail {
namespace {
using namespace device_online;
// Each canonical owner already includes all aliases and cross-device uses.
// Accumulate locally on those same devices; never download numerical gradients.
std::vector<ParameterVjp> accumulate_banks(const std::vector<ParameterVjp>& previous,
    const std::vector<ParameterVjp>& current,Index max_bytes,Index program_bytes) {
  if(max_bytes<1||current.empty()||(!previous.empty()&&previous.size()!=current.size()))
    throw std::invalid_argument("invalid gradient accumulation capacity/layout");
  long double bytes=0;
  for(const auto& g:previous)bytes+=g.values.nbytes()+static_cast<long double>(g.connected.nbytes());
  for(const auto& g:current) {
    const long double owners=g.owners.size();
    bytes+=g.values.nbytes()+static_cast<long double>(g.connected.nbytes())+4;
    if(!previous.empty())bytes+=8.L*(std::max(2.L,2*owners)+owners+1);
  }
  if(bytes>max_bytes)throw std::invalid_argument("gradient accumulation tensor budget exceeded");
  std::vector<ParameterVjp> result;
  std::vector<std::unique_ptr<DeviceProgram>> programs;
  std::vector<Tensor> errors;
  // Preflight/build every owner before any execution. The first copy isolates
  // public backward views; later updates consume that private numeric bank.
  // Fresh connection flags avoid cross-tile races. Keep the conservative
  // old/replacement admission bound until the consumer is separately calibrated.
  for(size_t i=0;i<current.size();++i) {
    const auto& g=current[i];const auto device=g.values.device();
    auto error=at::zeros({1},g.values.options().dtype(at::kInt));
    auto p=std::make_unique<DeviceProgram>(device);p->limit_workspace(program_bytes);
    if(previous.empty()) {
      auto out=g;out.values=at::empty_like(g.values);out.connected=at::empty_like(g.connected);
      p->copy(out.values,g.values);p->copy(out.connected,g.connected);result.push_back(std::move(out));
    } else result.push_back(append_private_parameter_accumulate(*p,previous[i],g,error,max_bytes));
    p->finish();errors.push_back(error);programs.push_back(std::move(p));
  }
  for(size_t i=0;i<programs.size();++i) {
    const auto device=current[i].values.device();
    c10::impl::VirtualGuardImpl(device.type()).synchronizeDevice(device.index());
    programs[i]->run();const auto code=errors[i].cpu().item<int>();programs[i]->close();
    if(code)throw std::runtime_error("gradient accumulation refusal code="+std::to_string(code));
  }
  return result;
}
void preflight(bool ready,Index count) {
  if(!ready)throw std::logic_error("accumulate requires a completed, unconsumed backward");
  if(count==std::numeric_limits<Index>::max())throw std::overflow_error("gradient accumulation count exhausted");
}
} // namespace

void ShardedTrainingOwner::accumulate(Index max_bytes) {
  auto& s=*impl_;s.check();preflight(s.gradients_ready,s.accumulated_batches);
  try {
    auto total=accumulate_banks(s.accumulated,s.gradient,max_bytes,s.limits.program_workspace_bytes);
    std::vector<Tensor> present;
    for(const auto& v:s.flow->state_shards_device())present.push_back(v.present.clone());
    s.accumulated=std::move(total);s.initial_present=std::move(present);
    ++s.accumulated_batches;s.gradient.clear();s.gradients_ready=false;
  }catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
} // namespace tide::training_detail

namespace tide {
void ResidentTrainingSession::accumulate(Index max_bytes) {
  auto& s=*impl_;s.check();if(s.sharded){s.sharded->accumulate(max_bytes);return;}
  training_detail::preflight(s.gradients_ready,s.accumulated_batches);
  try {
    auto total=training_detail::accumulate_banks(s.accumulated_batches?
      std::vector<device_online::ParameterVjp>{s.accumulated}:std::vector<device_online::ParameterVjp>{},
      {s.gradient},max_bytes,s.limits.program_workspace_bytes);
    auto present=s.flow->state_device().second.clone();
    s.accumulated=std::move(total.front());s.initial_present=std::move(present);
    ++s.accumulated_batches;s.gradient={};s.gradients_ready=false;
  }catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
} // namespace tide
