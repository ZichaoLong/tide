#include "weighted_training_internal.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
namespace tide::device_online {
ResidentOptimizerKind WeightedTrainingSession::optimizer_kind()const{impl_->check();return impl_->kind;}
std::vector<OptimizerGroup> WeightedTrainingSession::optimizer_groups()const{impl_->check();return impl_->optimizer->groups();}
std::vector<ParameterWrite> WeightedTrainingSession::Impl::writes()const{
  std::vector<WeightedBankRole> roles;
  for(const auto& n:flow->impl_->nodes)for(const auto& [name,role]:n->banks.roles())roles.push_back(role);
  for(const auto& r:flow->impl_->regions)for(const auto& [name,role]:r->banks.roles())roles.push_back(role);
  return weighted_training_writes(flow->impl_->model.parameters(false),layout,roles);
}
void WeightedTrainingSession::Impl::mirror_parameters(){
  const auto values=optimizer->values().cpu();
  for(size_t i=0;i<layout.owners.size();++i){const auto& x=layout.owners[i].value;
    x.copy_(values.narrow(0,layout.offsets[i],x.numel()).reshape(x.sizes()));}
}
void WeightedTrainingSession::accumulate(){
  auto& s=*impl_;s.check();
  if(!s.gradients_ready||!s.saved.empty())throw std::logic_error("weighted accumulate requires completed backward");
  if(s.accumulated_batches==std::numeric_limits<Index>::max())throw std::overflow_error("weighted accumulation count exhausted");
  try{
    auto error=at::zeros({1},s.layout.values.options().dtype(at::kInt));DeviceProgram p(s.device);p.limit_workspace(s.limits.forward.operator_bytes);
    ParameterVjp out;
    if(s.accumulated_batches)out=append_parameter_accumulate(p,s.accumulated,s.gradient,error,s.limits.optimizer_bytes/4);
    else{out=s.gradient;out.values=at::zeros_like(out.values);out.connected=at::zeros_like(out.connected);
      p.copy(out.values,s.gradient.values);p.copy(out.connected,s.gradient.connected);}
    s.run(p,error);s.accumulated=std::move(out);++s.accumulated_batches;s.discard(false);
  }catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
ResidentStep WeightedTrainingSession::step(){
  auto& s=*impl_;s.check();
  if(!s.saved.empty()||(!s.gradients_ready&&!s.accumulated_batches))throw std::logic_error("weighted step requires completed backward");
  if(s.gradients_ready&&s.accumulated_batches)throw std::logic_error("accumulate the final weighted backward before step");
  if(s.generation==std::numeric_limits<Index>::max()||s.flow->impl_->generation==std::numeric_limits<Index>::max())
    throw std::overflow_error("weighted parameter generation exhausted");
  try{
    auto error=at::zeros({1},s.layout.values.options().dtype(at::kInt));DeviceProgram p(s.device);p.limit_workspace(s.limits.forward.operator_bytes);
    s.optimizer->append_step(p,s.accumulated_batches?s.accumulated:s.gradient,error);
    append_owner_parameter_publish(p,s.writes(),s.optimizer->values(),error,s.limits.optimizer_bytes/8);
    p.finish();c10::impl::VirtualGuardImpl(s.device.type()).synchronizeDevice(s.device.index());p.run(s.limits.forward.timeout_ms);
    const auto code=error.cpu().item<int>();p.close();
    if(code==20||code==21)return {false,code,s.generation}; // Masters, slots and role banks unchanged.
    if(code)throw std::runtime_error("weighted optimizer refused, error="+std::to_string(code));
    s.mirror_parameters();++s.generation;++s.flow->impl_->generation;s.flow->impl_->ran=false;s.discard();return {true,0,s.generation};
  }catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
ResidentTrainingCheckpoint WeightedTrainingSession::checkpoint()const{
  const auto& s=*impl_;s.check();
  if(!s.saved.empty()||s.gradients_ready||s.accumulated_batches)throw std::logic_error("weighted checkpoint requires detach or completed update");
  ResidentTrainingCheckpoint out;out.mode="weighted";out.generation=s.generation;out.next_token=s.next_token;
  out.continuation=s.flow->checkpoint();out.optimizer=s.kind;out.groups=s.optimizer->groups();out.state=s.optimizer->snapshot();out.offsets=s.layout.offsets;
  const auto all=s.flow->impl_->model.parameters(false);out.aliases=all.alias_partitions();out.trainable=s.registry.names();
  for(const auto& owner:all.owners())out.parameters[owner.canonical]=owner.value.detach().clone();return out;
}
}
