#include "training_internal.h"
#include <stdexcept>

namespace tide {
ResidentTrainingCheckpoint ResidentTrainingSession::checkpoint() const {
  const auto& s=*impl_;s.check();
  if(!s.saved.empty()||s.gradients_ready)throw std::logic_error("checkpoint requires explicit detach or a completed optimizer step");
  ResidentTrainingCheckpoint out;
  out.generation=s.generation;out.next_token=s.next_token;out.continuation=s.flow->snapshot();
  out.aliases=s.model.parameters(false).alias_partitions();out.trainable=s.registry.names();
  out.optimizer=s.kind;out.groups=s.optimizer->groups();out.offsets=s.layout.offsets;out.state=s.optimizer->snapshot();
  out.mode=s.limits.forward.mode;out.zeta=s.limits.forward.zeta;
  for(const auto& owner:s.model.parameters(false).owners())out.parameters.emplace(owner.canonical,owner.value.detach().clone());
  // Only fixed unused owners retain their initial CPU value. Every optimizer
  // owner with a differentiable use is read from the current device parameter.
  for(size_t i=0;i<s.layout.owners.size();++i)if(s.layout.offsets[i]>=0) {
    const auto& owner=s.layout.owners[i];
    out.parameters.at(owner.canonical)=out.state.values.narrow(0,s.layout.offsets[i],owner.value.numel()).reshape(owner.value.sizes()).clone();
  }
  return out;
}
} // namespace tide
