#include "training_internal.h"
#include "sharded_training_internal.h"
#include <stdexcept>

namespace tide {
namespace {
void detached(bool outstanding,bool ready) {
  if(outstanding||ready)throw std::logic_error("device continuation requires detach, accumulate or step first");
}
}
ResidentContinuation ResidentTrainingSession::snapshot_device(Index max_bytes) const {
  return snapshot_device(max_bytes,false);
}
ResidentContinuation ResidentTrainingSession::snapshot_device(Index max_bytes,bool compact) const {
  auto& s=*impl_;s.check();if(s.sharded)return s.sharded->snapshot_device(max_bytes,compact);
  detached(!s.saved.empty(),s.gradients_ready);
  try{return s.flow->snapshot_device(max_bytes,compact);}
  catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
void ResidentTrainingSession::restore_device(const ResidentContinuation& saved) {
  auto& s=*impl_;s.check();if(s.sharded){s.sharded->restore_device(saved);return;}
  detached(!s.saved.empty(),s.gradients_ready);
  try {
    s.flow->restore_device(saved);s.cut=saved.cut();
    s.initial_present=s.flow->state_device().second.clone();
  }catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
namespace training_detail {
ResidentContinuation ShardedTrainingOwner::snapshot_device(Index max_bytes) const {
  return snapshot_device(max_bytes,false);
}
ResidentContinuation ShardedTrainingOwner::snapshot_device(Index max_bytes,bool compact) const {
  auto& s=*impl_;s.check();detached(!s.saved.empty(),s.gradients_ready);
  try{return s.flow->snapshot_device(max_bytes,compact);}
  catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
void ShardedTrainingOwner::restore_device(const ResidentContinuation& saved) {
  auto& s=*impl_;s.check();detached(!s.saved.empty(),s.gradients_ready);
  try {
    s.flow->restore_device(saved);s.cut=saved.cut();s.initial_present.clear();
    for(const auto& v:s.flow->state_shards_device())s.initial_present.push_back(v.present.clone());
  }catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
} // namespace training_detail
} // namespace tide
