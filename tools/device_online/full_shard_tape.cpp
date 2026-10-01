#include "full_shard_tape.h"
#include <map>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
std::vector<at::Tensor*> banks(FullTape& t) {
  return {&t.kinds,&t.weights,&t.biases,&t.extra.lh_kinds,&t.extra.lh_weights,&t.extra.lh_biases,
    &t.extra.swiglu_kinds,&t.extra.swiglu_mapping,&t.extra.gate,&t.extra.up,&t.extra.down};
}
}
int64_t sharded_reverse_tape_bytes(const ShardedReverseTape& source) {
  long double bytes=reverse_tape_bytes(source.coordinator);auto shards=source.shards;
  std::map<const void*,at::Tensor> seen;
  for(auto& s:shards)for(auto* x:banks(s.full))if(x->defined()&&seen.emplace(x->unsafeGetTensorImpl(),*x).second)bytes+=x->nbytes();
  for(const auto& s:source.states)bytes+=state_owner_tape_bytes(s);
  if(bytes>std::numeric_limits<int64_t>::max())throw std::invalid_argument("sharded retained extent overflow");
  return int64_t(bytes);
}
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape& source,int64_t budget) {
  return retain_sharded_reverse_tape(source,budget,nullptr);
}
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape& source,int64_t budget,RetainedProjection* projection) {
  const auto bytes=sharded_reverse_tape_bytes(source)-(projection?
    projection->reusable_bytes(source.coordinator.emission.weights,source.coordinator.emission.biases):0);
  if(budget<1||bytes>budget||source.shards.empty())throw std::invalid_argument("sharded retained tape budget exceeded");
  // Validate every owner before copying any numerical bank.
  auto shards=source.shards;
  for(auto& s:shards) {
    if(!s.full.kinds.defined()||s.full.kinds.device().type()!=c10::DeviceType::PrivateUse1)
      throw std::invalid_argument("invalid Full shard tape owner");
    for(auto* x:banks(s.full))if(x->defined()&&(x->device()!=s.full.kinds.device()||x->requires_grad()))
      throw std::invalid_argument("invalid Full shard retained ownership");
  }
  auto base=retain_reverse_tape(source.coordinator,budget,projection);
  RetainedShardedTape out{base.graph,{base.tape,std::move(shards)},bytes};
  std::map<const void*,at::Tensor> copies;
  for(auto& s:out.tape.shards)for(auto* x:banks(s.full))if(x->defined()) {
    auto& copy=copies[x->unsafeGetTensorImpl()];if(!copy.defined())copy=x->clone();*x=copy;
  }
  for(const auto& s:source.states)out.tape.states.push_back(retain_state_owner_tape(s,budget).tape);
  return out;
}
} // namespace tide::device_online
