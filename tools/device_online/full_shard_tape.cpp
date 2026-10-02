#include "full_shard_tape.h"
#include "retained_journals.h"
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
  for(const auto& s:source.coordinator.emission.shards)bytes+=RetainedProjection::bytes(s.weights,s.biases);
  if(bytes>std::numeric_limits<int64_t>::max())throw std::invalid_argument("sharded retained extent overflow");
  return int64_t(bytes);
}
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape& source,int64_t budget) {
  return retain_sharded_reverse_tape(source,budget,nullptr);
}
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape& source,int64_t budget,RetainedProjection* projection) {
  return retain_sharded_reverse_tape(source,budget,projection,false);
}
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape& input,int64_t budget,RetainedProjection* projection,bool compact_journals,RetainedAttention* attention) {
  auto source=input;
  if(compact_journals) {
    compact_retained_journals(source.coordinator);
    for(auto& s:source.states)compact_retained_journals(s);
  }
  auto bytes=sharded_reverse_tape_bytes(source)-(projection?
    projection->reusable_bytes(source.coordinator.emission.weights,source.coordinator.emission.biases):0);
  if(attention) {
    bytes-=attention->reusable_bytes(source.coordinator.attention,source.coordinator.fiber);
    for(size_t i=0;i<source.states.size();++i) {
      const auto& s=source.states[i];bytes-=attention->shard(i,source.states.size()).reusable_bytes(s.attention,s.fiber);
    }
  }
  const auto& emissions=source.coordinator.emission.shards;
  for(size_t i=0;i<emissions.size();++i) {
    const auto& s=emissions[i];
    if(!s.weights.defined()||!s.biases.defined()||s.weights.device().type()!=c10::DeviceType::PrivateUse1
        ||s.weights.device()!=s.biases.device()||s.weights.requires_grad()||s.biases.requires_grad())
      throw std::invalid_argument("invalid compact retained projection ownership");
    if(projection)bytes-=projection->shard(i,emissions.size()).reusable_bytes(s.weights,s.biases);
  }
  if(budget<1||bytes>budget||source.shards.empty())throw std::invalid_argument("sharded retained tape budget exceeded");
  // Validate every owner before copying any numerical bank.
  auto shards=source.shards;
  for(auto& s:shards) {
    if(!s.full.kinds.defined()||s.full.kinds.device().type()!=c10::DeviceType::PrivateUse1)
      throw std::invalid_argument("invalid Full shard tape owner");
    for(auto* x:banks(s.full))if(x->defined()&&(x->device()!=s.full.kinds.device()||x->requires_grad()))
      throw std::invalid_argument("invalid Full shard retained ownership");
  }
  auto coordinator=source.coordinator;coordinator.emission.shards.clear();
  auto base=retain_reverse_tape(coordinator,budget,projection,false,attention);
  RetainedShardedTape out{base.graph,{base.tape,std::move(shards)},bytes};
  out.tape.coordinator.emission.shards=emissions;
  std::map<const void*,at::Tensor> copies;
  for(auto& s:out.tape.shards)for(auto* x:banks(s.full))if(x->defined()) {
    auto& copy=copies[x->unsafeGetTensorImpl()];if(!copy.defined())copy=x->clone();*x=copy;
  }
  for(size_t i=0;i<source.states.size();++i)
    out.tape.states.push_back(retain_state_owner_tape(source.states[i],budget,attention?&attention->shard(i,source.states.size()):nullptr).tape);
  for(size_t i=0;i<emissions.size();++i) {
    auto& bank=out.tape.coordinator.emission.shards[i];
    if(projection) {
      auto& snapshot=projection->shard(i,emissions.size());snapshot.capture(bank.weights,bank.biases);
      bank.weights=snapshot.retained()[0];bank.biases=snapshot.retained()[1];
    } else {bank.weights=bank.weights.clone();bank.biases=bank.biases.clone();}
  }
  return out;
}
} // namespace tide::device_online
