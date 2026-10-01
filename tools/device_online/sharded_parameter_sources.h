#pragma once
#include "sharded_full_vjp.h"
#include "tide/parameters.h"

namespace tide::device_online {
// Static slices into completed device partials. A false connection is None;
// consumers must not evaluate its numerical payload. No gradients are read by
// this host layout planner, nor are CPU reference results accepted.
struct ParameterContribution {at::Tensor values,connected;};
struct ShardedParameterSources {
  std::vector<ParameterOwner> owners;
  std::vector<std::vector<ParameterContribution>> contributions;
};
// Contributions retain reverse-window order, then registry alias order. Aliases
// spanning Full owners and coordinator state/Read remain one canonical owner.
ShardedParameterSources sharded_parameter_sources(const Graph&,const ParameterRegistry&,
    const std::vector<ShardedGraphVjp>&,int64_t metadata_budget);
// Deterministic LPT balance of FP32 canonical owner storage. Owners without a
// differentiable use remain in the registry but need no gradient/master slot.
std::vector<int64_t> place_parameter_owners(const ShardedParameterSources&,int64_t devices);
} // namespace tide::device_online
