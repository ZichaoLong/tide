#pragma once
// Canonical parameter identities: architecture §2.2 and resident-weighted-parameters.md.
#include "weighted_window_bridge.h"
#include "sharded_parameter_banks.h"
namespace tide::device_online {
// A stable complete trainable-owner layout, including currently disconnected
// owners. Their connection bits, not their offsets, control optimizer steps.
ParameterVjp weighted_training_layout(const ParameterRegistry&,at::Device,int64_t byte_budget);
ParameterVjp append_weighted_training_parameters(DeviceProgram&,const ParameterVjp&,
  const ParameterVjp& layout,int64_t byte_budget);
std::vector<ParameterWrite> weighted_training_writes(const ParameterRegistry& all,
  const ParameterVjp&,const std::vector<WeightedBankRole>&);
}
