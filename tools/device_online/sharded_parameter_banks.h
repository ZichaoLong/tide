#pragma once
#include "parameter_publish.h"
#include "full_shard_tape.h"
namespace tide::device_online {
struct ShardedParameterBanks {ParameterBanks coordinator;std::vector<FullShardTape> full;std::vector<StateOwnerBanks> states;};
struct ParameterDestination {at::Tensor values;at::ScalarType payload_dtype;};
// Static named tensor views, including strided Q/K/V columns and HARD Read.
// Construction reads shape/placement only; no master returns through the CPU.
std::map<std::string,ParameterDestination> sharded_parameter_destinations(const ShardedParameterBanks&);
struct ParameterWrite {int64_t offset;ParameterDestination destination;};
void append_owner_parameter_publish(DeviceProgram&,const std::vector<ParameterWrite>&,
    const at::Tensor& masters,const at::Tensor& error,int64_t metadata_budget);
} // namespace tide::device_online
