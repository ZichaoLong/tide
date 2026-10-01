#pragma once
#include "device_optimizer.h"
namespace tide::device_online {
// Validate the global identity/storage boundary before splitting groups. Empty
// partitions keep explicit empty groups, never accidentally enable defaults.
void validate_sharded_optimizer_owners(const std::vector<ParameterVjp>&);
std::vector<std::unique_ptr<DeviceOptimizer>> make_sharded_optimizers(
    const std::vector<ParameterVjp>&,DeviceOptimizerKind,std::vector<OptimizerGroup>,int64_t per_device_budget);
} // namespace tide::device_online
