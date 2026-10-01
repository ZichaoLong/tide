#pragma once
#include "projection_shard.h"
namespace tide::device_online {
// Actual unscaled selected slot values and immutable projection parameters.
// Rows preserve physical local-slot identity, including parallel edges/phases.
struct EmissionTape {at::Tensor metadata,values,count,weights,biases;std::vector<ProjectionBank> shards;};
struct EmissionVjp {
  at::Tensor weights,biases,connected,chunks;
  std::vector<ProjectionGradient> shards;
  std::shared_ptr<ProjectionStage> program;
};
} // namespace tide::device_online
