#pragma once
#include "sharded_parameter_sources.h"
#include "device_optimizer.h"
#include "sharded_parameter_banks.h"
namespace tide::device_online {
// Canonical owners are distinct from forward Full-node placement. This owner
// consumes completed candidate adjoints, packs contributions per device pair
// and reduces each alias exactly once on the assigned canonical owner device.
class ShardedParameterReduce {
 public:
  ShardedParameterReduce(ShardedParameterSources,std::vector<at::Device>,const at::Tensor& upstream_error,
                        int64_t tensor_budget,int64_t per_program_workspace);
  // Completed canonical gradients: build an update/publication program without
  // repeating alias reduction or copying masters back to the coordinator.
  ShardedParameterReduce(std::vector<ParameterVjp>,const at::Tensor& upstream_error,
                        int64_t tensor_budget,int64_t per_program_workspace);
  ~ShardedParameterReduce();
  const std::vector<ParameterVjp>& gradients() const;
  const std::vector<at::Tensor>& errors() const;
  void append_step(const std::vector<DeviceOptimizer*>&);
  // Append after the common commit decision. Masters stay on canonical cards;
  // each receiver gets only its used owners, then publishes every local alias.
  void append_publish(const ShardedParameterBanks&,const std::vector<DeviceOptimizer*>&,int64_t tensor_budget);
  void finish();
  void run();
  void close();
  int64_t packet_bytes() const;
  int64_t stream_reserved_bytes() const; // Both endpoints + metadata, not allocator peak.
  int64_t stream_chunks() const; // Planned packet iterations, not numerical activity.
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
