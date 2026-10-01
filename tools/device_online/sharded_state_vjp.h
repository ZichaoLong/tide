#pragma once
#include "full_shard_tape.h"
#include "state_owner_reverse.h"
namespace tide::device_online {
class ShardedStateVjp {
 public:
  ShardedStateVjp(const ShardedReverseTape&,const at::Tensor& error,int64_t chunk,int64_t budget,int64_t workspace,
      const std::shared_ptr<ShardedStateVjp>& next={},
      const std::vector<std::vector<CacheCotangents>>& roots={});
  ~ShardedStateVjp();
  void prepare(CannProgram&,const ReverseLinks&);
  StateVjp append_stage(CannProgram&,const at::Tensor& range,const StateCotangents&,const ControlScores&);
  void append_sources(CannProgram&,const at::Tensor& messages,const at::Tensor& connected,const at::Tensor& partials);
  void append_stop(CannProgram&);
  void synchronize_inputs() const;void submit();void wait();void close();
  std::vector<StateShardGradient> gradients() const;
  int64_t packet_bytes() const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
