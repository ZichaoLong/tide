#pragma once
#include "device_sequence.h"
#include "full_shard_tape.h"
namespace tide::device_online {
struct FullShardGradient {std::vector<int64_t> nodes;FullVjp values;};
// One service per remote Full owner. Parameter partials accumulate where their
// actual forward bank lives; only stage cotangents/connection bits return.
class ShardedFullVjp {
 public:
  ShardedFullVjp(DeviceProgram&,const ShardedReverseTape&,int64_t chunk_rows,
                int64_t tensor_budget,int64_t per_program_workspace);
  ~ShardedFullVjp();
  FullVjp append_stage(DeviceProgram&,const FullTape&,const at::Tensor& gradient,
                       const at::Tensor& connected,const at::Tensor& error);
  void append_stop(DeviceProgram&);
  void synchronize_inputs() const;
  void submit();
  void wait();
  void close(); // Close coordinator first, after all waits.
  std::vector<FullShardGradient> gradients() const;
  std::vector<at::Tensor> work() const;
  int64_t packet_bytes() const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
class ShardedStateVjp;
struct ShardedGraphVjp {GraphVjp coordinator;std::shared_ptr<ShardedFullVjp> full;std::shared_ptr<ShardedStateVjp> state;};
ShardedGraphVjp append_sharded_graph_vjp(DeviceProgram&,const ShardedReverseTape&,
    const GraphCotangents&,const at::Tensor& error,int64_t chunk_rows,
    int64_t tensor_budget,int64_t per_program_workspace);
ShardedGraphVjp append_sharded_graph_vjp(DeviceProgram&,const ShardedReverseTape&,
    const GraphCotangents&,const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget,int64_t per_program_workspace,
    const std::shared_ptr<ShardedStateVjp>& next_state,const std::vector<std::vector<CacheCotangents>>& state_roots);
ShardedGraphVjp append_sharded_graph_vjp(DeviceProgram&,const ShardedReverseTape&,
    const GraphCotangents&,const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget,int64_t per_program_workspace,
    const std::shared_ptr<ShardedStateVjp>& next_state,const std::vector<std::vector<CacheCotangents>>& state_roots,
    const std::vector<ProjectionGradient>& reuse);
ShardedGraphVjp append_sharded_graph_vjp(DeviceProgram&,const ShardedReverseTape&,
    const GraphCotangents&,const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget,int64_t per_program_workspace,
    const std::shared_ptr<ShardedStateVjp>& next_state,const std::vector<std::vector<CacheCotangents>>& state_roots,
    const std::vector<ProjectionGradient>& reuse,bool reuse_attention);
void close_sharded_graph_vjp(const std::vector<ShardedGraphVjp>&); // Coordinator must already be closed.
// Submit all device programs before any wait, including retained windows.
void run_sharded_graph_vjp(DeviceProgram&,const std::vector<ShardedGraphVjp>&);
void run_sharded_graph_vjp(DeviceSequence&,const std::vector<ShardedGraphVjp>&);
class ShardedParameterReduce;
void run_sharded_graph_vjp(DeviceSequence&,const std::vector<ShardedGraphVjp>&,const std::vector<ShardedParameterReduce*>&);
} // namespace tide::device_online
