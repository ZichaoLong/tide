#pragma once
#include "full_shard_tape.h"
namespace tide::device_online {
struct FullShardGradient {std::vector<int64_t> nodes;FullVjp values;};
// One service per remote Full owner. Parameter partials accumulate where their
// actual forward bank lives; only stage cotangents/connection bits return.
class ShardedFullVjp {
 public:
  ShardedFullVjp(CannProgram&,const ShardedReverseTape&,int64_t chunk_rows,
                int64_t tensor_budget,int64_t per_program_workspace);
  ~ShardedFullVjp();
  FullVjp append_stage(CannProgram&,const FullTape&,const at::Tensor& gradient,
                       const at::Tensor& connected,const at::Tensor& error);
  void append_stop(CannProgram&);
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
ShardedGraphVjp append_sharded_graph_vjp(CannProgram&,const ShardedReverseTape&,
    const GraphCotangents&,const at::Tensor& error,int64_t chunk_rows,
    int64_t tensor_budget,int64_t per_program_workspace);
ShardedGraphVjp append_sharded_graph_vjp(CannProgram&,const ShardedReverseTape&,
    const GraphCotangents&,const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget,int64_t per_program_workspace,
    const std::shared_ptr<ShardedStateVjp>& next_state,const std::vector<std::vector<CacheCotangents>>& state_roots);
void close_sharded_graph_vjp(const std::vector<ShardedGraphVjp>&); // Coordinator must already be closed.
// Submit all device programs before any wait, including retained windows.
void run_sharded_graph_vjp(CannProgram&,const std::vector<ShardedGraphVjp>&);
} // namespace tide::device_online
