#pragma once
#include "full_placement.h"
#include "broadcast_router.h"
#include "full_shard_tape.h"
#include <memory>
namespace tide::device_online {
struct ContentProfile;
class ShardedFull {
 public:
  static long double minimum_bytes(const ContentProfile&,const FullPlacement&,int64_t capacity);
  ShardedFull(const ContentProfile&,FullPlacement,at::Device coordinator,int64_t capacity,
              int64_t max_rows,int64_t tensor_budget);
  ~ShardedFull();
  ActionBatch append_stage(CannProgram&,const ActionBatch&,const at::Tensor& content,
                          const at::Tensor& comparison,const at::Tensor& error,int64_t operator_budget);
  void append_stop(CannProgram&);
  void reset_window();
  void synchronize_inputs() const;
  void submit();
  void wait();
  void close();
  int64_t reserved_bytes() const;
  int64_t program_count() const; // Including the graph coordinator.
  int64_t workspace_bytes() const; // Excluding the graph coordinator.
  int64_t packet_bytes() const;
  int64_t retained_tensor_bytes() const;
  const at::Tensor& chunks() const;
  std::map<std::string,int64_t> stats() const;
  std::vector<FullShardTape> tapes(int64_t samples,int64_t width) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
