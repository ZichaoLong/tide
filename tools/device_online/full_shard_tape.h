#pragma once
#include "retained_tape.h"
#include "state_owner_tape.h"

namespace tide::device_online {
// Static global node IDs and the compact banks actually used by forward.
// Numerical banks remain on their owner; journals are packed there on device.
struct FullShardTape {std::vector<int64_t> nodes;FullTape full;};
struct ShardedReverseTape {
  ReverseTape coordinator; // No Full banks: use append_sharded_graph_vjp only.
  std::vector<FullShardTape> shards;
  std::vector<StateOwnerTape> states; // Empty for historical coordinator state/KV.
};
struct RetainedShardedTape {
  std::shared_ptr<const Graph> graph;
  ShardedReverseTape tape;
  int64_t tensor_bytes=0;
};
int64_t sharded_reverse_tape_bytes(const ShardedReverseTape&);
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape&,int64_t budget);
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape&,int64_t budget,RetainedProjection*);
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape&,int64_t budget,RetainedProjection*,bool compact_journals,RetainedAttention* = nullptr);
RetainedShardedTape retain_sharded_reverse_tape(const ShardedReverseTape&,int64_t budget,RetainedProjection*,bool compact_journals,RetainedAttention*,RetainedFull*);
} // namespace tide::device_online
