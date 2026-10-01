#pragma once
#include "reverse_statistics.h"
#include "state_owner_read.h"
#include "event_reverse.h"
#include "fiber_reverse.h"

namespace tide::device_online {
struct StateShardGradient {
  std::vector<int64_t> nodes;
  StateReverseLayout layout;
  at::Tensor decay,decay_connected,retention,retention_connected,read,read_connected;
  at::Tensor attention,attention_connected,fiber,fiber_connected;
  std::vector<CacheGradient> cache;
  ReverseStatistics statistics;
};
struct StateOwnerVjp {StateVjp state;at::Tensor messages,connected,scale_partials;};
// All actual KV records, parameter versions and cache boundary adjoints remain
// on this owner. A stage contains only current event cotangents and carry.
class StateOwnerReverse {
 public:
  StateOwnerReverse(CannProgram&,const StateOwnerTape&,const StateReversePacket&,
      const std::vector<CacheCotangents>&,const std::vector<CacheGradient>& next_cache,
      const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget);
  StateOwnerVjp append_stage(CannProgram&,const StateReverseStage&,const at::Tensor& error);
  void append_finish(CannProgram&,const at::Tensor& error);
  StateShardGradient gradient() const {return total_;}
 private:
  StateOwnerTape owner_;
  StateReverseView view_;
  ReverseLinks links_;
  StateShardGradient total_;
  std::vector<EventReverse> events_;
  std::vector<FiberReverse> fibers_;
  at::Tensor messages_,connected_,partials_,decay_,retention_,decay_on_,retention_on_;
  int64_t chunk_,budget_;
  bool built_=false,finished_=false;
};
} // namespace tide::device_online
