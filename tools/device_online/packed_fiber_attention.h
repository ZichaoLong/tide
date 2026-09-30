#pragma once
#include "content_profile.h"
#include "device_journal.h"
#include "packed_fiber_pool.h"

namespace tide::device_online {
struct FiberCache {at::Tensor key,value,bias,lengths;};
struct FiberStage {
  at::Tensor values,events,tokens,counts;
  FiberCache cache;
  JournalProposal journal;
};
// Bounded same-fiber attention. One event per owner in a ready stage; independent
// owners and all message rows are packed. Physical query chunks retain the full
// candidate cache and softmax denominator. No cache eviction is implied.
class PackedFiberAttention {
 public:
  PackedFiberAttention(const ContentProfile&,const Continuation&,at::Device,
                       const ContentLimits&,int64_t byte_budget);
  FiberStage propose(CannProgram&,const ContentProfile&,const ReadyBatch&,
                     const ContentBatch&,const ContentState&,const at::Tensor& error);
  void commit(CannProgram&,const FiberStage&,const SelectionProposal&,const at::Tensor& error);
  void reset_window();
  void export_states(Continuation&) const;
  void export_trace(std::vector<Event>&) const;
  int64_t reserved_bytes() const {return reserved_;}
  int64_t chunk_rows() const {return chunk_;}
  at::Tensor chunks() const {return chunks_;}
  at::Tensor peak() const {return peak_;}
 private:
  int64_t nodes_,width_,parameters_,owners_,rows_,capacity_,chunk_,reserved_,max_ticks_;
  std::vector<int64_t> node_map_,node_heads_,head_groups_;
  std::vector<bool> adopt_all_,clear_;
  at::Tensor mapping_,heads_,qkv_,qkv_bias_,projection_,projection_bias_,decay_,config_;
  at::Tensor chunks_,peak_;
  FiberCache cache_;
  std::unique_ptr<DeviceJournal> journal_;
  std::unique_ptr<PackedFiberPool> pool_;
};
} // namespace tide::device_online
