#pragma once
#include "content_profile.h"
#include "device_journal.h"
#include "packed_fiber_pool.h"
#include "fiber_tape.h"

namespace tide::device_online {
struct FiberCache {at::Tensor key,value,bias,lengths;};
struct FiberStage {
  at::Tensor values,events,tokens,counts,query_bias;
  FiberCache cache;
  JournalProposal journal;
};
// Bounded same-fiber attention with ordered node-time prefixes when adoption is
// selection-independent. Physical chunks preserve complete current-fiber
// visibility and each event's exact repeated bias decay. No cache eviction.
class PackedFiberAttention {
 public:
  static long double minimum_bytes(const ContentProfile&,const Continuation&,const ContentLimits&);
  static long double minimum_bytes(const StateKernelProfile&,const Continuation&,const ContentLimits&);
  PackedFiberAttention(const ContentProfile&,const Continuation&,at::Device,
                       const ContentLimits&,int64_t byte_budget);
  PackedFiberAttention(const StateKernelProfile&,const Continuation&,at::Device,
                       const ContentLimits&,int64_t byte_budget);
  FiberStage propose(CannProgram&,const ContentProfile&,const ReadyBatch&,
                     const ContentBatch&,const ContentState&,const at::Tensor& error);
  FiberStage propose(CannProgram&,const StateKernelProfile&,const ReadyBatch&,
                     const ContentBatch&,const ContentState&,const at::Tensor& error);
  void commit(CannProgram&,const FiberStage&,const SelectionProposal&,const at::Tensor& error);
  void reset_window();
  void export_states(Continuation&) const;
  void export_trace(std::vector<Event>&) const;
  int64_t reserved_bytes() const {return reserved_;}
  int64_t chunk_rows() const {return chunk_;}
  int64_t key_rows() const {return key_rows_;}
  at::Tensor key_work() const {return key_work_;}
  at::Tensor chunks() const {return chunks_;}
  at::Tensor peak() const {return peak_;}
  std::vector<FiberAttentionTape> tape() const;
  FiberParameterBanks banks() const;
 private:
  int64_t nodes_,width_,parameters_,owners_,rows_,capacity_,chunk_,key_rows_,reserved_,max_ticks_;
  std::vector<int64_t> node_map_,node_heads_,head_groups_,source_lengths_;
  std::vector<bool> adopt_all_,clear_;
  at::Tensor mapping_,heads_,qkv_,qkv_bias_,projection_,projection_bias_,decay_,config_;
  at::Tensor chunks_,peak_,key_work_;
  FiberCache cache_;
  std::unique_ptr<DeviceJournal> journal_;
  std::unique_ptr<PackedFiberPool> pool_;
};
} // namespace tide::device_online
