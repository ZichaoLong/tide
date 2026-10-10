#pragma once
#include "content_profile.h"
#include "device_journal.h"
#include "event_tape.h"
#include "continuation_pack.h"

namespace tide::device_online {
struct EventCache {at::Tensor key,value,lengths;};
struct EventGroupStage {at::Tensor events,counts;EventCache cache;JournalProposal journal;};
struct EventAttentionStage {at::Tensor values;std::vector<EventGroupStage> groups;};
// Ordered actual node-time events share a staged immutable prefix plus compact
// new KV. Groups share head geometry only; topology/input do not specialize the
// schedule. Selection-dependent adoption/clear retains single-frame fallback.
struct EventAttentionGroup {
  int64_t nodes,width,query_heads,kv_heads,head_width,kv_width,parameters,owners,rows,capacity,chunk,key_rows;
  std::vector<int64_t> node_map;
  std::vector<bool> adopt_all,clear;
  at::Tensor mapping,windows,config,qkv,projection,chunks,peak,key_work;
  EventCache live;
  std::unique_ptr<DeviceJournal> journal;
  EventAttentionGroup(const ContentProfile&,const Continuation&,at::Device,const ContentLimits&,
                      int64_t query_heads,int64_t kv_heads,int64_t chunk,int64_t key_rows);
  EventAttentionGroup(const StateKernelProfile&,const Continuation&,at::Device,const ContentLimits&,
                      int64_t query_heads,int64_t kv_heads,int64_t chunk,int64_t key_rows);
  EventGroupStage propose(DeviceProgram&,const ReadyBatch&,const ContentBatch&,const at::Tensor& values,const at::Tensor& error);
  void commit(DeviceProgram&,const EventGroupStage&,const SelectionProposal&,const at::Tensor& error);
  void export_states(Continuation&) const;
  void export_trace(std::vector<Event>&) const;
};
class PackedEventAttention {
 public:
  static long double minimum_bytes(const ContentProfile&,const Continuation&,const ContentLimits&);
  static long double minimum_bytes(const StateKernelProfile&,const Continuation&,const ContentLimits&);
  PackedEventAttention(const ContentProfile&,const Continuation&,at::Device,const ContentLimits&,int64_t budget);
  PackedEventAttention(const StateKernelProfile&,const Continuation&,at::Device,const ContentLimits&,int64_t budget);
  EventAttentionStage propose(DeviceProgram&,const ReadyBatch&,const ContentBatch&,const at::Tensor& error,
                              const at::Tensor& initial_values={});
  void commit(DeviceProgram&,const EventAttentionStage&,const SelectionProposal&,const at::Tensor& error);
  void reset_window();
  std::vector<EventAttentionTape> tape() const;
  void export_states(Continuation&) const;
  void export_trace(std::vector<Event>&) const;
  int64_t reserved_bytes() const {return reserved_;}
  int64_t chunk_rows() const {return chunk_;}
  int64_t key_rows() const {return key_rows_;}
  at::Tensor key_work() const;
  at::Tensor chunks() const;
  at::Tensor peak() const;
  std::vector<const DeviceJournal*> journals() const {
    std::vector<const DeviceJournal*> out;
    for(const auto& g:groups_)if(g->journal)out.push_back(g->journal.get());
    return out;
  }
  std::vector<ContinuationRows> continuation_rows() const {
    std::vector<ContinuationRows> out;
    for(const auto& g:groups_)out.push_back({{g->live.key,g->live.value},g->live.lengths,g->capacity});
    return out;
  }
  std::vector<Tensor> continuation_tensors() const {
    std::vector<Tensor> out;
    for(const auto& g:groups_)out.insert(out.end(),{g->live.key,g->live.value,g->live.lengths});
    return out;
  }
 private:
  int64_t rows_,width_,chunk_,key_rows_,reserved_;
  std::vector<std::unique_ptr<EventAttentionGroup>> groups_;
};
} // namespace tide::device_online
