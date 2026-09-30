#pragma once
#include "content_profile.h"
#include "device_journal.h"

namespace tide::device_online {
struct EventCache {at::Tensor key,value,lengths;};
struct EventGroupStage {at::Tensor events,counts;EventCache cache;JournalProposal journal;};
struct EventAttentionStage {at::Tensor values;std::vector<EventGroupStage> groups;};
// One actual event per owner in a ready stage. Groups share head geometry only;
// topology/input do not specialize the schedule. KV storage stays compact.
struct EventAttentionGroup {
  int64_t nodes,width,query_heads,kv_heads,head_width,kv_width,parameters,owners,rows,capacity,chunk;
  std::vector<int64_t> node_map;
  std::vector<bool> adopt_all,clear;
  at::Tensor mapping,windows,config,qkv,projection,chunks,peak;
  EventCache live;
  std::unique_ptr<DeviceJournal> journal;
  EventAttentionGroup(const ContentProfile&,const Continuation&,at::Device,const ContentLimits&,
                      int64_t query_heads,int64_t kv_heads,int64_t chunk);
  EventGroupStage propose(CannProgram&,const ReadyBatch&,const ContentBatch&,const at::Tensor& values,const at::Tensor& error);
  void commit(CannProgram&,const EventGroupStage&,const SelectionProposal&,const at::Tensor& error);
  void export_states(Continuation&) const;
  void export_trace(std::vector<Event>&) const;
};
class PackedEventAttention {
 public:
  static long double minimum_bytes(const ContentProfile&,const Continuation&,const ContentLimits&);
  PackedEventAttention(const ContentProfile&,const Continuation&,at::Device,const ContentLimits&,int64_t budget);
  EventAttentionStage propose(CannProgram&,const ReadyBatch&,const ContentBatch&,const at::Tensor& error,
                              const at::Tensor& initial_values={});
  void commit(CannProgram&,const EventAttentionStage&,const SelectionProposal&,const at::Tensor& error);
  void reset_window();
  void export_states(Continuation&) const;
  void export_trace(std::vector<Event>&) const;
  int64_t reserved_bytes() const {return reserved_;}
  int64_t chunk_rows() const {return chunk_;}
  at::Tensor chunks() const;
  at::Tensor peak() const;
 private:
  int64_t rows_,width_,chunk_,reserved_;
  std::vector<std::unique_ptr<EventAttentionGroup>> groups_;
};
} // namespace tide::device_online
