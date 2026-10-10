#pragma once
#include "content_profile.h"
#include "tide/ops.h"
#include "device_journal.h"
#include "queue_transaction.h"
#include "packed_full.h"
#include "packed_lh_full.h"
#include "packed_emission.h"
#include "packed_swiglu_full.h"
#include "packed_fiber_attention.h"
#include "packed_event_attention.h"
#include "packed_aggregate.h"
#include "remote_full.h"
#include "sharded_full.h"
#include "sharded_state.h"

namespace tide::device_online {
struct ContentFlow::Impl {
  ContentProfile profile;
  ContentLimits limits;
  at::Device device;
  at::Device full_device;
  Continuation boundary;
  ContentState state;
  SelectionHistory history,history_before;
  std::unique_ptr<FrameSelector> selector;
  std::unique_ptr<PackedFull> full;
  std::unique_ptr<PackedLhFull> lh_full;
  std::unique_ptr<PackedEmission> emission;
  std::unique_ptr<PackedSwiGluFull> swiglu_full;
  std::unique_ptr<PackedFiberAttention> attention;
  std::unique_ptr<PackedEventAttention> event_attention;
  std::unique_ptr<PackedAggregate> aggregate;
  std::unique_ptr<QueueTransaction> pending,outputs,messages;
  std::unique_ptr<DeviceJournal> events,fibers,contributions,full_trace,raw_full_trace,emission_trace;
  AtomBatch external;
  at::Tensor error,stop,stages,event_count,source_scales_before,full_chunks;
  std::unique_ptr<RemoteFull> remote_full;
  std::unique_ptr<ShardedFull> sharded_full;
  std::unique_ptr<ShardedState> sharded_state;
  std::unique_ptr<DeviceProgram> program;
  int64_t planned_buffer_bytes=0,operator_workspace_budget=0,usable_memory_budget=0;
  bool failed=false,export_diagnostics=true;
  int64_t window_start=0;
  std::shared_ptr<const int> continuation_owner=std::make_shared<const int>(0);
  Impl(Graph,Model,const Continuation&,at::Device,ContentLimits,at::Device,FullPlacement={},FullPlacement={},bool retain_backward=false);
  void construct();
  void reset_window();
  std::vector<Tensor> continuation_tensors() const;
  Continuation export_continuation() const;
  Result export_result() const;
  ReverseTape reverse_view(const StateTape&,const FullTape&) const;
};
void upload_atoms(const std::vector<Atom>&,const AtomBatch&);
ValidatedInput prepare_external(const Graph&,const Model&,const Continuation&,
    const std::vector<External>&,Index,at::Device,const AtomBatch&);
std::vector<Atom> download_atoms(const AtomBatch&);
} // namespace tide::device_online
