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

namespace tide::device_online {
struct ContentFlow::Impl {
  ContentProfile profile;
  ContentLimits limits;
  at::Device device;
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
  std::unique_ptr<DeviceJournal> events,fibers,contributions,full_trace,emission_trace;
  AtomBatch external;
  at::Tensor error,stop,stages,event_count;
  std::unique_ptr<CannProgram> program;
  int64_t planned_buffer_bytes=0,operator_workspace_budget=0,usable_memory_budget=0;
  bool failed=false;
  Impl(Graph,Model,const Continuation&,at::Device,ContentLimits);
  void construct();
  Continuation export_continuation() const;
  Result export_result() const;
};
void upload_atoms(const std::vector<Atom>&,const AtomBatch&);
ValidatedInput prepare_external(const Graph&,const Model&,const Continuation&,
    const std::vector<External>&,Index,at::Device,const AtomBatch&);
std::vector<Atom> download_atoms(const AtomBatch&);
} // namespace tide::device_online
