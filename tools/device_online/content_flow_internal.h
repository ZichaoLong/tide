#pragma once
#include "content_profile.h"
#include "device_journal.h"
#include "queue_transaction.h"
#include "packed_full.h"
#include "packed_lh_full.h"
#include "packed_emission.h"

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
  std::unique_ptr<QueueTransaction> pending,outputs,messages;
  std::unique_ptr<DeviceJournal> events,fibers,contributions,full_trace,emission_trace;
  AtomBatch external;
  at::Tensor error,stop,stages,event_count;
  std::unique_ptr<CannProgram> program;
  bool failed=false;
  Impl(Graph,Model,const Continuation&,at::Device,ContentLimits);
  void construct();
  Continuation export_continuation() const;
  Result export_result() const;
};
void upload_atoms(const std::vector<Atom>&,const AtomBatch&);
std::vector<Atom> download_atoms(const AtomBatch&);
} // namespace tide::device_online
