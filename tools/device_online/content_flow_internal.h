#pragma once
#include "content_profile.h"
#include "device_journal.h"
#include "queue_transaction.h"

namespace tide::device_online {
struct ContentFlow::Impl {
  ContentProfile profile;
  ContentLimits limits;
  at::Device device;
  Continuation boundary;
  ContentState state;
  SelectionHistory history;
  std::unique_ptr<FrameSelector> selector;
  std::unique_ptr<QueueTransaction> pending,outputs,messages;
  std::unique_ptr<DeviceJournal> events,fibers,contributions;
  AtomBatch external;
  at::Tensor error,stop,stages;
  std::unique_ptr<CannProgram> program;
  bool failed=false;
  Impl(Graph,Model,const Continuation&,at::Device,ContentLimits);
  void construct();
  Result export_result(const Continuation& before);
};
void upload_atoms(const std::vector<Atom>&,const AtomBatch&);
std::vector<Atom> download_atoms(const AtomBatch&);
} // namespace tide::device_online
