#pragma once
#include "tide/types.h"
#include "packed_queue.h"
#include <memory>

namespace tide::device_online {
struct ContentLimits {
  int64_t queue=1024, arrivals=1024, outputs=1024, trace=4096, stages=4096;
  int64_t workspace_bytes=64*1024*1024;
  int64_t full_chunk_rows=16;
  int64_t max_repeat_ticks=65536; // Per Add candidate; explicit work refusal, never a power shortcut.
  bool prefill=true;
  bool diagnostics=true; // Event/message journals are optional per-window work.
  bool vectorized_aggregate=true; // Scalar device implementation remains selectable.
  bool vectorized_state=true;
  bool vectorized_read=true;
};
// Borrowed read-only NPU buffers, valid until the next advance or owner destruction.
// Output coordinates use the AtomBatch layout; field4 is the output port.
// Clone values before retaining them across calls. No persistent state download
// or event journal materialization is required to consume this view.
struct ContentWindow {
  AtomBatch outputs;
  at::Tensor output_stats,pending_stats,stages,events,full_chunks;
};
// Experimental complete forward loop for an explicit existing-module profile:
// sum Aggregate, identity/EMA/Add-repeat memory, linear/FP32-norm Read, count/positive
// selection, adopt/clear Next and identity/tanh broadcast Full. FP32, no autograd.
// Arbitrary legal positive-delay topology, including feedback. Inputs/initial
// state and exported observables are CPU values; persistent runtime data and
// all decisions between submission and the complete-cut boundary stay on NPU.
// An execution failure poisons this owner; restore a prior cut into a new one.
class ContentFlow {
 public:
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits={});
  ~ContentFlow();
  ContentFlow(const ContentFlow&)=delete;
  ContentFlow& operator=(const ContentFlow&)=delete;
  Result advance(const std::vector<External>&,Index stop);
  ContentWindow advance_device(const std::vector<External>&,Index stop);
  Continuation snapshot() const; // Explicit complete-cut CPU materialization.
  Result result() const; // Latest window; trace/messages require diagnostics.
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
