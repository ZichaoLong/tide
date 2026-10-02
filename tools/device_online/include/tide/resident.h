#pragma once
#include "tide/types.h"
#include <memory>

namespace tide {
namespace device_online {class ContentFlow;struct SavedContent;}
// Opaque, detached numerical continuation on its original NPU owners. Only the
// originating live session can restore it; parameters/optimizer are not copied.
// Copies of this handle share immutable saved buffers. Not a disk checkpoint.
class ResidentContinuation {
 public:
  ResidentContinuation()=default;
  Index cut() const;
  Index batch_size() const;
  Index tensor_bytes() const;
 private:
  std::shared_ptr<const device_online::SavedContent> data_;
  friend class device_online::ContentFlow;
};
enum class ResidentChunkPolicy { conservative, aggressive };
struct ResidentLimits {
  int64_t queue=1024, arrivals=1024, outputs=1024, trace=4096, stages=4096;
  int64_t workspace_bytes=64*1024*1024;
  ResidentChunkPolicy chunk_policy=ResidentChunkPolicy::conservative;
  int64_t full_chunk_rows=16, emission_chunk_rows=16, aggregate_chunk_rows=8;
  int64_t attention_chunk_rows=8, attention_key_rows=128, kv_rows=128, kv_trace_rows=4096;
  int64_t max_repeat_ticks=65536;
  bool prefill=true, diagnostics=true;
  bool vectorized_aggregate=true, vectorized_state=true, vectorized_read=true;
  std::string mode="hard";
  double zeta=1.;
};
// Empty devices keeps the original single-device implementation. Otherwise
// devices[0] must equal the coordinator. Empty owner maps request static generic
// planning; explicit maps index this list, one entry per execution graph node.
struct ResidentPlacement {
  std::vector<at::Device> devices;
  std::string policy="locality";
  std::vector<Index> full_owners,state_owners;
};
// Read-only borrowed buffers. A view expires on advance/restore/close/destruction;
// clone the tensors to retain them. valid distinguishes absent and zero outputs.
// coordinates: [sample,node,time,kind,output-port,position]. No state export.
struct ResidentWindow {
  Tensor coordinates, values, valid;
  Tensor output_stats, pending_stats, stages, events, full_chunks, emission_chunks;
};
// Optional CANN backend, built separately from the portable core. Single/sharded
// NPU FP32/FP16 inference; construction and execution require explicit no-grad.
// Accepts the documented built-in module profiles and arbitrary legal topology.
// Construction freezes parameter values; normal in-place updates are refused
// until a new session is constructed. No training/autograd is implied.
class ResidentSession {
 public:
  ResidentSession(Graph, Model, const Continuation&, at::Device, ResidentLimits={});
  ResidentSession(Graph, Model, const Continuation&, at::Device, ResidentLimits, ResidentPlacement);
  ~ResidentSession();
  ResidentSession(const ResidentSession&)=delete;
  ResidentSession& operator=(const ResidentSession&)=delete;
  // Input values may be CPU or on this NPU, all on one device per call. Only
  // input coordinates/seals live on the host; payloads are transferred in bulk.
  ResidentWindow advance(const std::vector<External>&, Index stop, Index sealed_until);
  Continuation snapshot() const; // Explicit CPU checkpoint materialization.
  ResidentContinuation snapshot_device(Index max_bytes) const;
  void restore_device(const ResidentContinuation&); // Clears latest-window diagnostics.
  Result result() const;         // Explicit latest-window CPU diagnostics.
  Index cut() const;
  ResidentPlacement placement() const; // Resolved static placement; no device read.
  void close();                  // Checked resource drain; idempotent.
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide
