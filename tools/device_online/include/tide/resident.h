#pragma once
#include "tide/types.h"
#include <memory>

namespace tide {
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
// Read-only borrowed device buffers. A view expires on advance/close/destruction;
// clone the tensors to retain them. valid distinguishes absent and zero outputs.
// coordinates: [sample,node,time,kind,output-port,position]. No state export.
struct ResidentWindow {
  Tensor coordinates, values, valid;
  Tensor output_stats, pending_stats, stages, events, full_chunks, emission_chunks;
};
// Optional CANN backend, built separately from the portable core. Single NPU,
// FP32 HARD inference; construction and execution require explicit no-grad.
// Accepts the documented built-in module profiles and arbitrary legal topology.
// Construction freezes parameter values; normal in-place updates are refused
// until a new session is constructed. No training/autograd is implied.
class ResidentSession {
 public:
  ResidentSession(Graph, Model, const Continuation&, at::Device, ResidentLimits={});
  ~ResidentSession();
  ResidentSession(const ResidentSession&)=delete;
  ResidentSession& operator=(const ResidentSession&)=delete;
  // Input values may be CPU or on this NPU, all on one device per call. Only
  // input coordinates/seals live on the host; payloads are transferred in bulk.
  ResidentWindow advance(const std::vector<External>&, Index stop, Index sealed_until);
  Continuation snapshot() const; // Explicit CPU checkpoint materialization.
  Result result() const;         // Explicit latest-window CPU diagnostics.
  Index cut() const;
  void close();                  // Checked resource drain; idempotent.
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide
