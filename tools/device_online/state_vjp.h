#pragma once
#include "device_program.h"

namespace tide::device_online {
// Borrowed actual forward records. No CPU event trace is accepted by the
// integration boundary. Rows use DeviceJournal's state-event layout.
struct StateTape {
  at::Tensor metadata, values, count, config, decay, retention, clock_policy;
  int64_t samples;
  bool has_repeat=false;
  int64_t max_repeat_ticks=65536;
  bool has_attention=false;
};
// Independent cotangents for content, old, proposal, comparison and next.
// A false connection bit means absent, not a connected numerical zero. Values
// in absent rows may be poisoned and must never enter numerical operations.
struct StateCotangents {
  at::Tensor events, connected;       // [capacity,5,width], [capacity,5]
  at::Tensor final, final_connected;  // [samples,nodes,width], [samples,nodes]
};
struct StateVjp {
  at::Tensor content, content_connected;
  at::Tensor initial, initial_connected;
  // Independent sample partials, prior to shared parameter-owner reduction.
  at::Tensor decay, decay_connected, retention_components, retention_connected;
  at::Tensor proposal,proposal_connected; // Attention only; does not flow to old visible value.
};
// First-order state-chain VJP for identity, EMA and literal Add-repeat. Attention
// exposes proposal roots for its separate KV adjoint. All reverse links,
// adopt/clear choices and reverse progression are device work. This component
// alone is not a graph-training backend: routing/Full/Read/optimizer VJPs are
// separate obligations. Unsupported state kinds fail on device before writes.
// FP16 forward parameters retain their dtype; journal values are exact FP32
// widenings. Add replay rounds each forward tick, while all cotangents and
// returned adjoints accumulate in FP32. No whole-forward FP32 substitution.
StateVjp append_state_vjp(DeviceProgram&, const StateTape&, const StateCotangents&,
                         const at::Tensor& error, int64_t workspace_bytes,
                         int64_t repeat_chunk_ticks=256);
} // namespace tide::device_online
