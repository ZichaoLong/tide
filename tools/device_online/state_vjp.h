#pragma once
#include "cann_program.h"

namespace tide::device_online {
// Borrowed actual forward records. No CPU event trace is accepted by the
// integration boundary. Rows use DeviceJournal's state-event layout.
struct StateTape {
  at::Tensor metadata, values, count, config, decay, retention, clock_policy;
  int64_t samples;
  bool has_repeat=false;
  int64_t max_repeat_ticks=65536;
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
};
// First-order HARD state-chain VJP for identity, EMA and literal Add-repeat. All reverse links,
// adopt/clear choices and reverse progression are device work. This component
// alone is not a graph-training backend: routing/Full/Read/optimizer VJPs are
// separate obligations. Unsupported state kinds fail on device before writes.
StateVjp append_state_vjp(CannProgram&, const StateTape&, const StateCotangents&,
                         const at::Tensor& error, int64_t workspace_bytes,
                         int64_t repeat_chunk_ticks=256);
} // namespace tide::device_online
