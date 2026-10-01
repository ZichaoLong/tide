#pragma once
#include "state_vjp.h"
#include "event_tape.h"
#include "fiber_tape.h"
#include "full_vjp.h"
#include "aggregate_tape.h"
#include "control_vjp.h"
#include "packed_queue.h"
#include "tide/types.h"

namespace tide::device_online {
// Borrowed actual forward journals and frozen static topology/parameter tables.
// Valid only while the owner lives and before its next advance. This first
// reverse profile is HARD/HST/SOFTP, built-in Aggregate, broadcast (including phases),
// identity/EMA/Add/event/fiber-attention state and identity/tanh/LH/SwiGLU Full; no CPU event trace is accepted.
struct ReverseTape {
  const Graph* graph;
  StateTape state;
  FullTape full;
  at::Tensor full_values,fiber_meta,fiber_values,fiber_count,sources,source_scales,delivery_scales;
  AtomBatch pending,outputs;
  at::Tensor pending_count,output_count;
  int64_t cut,stop;
  AggregateTape aggregate;
  ControlTape control;
  std::vector<EventAttentionTape> attention;
  std::vector<FiberAttentionTape> fiber;
};
struct ReverseLinks {
  // Physical rows: all fiber capacity, pending capacity, output capacity.
  // Columns: consuming event, producing event, Aggregate scale, delivery scale.
  // -1 is a boundary/no-owner index, never a numerical zero message.
  at::Tensor messages,valid;
  at::Tensor producer_head,producer_next,consumer_head,consumer_next;
  // Two possible scale references per message: Aggregate then delivery.
  at::Tensor scale_head,scale_next,stage_offsets,stages,scales;
  int64_t fibers,pending,outputs,parameters;
};
// Static topology preparation is allowed; message/event association and stage
// boundaries are computed on device from this candidate's actual forward tape.
ReverseLinks append_reverse_links(CannProgram&,const ReverseTape&,const at::Tensor& error,
                                  int64_t tensor_budget_bytes);
} // namespace tide::device_online
