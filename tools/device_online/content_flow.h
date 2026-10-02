#pragma once
#include "tide/types.h"
#include "tide/resident.h"
#include "packed_queue.h"
#include "state_vjp.h"
#include "full_vjp.h"
#include "reverse_links.h"
#include "parameter_publish.h"
#include "full_placement.h"
#include "full_shard_tape.h"
#include "sharded_parameter_banks.h"
#include <memory>

namespace tide::device_online {
using ChunkPolicy=ResidentChunkPolicy;
using ContentLimits=ResidentLimits;
// Borrowed read-only NPU buffers, valid until the next advance or owner destruction.
// Output coordinates use the AtomBatch layout; field4 is the output port.
// Clone values before retaining them across calls. No persistent state download
// or event journal materialization is required to consume this view.
struct ContentWindow {
  AtomBatch outputs;
  at::Tensor output_stats,pending_stats,stages,events,full_chunks,emission_chunks;
};
// Experimental complete forward loop for an explicit existing-module profile:
// built-in Aggregate, identity/EMA/Add-repeat/fiber/event attention, linear/FP32-norm Read, count/positive
// selection, adopt/clear Next and identity/tanh/SwiGLU/LH Full with broadcast/slot-affine
// phase-aware emission; HST/SOFTP require broadcast. FP32, no autograd.
// Arbitrary legal positive-delay topology, including feedback. Inputs/initial
// state and exported observables are CPU values; persistent runtime data and
// all decisions between submission and the complete-cut boundary stay on NPU.
// An execution failure poisons this owner; restore a prior cut into a new one.
struct ModelPlacement;
class ContentFlow {
 public:
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits={});
  // Experimental inference-only peer Full placement. The coordinator retains
  // graph control/state; the peer executes selected Full batches. Not public
  // multi-device training or complete model/parameter sharding.
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits,at::Device full_device);
  // Compact Full banks on the explicit node owners; state/KV stay coordinator
  // owned. Explicit sharded tapes support the internal reverse executor;
  // optimizer publication and public multi-device training are separate.
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits,FullPlacement);
  // Internal full + state/KV forward placement; reverse/public training uses a
  // separate capability gate until compact owner tapes are integrated.
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits,ModelPlacement);
  // Training keeps the VJP journals independently of optional Result trace and
  // message exports. Public limits/ABI stay unchanged; this is an owner-only path.
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits,bool retain_backward);
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits,ModelPlacement,bool retain_backward);
  ~ContentFlow();
  ContentFlow(const ContentFlow&)=delete;
  ContentFlow& operator=(const ContentFlow&)=delete;
  Result advance(const std::vector<External>&,Index stop);
  ContentWindow advance_device(const std::vector<External>&,Index stop);
  Continuation snapshot() const; // Explicit complete-cut CPU materialization.
  Result result() const; // Latest window; trace/messages require diagnostics.
  StateTape state_tape() const; // Borrowed actual device journal; diagnostics required.
  FullTape full_tape() const; // Built-in identity/tanh/LH/SwiGLU journals and banks.
  ReverseTape reverse_tape() const; // Declared training profile; actual journals only.
  ShardedReverseTape sharded_reverse_tape() const; // Explicit compact multi-device banks.
  ParameterBanks parameter_banks() const; // Internal explicit training owner only.
  ShardedParameterBanks sharded_parameter_banks() const;
  std::pair<Tensor,Tensor> state_device() const; // Borrowed values/presence, no CPU export.
  std::vector<StateOwnerValues> state_shards_device() const; // Borrowed owner-local views.
  void close(); // Explicit checked drain; all operations except close then fail.
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
