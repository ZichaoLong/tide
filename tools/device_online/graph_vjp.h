#pragma once
#include "reverse_links.h"

namespace tide::device_online {
struct GraphCotangents {
  at::Tensor outputs,outputs_connected,pending,pending_connected,final,final_connected;
  std::vector<CacheCotangents> cache;
};
struct GraphVjp {
  ReverseLinks links;
  // Gradients for physical fiber/pending/output rows; producer -1 marks an
  // incoming boundary leaf. Present zeros and None remain independent.
  at::Tensor messages,message_connected,initial,initial_connected;
  at::Tensor weights,biases,full_connected,decay,decay_connected,retention,retention_connected;
  at::Tensor scales,scale_connected,reverse_stages;
  FullExtraVjp extra;
  AggregateVjp aggregate;
  at::Tensor read,read_connected;
  std::vector<CacheGradient> cache;
  at::Tensor attention,attention_connected;
  at::Tensor fiber,fiber_connected;
  EmissionVjp emission;
};
// Internal first-order single-window HARD/HST/SOFTP graph adjoint. All reverse stage,
// state-chain and message progression remains on device. Returned parameter
// rows require the public registry's alias accumulation before optimizer use.
// FP16/FP32 forward tapes retain actual payload precision; roots, journals and
// returned adjoints are FP32. Public training has separate capability gates.
GraphVjp append_graph_vjp(CannProgram&,const ReverseTape&,const GraphCotangents&,
                         const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes);
// Internal composition seam: invoked once while constructing the device stage
// loop. The returned Full tensors participate in that loop, not host dispatch.
using FullStageVjp=std::function<FullVjp(CannProgram&,const FullTape&,const at::Tensor&,
                                       const at::Tensor&,const at::Tensor&)>;
GraphVjp append_graph_vjp(CannProgram&,const ReverseTape&,const GraphCotangents&,
                         const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes,
                         const FullStageVjp&);
// State/cache owners prepare once, consume each actual reverse stage, and
// publish fiber adjoints after coordinator Aggregate processing. Callbacks only
// construct device programs; they never dispatch individual events on the host.
struct GraphStateVjp {
  std::function<void(CannProgram&,const ReverseLinks&)> prepare;
  std::function<StateVjp(CannProgram&,const at::Tensor&,const StateCotangents&,const ControlScores&)> stage;
  std::function<void(CannProgram&,const at::Tensor&,const at::Tensor&,const at::Tensor&)> sources;
};
GraphVjp append_graph_vjp(CannProgram&,const ReverseTape&,const GraphCotangents&,
    const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes,const FullStageVjp&,const GraphStateVjp&);

} // namespace tide::device_online
