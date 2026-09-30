#pragma once
#include "reverse_links.h"

namespace tide::device_online {
struct GraphCotangents {
  at::Tensor outputs,outputs_connected,pending,pending_connected,final,final_connected;
};
struct GraphVjp {
  ReverseLinks links;
  // Gradients for physical fiber/pending/output rows; producer -1 marks an
  // incoming boundary leaf. Present zeros and None remain independent.
  at::Tensor messages,message_connected,initial,initial_connected;
  at::Tensor weights,biases,full_connected,decay,decay_connected,retention,retention_connected;
  at::Tensor scales,scale_connected,reverse_stages;
  FullExtraVjp extra;
};
// Internal first-order single-window HARD graph adjoint. All reverse stage,
// state-chain and message progression remains on device. Returned parameter
// rows require the public registry's alias accumulation before optimizer use.
GraphVjp append_graph_vjp(CannProgram&,const ReverseTape&,const GraphCotangents&,
                         const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes);
} // namespace tide::device_online
