#pragma once
#include "cann_program.h"
#include "full_extra.h"

namespace tide::device_online {
// Actual forward event journal and immutable PackedFull parameter banks.
// Tanh parameter bank row nodes is its owned zero sentinel. Non-tanh owners are
// never evaluated. A no-tanh profile has undefined parameter banks.
struct FullTape {
  at::Tensor metadata,values,count,kinds,weights,biases;
  int64_t samples,width;
  bool has_tanh;
  FullExtraTape extra;
};
struct FullVjp {
  at::Tensor content,comparison,content_connected,comparison_connected;
  at::Tensor weights,biases,parameter_connected,chunks;
  int64_t chunk_rows;
  FullExtraVjp extra;
};
// First-order built-in Full adjoints. Parameters are physical partials, before
// parameter alias reduction. A false connection bit represents None, not zero.
// Chunk choice/progression and parameter reduction stay on device.
FullVjp append_full_vjp(CannProgram&,const FullTape&,const at::Tensor& gradient,
                       const at::Tensor& connected,const at::Tensor& error,
                       int64_t max_chunk_rows,int64_t tensor_budget_bytes);
} // namespace tide::device_online
