#pragma once
#include "parameter_vjp.h"

namespace tide::device_online {
// Internal writable view of an existing forward owner's numeric banks. Static
// topology remains unchanged; no CPU parameter copy participates in publish.
struct ParameterBanks {
  const Graph* graph=nullptr;
  at::Tensor weights,biases,decay,retention,read,sources,emission;
  FullExtraTape extra;
  AggregateTape aggregate;
  std::vector<EventAttentionTape> attention;
};
// Publish a packed owner vector, including every used alias (HARD Read too).
// On a sticky reverse/optimizer error this records no live bank writes.
void append_parameter_publish(CannProgram&,const ParameterBanks&,const ParameterVjp&,
                              const at::Tensor& values,const at::Tensor& error,int64_t tensor_budget_bytes);
} // namespace tide::device_online
