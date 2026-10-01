#pragma once
#include "cann_program.h"

namespace tide::device_online {
// Bounded packed local attention adjoint. Each query has its own complete
// visible prefix; GQA shares K/V heads. Lengths and root connectivity are device
// inputs, and may change on every replay. Padding may contain arbitrary values.
struct AttentionVjpInput {
  at::Tensor query;       // [queries, query_heads, head_width]
  at::Tensor key,value;   // [queries, kv_heads, key_capacity, head_width]
  at::Tensor bias;        // [queries, key_capacity], additive per-key log bias
  at::Tensor lengths;    // int64 [queries]
  at::Tensor cotangent;  // FP32; same shape as query, including half payloads.
  at::Tensor connected;  // bool [queries]; false cotangents must never be read
};
struct AttentionVjp {
  at::Tensor query,key,value,bias,connected,tiles,output;
};
// Mathematical output: softmax(scale*Q*K^T + bias)*V. Returned per-query
// connection bits preserve None versus connected zero; graph/cache ownership
// and parameter-chain accumulation are deliberately separate integrations.
// Half Q/K/V retain payload QK rounding; normalization and adjoints use FP32.
// output is the actual (payload-rounded) forward result, widened to FP32.
AttentionVjp append_attention_vjp(CannProgram&,const AttentionVjpInput&,
    const at::Tensor& error,double scale,int64_t key_tile,int64_t tensor_budget_bytes);
} // namespace tide::device_online
