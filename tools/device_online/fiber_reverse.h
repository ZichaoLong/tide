#pragma once
#include "fiber_tape.h"
#include "graph_vjp.h"

namespace tide::device_online {
struct FiberReverse {
  at::Tensor ranges,previous,tails,tokens,ticks,node_offsets;
  CacheGradient cache;
};
CacheCotangents append_fiber_cache_seed(CannProgram&,const FiberAttentionTape&,const CacheCotangents&,
    const CacheGradient* later,const at::Tensor& error,int64_t tensor_budget_bytes);
FiberReverse prepare_fiber_reverse(CannProgram&,const ReverseTape&,const ReverseLinks&,const FiberAttentionTape&,
    const CacheCotangents&,const at::Tensor& error,int64_t tensor_budget_bytes);
void append_fiber_reverse(CannProgram&,const ReverseTape&,const ReverseLinks&,const FiberAttentionTape&,const FiberReverse&,
    const at::Tensor& stage,const StateVjp&,const at::Tensor& messages,const at::Tensor& message_connected,
    const at::Tensor& scale_partials,const at::Tensor& parameters,const at::Tensor& parameter_connected,
    const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes);
} // namespace tide::device_online
