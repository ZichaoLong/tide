#pragma once
#include "parameter_vjp.h"

namespace tide::device_online {
struct EventReverse {
  CacheGradient cache;
  at::Tensor previous,tails,ranges,node_offsets;
};
EventReverse prepare_event_reverse(CannProgram&,const ReverseTape&,const EventAttentionTape&,
    const CacheCotangents&,const at::Tensor& error,int64_t tensor_budget_bytes);
// One reverse stage; dependencies between successive caches of the same owner
// are device-controlled. Independent owners are packed into the same batch.
void append_event_reverse(CannProgram&,const ReverseTape&,const EventAttentionTape&,const EventReverse&,
    const at::Tensor& range,StateVjp&,const at::Tensor& parameters,const at::Tensor& parameter_connections,
    const at::Tensor& error,int64_t chunk,int64_t tensor_budget_bytes);
CacheCotangents append_cache_bridge(CannProgram&,const EventAttentionTape&,const CacheCotangents&,
    const CacheGradient&,const at::Tensor& error,int64_t tensor_budget_bytes);
void append_event_publish(CannProgram&,const std::vector<EventAttentionTape>&,const ParameterVjp&,
    const at::Tensor& values,const at::Tensor& error,int64_t tensor_budget_bytes);
} // namespace tide::device_online
