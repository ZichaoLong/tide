#pragma once
#include "cann_program.h"
#include "tide/types.h"
#include <vector>

namespace tide::device_online {
// Candidate-owned forward cache journals, grouped only by head geometry.
// KV rows retain event coordinates; reverse association is device work.
struct EventAttentionTape {
  std::vector<int64_t> nodes;
  at::Tensor mapping,windows,config,qkv,projection;
  at::Tensor metadata,values,count,key,value,lengths;
  int64_t samples=0,heads=0,kv_heads=0,width=0,capacity=0;
};
struct CacheCotangents {
  at::Tensor key,value,key_connected,value_connected;
  at::Tensor bias,bias_connected; // same-fiber log-bias only
};
struct CacheGradient : CacheCotangents {
  at::Tensor lengths;
};
// Flattened Q,K,V,O parameter layout in node order, without padded KV heads.
std::vector<int64_t> event_parameter_offsets(const Graph&,int64_t width);
std::vector<int64_t> event_parameter_offsets(const std::vector<Node>&,int64_t width);
} // namespace tide::device_online
