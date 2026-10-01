#pragma once
#include "event_tape.h"

namespace tide::device_online {
// Same-fiber records are grouped by head geometry. Shared actual journal rows
// include all fiber groups; their metadata is associated on device.
struct FiberAttentionTape {
  EventAttentionTape cache;
  at::Tensor bias,qkv_bias,projection_bias,decay;
  at::Tensor pool_kinds,pool_lengths,pool_weights;
};
// Live, ungrouped writable banks; a grouped reverse tape must never be used to
// publish into gathered copies instead of the forward owner's actual storage.
struct FiberParameterBanks {
  std::vector<int64_t> nodes;
  at::Tensor qkv,qkv_bias,projection,projection_bias,decay,pool;
};
std::vector<int64_t> fiber_parameter_offsets(const Graph&,int64_t width);
} // namespace tide::device_online
