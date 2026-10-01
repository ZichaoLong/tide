#pragma once
#include "attention_vjp.h"

namespace tide::device_online {
// Local same-fiber adjoint. Rows are already physically scaled, in stable slot
// order. The proposed cache/bias comes from this candidate's actual forward.
// Graph adoption/clear and old-cache carry are composed outside this component.
struct FiberVjpInput {
  at::Tensor rows,slots,counts;          // [B,S,D], [B,S], [B]
  at::Tensor key,value,bias;            // [B,H,K,d], [B,H,K,d], [B,K]
  at::Tensor lengths,old_lengths,ticks; // [B], all int64
  at::Tensor qkv,qkv_bias,projection;   // [B,D,3D], [B,3D], [B,D,D]
  at::Tensor pool_kinds,pool_lengths,pool_weights; // [B], [B], [B,L]
  at::Tensor cotangent,connected;      // [B,D], [B]
  at::Tensor key_root,value_root,bias_root,key_on,value_on,bias_on;
  int64_t max_repeat_ticks=4096;
};
struct FiberVjp {
  at::Tensor rows,rows_connected;
  at::Tensor key,value,bias; // old-cache prefixes; padded suffix is zero
  at::Tensor cache_connected; // [B,3], key/value/log_bias, including empty slices
  at::Tensor qkv,qkv_bias,projection,projection_bias,decay,pool;
  at::Tensor parameter_connected; // [B,6], in the parameter order above
  at::Tensor chunks;
};
FiberVjp append_fiber_vjp(CannProgram&,const FiberVjpInput&,const at::Tensor& error,
    int64_t query_chunk_rows,int64_t key_tile_rows,int64_t tensor_budget_bytes);
} // namespace tide::device_online
