#pragma once
#include "content_profile.h"
#include "tide/fiber_attention.h"

namespace tide::device_online {
// Coefficients are applied after attention; every source still contributes KV.
// Domain logits include absent slots only for all-softmax.
class PackedFiberPool {
 public:
  static long double reserved_bytes(const ContentProfile&,int64_t rows,int64_t chunk);
  static long double reserved_bytes(const StateKernelProfile&,int64_t rows,int64_t chunk);
  PackedFiberPool(const ContentProfile&,at::Device,int64_t rows,int64_t chunk);
  PackedFiberPool(const StateKernelProfile&,at::Device,int64_t rows,int64_t chunk);
  at::Tensor append(DeviceProgram&,const at::Tensor& events,const at::Tensor& tokens,
                    const at::Tensor& counts,const ReadyBatch&,const at::Tensor& error,
                    const at::Tensor& chunks) const;
  const at::Tensor& kinds() const {return kinds_;}
  const at::Tensor& lengths() const {return lengths_;}
  const at::Tensor& weights() const {return weights_;}
 private:
  int64_t rows_,chunk_,parameters_,slots_,inputs_;
  at::Tensor kinds_,weights_,lengths_,sources_;
};
} // namespace tide::device_online
