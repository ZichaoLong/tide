#pragma once
#include "ready_batch.h"

namespace tide::device_online {
struct PackedSum {at::Tensor content,weighted,keys,order;};
// Consume actual packed fibers in their stable order. Vectorization tiles the
// payload dimension, never the logical message group or its summation order.
// Optional [edge,2] origins declare (visible port or -1, positive stride).
// Sorting affects the Aggregate view; weighted rows retain physical atom order.
// Inference only: rejects autograd and tensors requiring gradients.
PackedSum append_packed_sum(CannProgram&,const ReadyBatch&,const at::Tensor& sources,
    const at::Tensor& scales,int64_t nodes,int64_t inputs,int64_t edges,
    const at::Tensor& error,bool vectorized,const at::Tensor& input_origins={});
} // namespace tide::device_online
