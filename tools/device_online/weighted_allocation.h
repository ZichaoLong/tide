#pragma once
#include "device_program.h"
#include "weighted_allocation_layout.h"
namespace tide::device_online {
// One whole region per row, lengths owned by the device. Padding is not a
// candidate and may contain arbitrary values. Numerical fields are FP32;
// lengths/branches are int64. See docs/resident-weighted-allocation.md.
struct WeightedAllocationTape {
  at::Tensor controls,slopes,statistics,branches,lengths;
  bool has_shift=true;
  at::Tensor row_errors; // actual last invocation; read only after completion
};
struct WeightedAllocationGradient {
  at::Tensor vectors;     // [rows,2,width]: score/cap cotangents
  at::Tensor scalars;     // [rows,4]: lower,upper,temperature,shift
  at::Tensor connected;   // bool [rows,6], independent of numerical zero
};
WeightedAllocationTape append_weighted_allocation(DeviceProgram&,const at::Tensor& scores,
    const at::Tensor& caps,const at::Tensor& bounds,const at::Tensor& lengths,
    const at::Tensor& error,weighted_allocation::Rule rule=weighted_allocation::Quad,
    float alpha=1.f,bool has_shift=true,int64_t budget_bytes=64*1024*1024);
// Explicit first-order map, not autograd through iterations. The tape is the
// latest actual invocation; a retained caller must journal it before reuse.
WeightedAllocationGradient append_weighted_allocation_vjp(DeviceProgram&,const WeightedAllocationTape&,
    const at::Tensor& cotangent,const at::Tensor& roots,const at::Tensor& error,
    int64_t budget_bytes=64*1024*1024);
}
