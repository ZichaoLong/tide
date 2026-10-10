#pragma once
#include "full_vjp.h"

namespace tide::device_online {
// Shared bounded row planner and ordered parameter reduction. All row choices
// use the candidate's actual cotangent connectivity and forward journal.
struct ExtraFullRows {
  at::Tensor source,parameters,destination,owners,owner_count,cursor,branch,x,gradient,output;
  int64_t chunk;
  ExtraFullRows(DeviceProgram&,const FullTape&,const FullVjp&,int64_t rows);
  void plan(DeviceProgram&,const FullTape&,const at::Tensor& connected,FullVjp&,
            const at::Tensor& kinds,const at::Tensor& mapping,int64_t kind,
            int64_t parameter_count,bool residual,const at::Tensor& error) const;
  void payload(DeviceProgram&,const FullTape&,const at::Tensor& upstream,FullVjp&,
               bool residual,const at::Tensor& error) const;
  void reduce(DeviceProgram&,const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,
              const at::Tensor& oa,const at::Tensor& ob,const at::Tensor& oc,const at::Tensor& error) const;
};
void append_lh_full_vjp(DeviceProgram&,const FullTape&,const at::Tensor& gradient,const at::Tensor& connected,
                       FullVjp&,const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes);
void append_swiglu_vjp(DeviceProgram&,const FullTape&,const at::Tensor& gradient,const at::Tensor& connected,
                       FullVjp&,const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes);
void full_extra_tensor(const at::Tensor&,at::Device,at::ScalarType,at::IntArrayRef);
} // namespace tide::device_online
