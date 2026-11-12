#pragma once
#include "optimizer_layout.h"
#include "resident_storage_budget.h"
#include <algorithm>
namespace tide::device_online {
// Exact retained tensor storage, excluding allocator rounding and runtime
// internals. Propose recomputes on commit; there are no parameter-sized proposals.
struct OptimizerStoragePlan {
  int64_t elements,physical_owners,first,second,maximum,bytes;
};
inline OptimizerStoragePlan optimizer_storage_plan(int64_t elements,int64_t owners,int64_t tasks,
    int64_t groups,bool adam,bool momentum,bool amsgrad){
  if(elements<1||owners<0||tasks<0||groups<0)throw std::invalid_argument("invalid optimizer storage shape");
  const auto physical=std::max<int64_t>(1,owners),first=adam||momentum?elements:1,
    second=adam?elements:1,maximum=adam&&amsgrad?elements:1;
  int64_t bytes=0;
  auto add=[&](int64_t count,int64_t width){bytes=resident_bytes_add(bytes,resident_bytes_multiply(count,width));};
  for(auto count:{elements,first,second,maximum})add(count,4);
  // Live/next int64 steps and two FP32 bias corrections per owner.
  add(physical,32);
  add(std::max<int64_t>(1,tasks),64); // One independent finite-error cache line per tile.
  add(physical,8*tide_device::OWNER_FIELDS);
  add(resident_bytes_add(owners,1),8); // Tile prefix, including empty-layout sentinel.
  add(groups,8*tide_device::FLAG_COUNT+4*tide_device::OPTION_COUNT);
  return {elements,physical,first,second,maximum,bytes};
}
}
