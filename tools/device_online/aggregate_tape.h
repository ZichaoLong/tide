#pragma once
#include <ATen/ATen.h>

namespace tide::device_online {
struct AggregateTape {
  at::Tensor kinds,lengths,weights;
  int64_t slots=0;
};
struct AggregateVjp {
  at::Tensor values,connected,chunks;
};
} // namespace tide::device_online
