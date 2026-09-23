#pragma once
#include "tide/packed.h"

namespace tide {
struct FiberRows {
  Tensor values;
  std::vector<Index> offsets{0}, slots;
};
// Sorted local slots for attention; Aggregate's canonical fold order is unchanged.
FiberRows fiber_rows(const ContentViews&);
} // namespace tide
