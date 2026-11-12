#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace tide::device_online {
struct BlockDispatchPlan {
  // group, sample, owner, original-time lower bound, unconditional
  std::vector<std::array<int64_t,5>> tasks;
  // source slot, target slot, whether presence (as well as closure) propagates
  std::vector<std::array<int64_t,3>> dependencies;
  int64_t groups=0, batch_capacity=0, journal_capacity=0, tensor_bytes=0;
};
}
