#pragma once
#include <cstdint>

namespace tide {
// Valid global phases are [first, first+count) in each period. No per-node heap.
struct StateClock {
  std::int64_t period = 1, first = 0, count = 1;
  void validate() const;
  std::int64_t to_local(std::int64_t) const;  // -1 is the initial-state sentinel.
  std::int64_t to_global(std::int64_t) const;
  std::int64_t cut(std::int64_t) const;
  bool operator==(const StateClock& b) const { return period == b.period && first == b.first && count == b.count; }
  bool operator!=(const StateClock& b) const { return !(*this == b); }
};
}  // namespace tide
