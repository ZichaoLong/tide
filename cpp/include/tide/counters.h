#pragma once
#include "tide/types.h"
#include <limits>
#include <stdexcept>

namespace tide {
inline Index increment(Index value, Index amount = 1) {
  if (value < 0 || amount < 0 || value > std::numeric_limits<Index>::max() - amount)
    throw std::invalid_argument("int64 counter overflow or invalid counter");
  return value + amount;
}
}  // namespace tide
