#pragma once
#include "tide/full.h"

namespace tide {
// Shared dense row projection, with phase-specific physical aliases. Logical
// index -1 exposes fresh directly; other indices choose width-sized slices.
std::shared_ptr<const FullKernel> make_row_emit(std::vector<Index> logical,
    std::vector<Index> phases, Index period, Index targets);
}  // namespace tide
