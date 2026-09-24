#pragma once
#include "tide/types.h"

namespace tide {
// Independent events of one node. Clients own scheduling and initialize old.
// Optional contents is numeric packed Aggregate storage, never a semantic root.
void evaluate_state(const Model&, std::vector<Event>&, const std::vector<size_t>&,
                    bool packed, const Tensor& contents = {});
}
