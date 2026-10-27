#pragma once
#include "tide/types.h"

namespace tide {
// Explicit pure State subexpressions, scoped to a legal current event block.
// Not mathematical Content/State, and never serialized in a continuation.
void prepare_shared_state(const Model&,std::vector<Event>&,std::map<std::string,Index>&);
}
