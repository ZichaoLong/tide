#pragma once
#include "tide/types.h"

namespace tide {
// Resolve the optional logical source domain after physical ports are compiled.
void compile_source_domain(Graph&);
}  // namespace tide
