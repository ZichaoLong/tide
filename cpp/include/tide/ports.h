#pragma once
#include "tide/types.h"

namespace tide {
// Compilation resolves absent layouts and validates a per-node slot bijection.
void compile_ports(Graph&);
}  // namespace tide
