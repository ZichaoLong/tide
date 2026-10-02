#pragma once
#include "graph_vjp.h"
#include "state_owner_tape.h"

namespace tide::device_online {
// Shape views only, before the retained owner clones numerical records. Valid
// prefix positions stay unchanged; an empty journal keeps one unread padding
// row for kernels requiring positive capacities. No KV/queue compaction here.
void compact_retained_journals(ReverseTape&);
void compact_retained_journals(StateOwnerTape&);
} // namespace tide::device_online
