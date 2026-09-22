#pragma once
#include "tide/types.h"
#include "tide/pool.h"

namespace tide {
using NodeEvents = std::map<Index, std::vector<size_t>>;
using RegionEvents = std::map<Owner, std::vector<size_t>>;
std::vector<Event> compact_stream_events(const Graph&, const Model&, const Continuation&,
                                        std::vector<Atom>&, NodeEvents&, RegionEvents&);
void parallel_stream_regions(const Graph&, const Model&, Continuation&, std::vector<Event>&,
                             const RegionEvents&, NodePool&, Index workers, bool trace);
void release_stream_events(std::vector<Event>&, NodePool&, Index workers);
} // namespace tide
