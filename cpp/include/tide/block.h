#pragma once
#include "tide/kernel.h"
#include "tide/pool.h"

namespace tide {
void advance_block_events(const Graph&, const Model&, Continuation&, std::vector<Event>&,
                          const std::vector<std::vector<size_t>>&, const Options&, NodePool&,
                          std::map<std::string, Index>&);
void prefill_states(const Graph&, const Model&, const Continuation&, const Options&, NodePool&,
                    std::vector<Event>&, const std::map<Owner, std::vector<size_t>>&,
                    std::map<std::string, Index>&);
}  // namespace tide
