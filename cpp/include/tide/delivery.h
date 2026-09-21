#pragma once
#include "tide/types.h"
#include <limits>
#include <stdexcept>

namespace tide {
// Resolve only present slots; a numerical zero still creates a real record.
template<class Message, class OutputFn>
void deliver(const Graph& g, const Model& m, const Event& e, Message&& message, OutputFn&& output) {
  const auto start = g.outgoing_ports.offsets[e.node];
  for (const auto& emission : e.emitted) {
    const auto& binding = g.outgoing_ports.bindings[start + emission.slot];
    const auto id = binding.id;
    if (binding.kind == 1) {
      const auto& edge = g.edges[id];
      if (e.time > std::numeric_limits<Index>::max() - edge.delay) throw std::overflow_error("logical time overflow");
      message(Atom{e.batch, edge.target, e.time + edge.delay, 1, id, e.time, emission.value * m.edge_scale[id]});
    } else output(Output{e.batch, e.time, id, emission.value * m.output_scale[id]});
  }
}
}  // namespace tide
