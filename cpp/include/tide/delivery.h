#pragma once
#include "tide/types.h"
#include <limits>
#include <stdexcept>

namespace tide {
struct Delivery {std::vector<Atom> messages;std::vector<Output> outputs;};
Delivery prepare_delivery(const Graph& g,const Model& m,const std::vector<Event>& events,
                          bool packed,std::map<std::string,Index>& stats);
template<class Message,class OutputFn>
void deliver_batch(const Graph& g,const Model& m,const std::vector<Event>& events,
                   bool packed,std::map<std::string,Index>& stats,Message&& message,OutputFn&& output) {
  auto batch=prepare_delivery(g,m,events,packed,stats);
  for(const auto& atom:batch.messages)message(atom);
  for(const auto& value:batch.outputs)output(value);
}
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
      auto value = (emission.value * m.edge_scale[id].to(emission.value.device())).to(m.nodes[edge.target].bias.device());
      message(Atom{e.batch, edge.target, e.time + edge.delay, 1, id, e.time, value});
    } else output(Output{e.batch, e.time, id, emission.value * m.output_scale[id].to(emission.value.device())});
  }
}
}  // namespace tide
