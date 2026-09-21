#include "tide/ports.h"
#include <stdexcept>

namespace tide {
namespace {
PortIndex index_ports(Index nodes, const std::vector<Index>& edges, const std::vector<Index>& ports,
                      const std::vector<Index>& edge_slots, const std::vector<Index>& port_slots) {
  if (edges.size() != edge_slots.size() || ports.size() != port_slots.size())
    throw std::invalid_argument("local slot layout size mismatch");
  PortIndex result;
  result.offsets.assign(nodes + 1, 0);
  for (auto node : edges) ++result.offsets[node + 1];
  for (auto node : ports) ++result.offsets[node + 1];
  for (Index v = 0; v < nodes; ++v) result.offsets[v + 1] += result.offsets[v];
  result.bindings.assign(edges.size() + ports.size(), {-1, -1});
  auto assign = [&](const auto& owners, const auto& slots, Index kind) {
    for (size_t id = 0; id < owners.size(); ++id) {
      const auto node = owners[id], slot = slots[id];
      const auto start = result.offsets[node], count = result.offsets[node + 1] - start;
      if (slot < 0 || slot >= count || result.bindings[start + slot].kind != -1)
        throw std::invalid_argument("local slots must be a bijection for each node/direction");
      result.bindings[start + slot] = {kind, static_cast<Index>(id)};
    }
  };
  assign(edges, edge_slots, 1); assign(ports, port_slots, 0);
  return result;
}
}  // namespace
void compile_ports(Graph& g) {
  std::vector<Index> source, target;
  for (const auto& e : g.edges) { source.push_back(e.source); target.push_back(e.target); }
  if (!g.layout) {
    PortLayout layout;
    std::vector<Index> incoming(g.nodes.size(), 0), outgoing(g.nodes.size(), 0);
    auto allocate = [](const auto& owners, auto& counts) {
      std::vector<Index> slots;
      for (auto node : owners) slots.push_back(counts[node]++);
      return slots;
    };
    layout.input = allocate(g.inputs, incoming);
    layout.edge_target = allocate(target, incoming);
    layout.edge_source = allocate(source, outgoing);
    layout.output = allocate(g.outputs, outgoing);
    g.layout = std::move(layout);
  }
  const auto& layout = *g.layout;
  g.incoming_ports = index_ports(g.nodes.size(), target, g.inputs, layout.edge_target, layout.input);
  g.outgoing_ports = index_ports(g.nodes.size(), source, g.outputs, layout.edge_source, layout.output);
}
}  // namespace tide
