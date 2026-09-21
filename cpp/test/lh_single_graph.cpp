#include "lh_single_graph.h"
#include <algorithm>

namespace lh_iocortex {
SingleGraph::SingleGraph(const Fixture& f)
    : body_clock{f.policy.layers()+1, 0, f.policy.layers()},
      read_clock{f.policy.layers()+1, f.policy.layers(), 1},
      body_nodes(f.body.nodes.size()), body_regions(f.body.regions.size()),
      body_edges(f.body.edges.size()), layers(f.policy.layers()) {
  graph.nodes = f.body.nodes; graph.regions = f.body.regions;
  graph.inputs = f.body.inputs; graph.outputs = {body_nodes};
  auto read = f.readout.nodes.at(0); read.region = body_regions; read.state_clock = read_clock;
  graph.nodes.push_back(read); graph.regions.push_back(f.readout.regions.at(0));
  graph.layout = tide::PortLayout{}; graph.source_domain = tide::SourceDomain{};
  auto& ports = *graph.layout; auto& domain = *graph.source_domain;
  std::vector<Index> in(body_nodes+1), out(body_nodes+1);
  output_origin.resize(body_nodes);
  model.nodes = f.model.nodes; model.nodes.push_back(f.read_model.nodes.at(0));
  for (Index v = 0; v < body_nodes; ++v) {
    graph.nodes[v].state_clock = body_clock; graph.nodes[v].emit_period = layers+1;
    graph.nodes[v].emit_phases.clear();
    auto& extra = model.nodes[v].extra;
    for (auto it = extra.begin(); it != extra.end();)
      if (it->first.rfind("emit_w_", 0) == 0 || it->first.rfind("emit_b_", 0) == 0) it = extra.erase(it);
      else ++it;
  }
  auto add = [&](Index source, Index target, Index delay, Index phase, Index old_slot, Index logical,
                 Index origin, const at::Tensor& send, const at::Tensor& receive) {
    const auto slot = out[source]++;
    graph.edges.push_back({source, target, delay}); edge_origin.push_back(origin);
    ports.edge_source.push_back(slot); ports.edge_target.push_back(in[target]++);
    domain.edge_target.push_back(logical); graph.nodes[source].emit_phases.push_back(phase);
    output_origin[source].push_back(old_slot);
    for (const auto& prefix : {"emit_w_", "emit_b_"})
      model.nodes[source].extra[std::string(prefix)+std::to_string(slot)] =
          f.model.nodes[source].extra.at(std::string(prefix)+std::to_string(old_slot));
    model.edge_scale.push_back(send); model.agg_scale.push_back(receive);
  };
  for (Index e = 0; e < body_edges; ++e) {
    const auto& edge = f.body.edges[e];
    require(edge.delay == 1, "single IOCortex fixture expects unit body delays");
    for (Index phase = 0; phase < layers; ++phase)
      add(edge.source, edge.target, phase == layers-1 ? 2 : 1, phase,
          f.body.layout->edge_source[e], f.body.source_domain->edge_target[e], e,
          f.model.edge_scale[e], f.model.agg_scale[e]);
  }
  for (Index phase = 0; phase < layers; ++phase)
    add(f.body.outputs.at(0), body_nodes, layers-phase, phase, f.body.layout->output.at(0),
        f.readout.source_domain->input.at(phase), -1, f.model.output_scale.at(0), f.read_model.input_scale.at(phase));
  for (size_t p = 0; p < graph.inputs.size(); ++p) {
    ports.input.push_back(in[graph.inputs[p]]++); domain.input.push_back(f.body.source_domain->input[p]);
  }
  ports.output = {0}; graph.compile();
  require(std::equal(f.body.source_counts.begin(), f.body.source_counts.end(), graph.source_counts.begin()),
          "single IOCortex changed the logical body source domain");
  model.input_scale = f.model.input_scale; model.output_scale = f.read_model.output_scale;
}

tide::Atom SingleGraph::body_atom(tide::Atom a) const {
  a.time = body_clock.to_local(a.time);
  if (a.kind) {
    a.source = edge_origin.at(a.source); require(a.source >= 0, "readout wire in body projection");
    a.position = body_clock.to_local(a.position);
  }
  return a;
}
namespace {
tide::State local(tide::State s, const tide::StateClock& clock) {
  s.last_time = clock.to_local(s.last_time); return s;
}
tide::History local(tide::History h, const tide::StateClock& clock) {
  h.last_time = clock.to_local(h.last_time); return h;
}
void sort_atoms(std::vector<tide::Atom>& xs) {
  std::sort(xs.begin(), xs.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
}
}  // namespace
tide::Result SingleGraph::body_view(const tide::Result& r, const tide::Graph& body) const {
  tide::Result result; auto& q = result.continuation;
  q.identity = body.identity; q.batch_size = r.continuation.batch_size; q.cut = body_clock.cut(r.continuation.cut);
  for (const auto& [owner, s] : r.continuation.states) if (owner.second < body_nodes)
    q.states[owner] = local(s, body_clock);
  for (const auto& [owner, h] : r.continuation.history) if (owner.second < body_regions)
    q.history[owner] = local(h, body_clock);
  for (const auto& [owner, last] : r.continuation.ledger) q.ledger[owner] = {last.first, body_clock.to_local(last.second)};
  for (const auto& a : r.continuation.pending) if (edge_origin.at(a.source) >= 0) q.pending.push_back(body_atom(a));
  for (auto e : r.trace) if (e.node < body_nodes) {
    e.time = body_clock.to_local(e.time); e.history = local(e.history, body_clock);
    e.old = local(e.old, body_clock); e.proposed_state = local(e.proposed_state, body_clock);
    e.comparison_state = local(e.comparison_state, body_clock); e.next_state = local(e.next_state, body_clock);
    for (auto& a : e.fiber) a = body_atom(a);
    sort_atoms(e.fiber);
    for (auto& source : e.sources) source.atom = body_atom(source.atom);
    for (auto& slot : e.emitted) slot.slot = output_origin[e.node].at(slot.slot);
    std::sort(e.emitted.begin(), e.emitted.end(), [](const auto& a, const auto& b) { return a.slot < b.slot; });
    result.trace.push_back(std::move(e));
  }
  for (const auto& a : r.messages) {
    if (edge_origin.at(a.source) >= 0) result.messages.push_back(body_atom(a));
    else result.outputs.push_back({a.batch, body_clock.to_local(a.position), 0, a.value});
  }
  return result;
}
tide::Continuation SingleGraph::read_view(const tide::Continuation& raw, const tide::Graph& read) const {
  tide::Continuation q; q.identity = read.identity; q.batch_size = raw.batch_size; q.cut = read_clock.cut(raw.cut);
  for (const auto& [owner, s] : raw.states) if (owner.second == body_nodes) q.states[{owner.first, 0}] = local(s, read_clock);
  for (const auto& [owner, h] : raw.history) if (owner.second == body_regions) {
    auto history = local(h, read_clock);
    for (auto& [name, values] : history.node_maps) {
      std::map<Index, Index> remapped; for (auto [node, value] : values) remapped[node-body_nodes] = value;
      values = std::move(remapped);
    }
    q.history[{owner.first, 0}] = history;
  }
  // No External.position exists for internal readout wires. This is a state/
  // history view, intentionally not the two-graph readout's continuation.
  return q;
}
std::vector<tide::Output> SingleGraph::buffer(const tide::Continuation& q) const {
  std::vector<tide::Output> result;
  for (const auto& a : q.pending) if (edge_origin.at(a.source) < 0)
    result.push_back({a.batch, body_clock.to_local(a.position), 0, a.value});
  return result;
}
}  // namespace lh_iocortex
