#include "scale.h"
#include "tide/fiber_attention.h"
#include <stdexcept>

namespace pdg_scale {
Fixture fixture(const Config& c, const Topology& t) {
  at::NoGradGuard guard;
  Fixture f; auto& g = f.graph; auto& m = f.model;
  const auto n = t.nodes, body = 2*n, width = c.width, period = t.layers+1;
  auto opts = at::TensorOptions().dtype(c.runtime.dtype).device(at::kCPU);
  auto zero = at::zeros({width}, opts), dummy = at::zeros({width, width}, opts);
  auto identity = at::eye(width, opts), one = at::ones({}, opts);
  auto parameter = [&](const std::vector<Index>& shape, bool random = true, bool projection = false) {
    auto value = at::empty(shape, opts);
    if (random) value.normal_(0, .02); else value.fill_(1);
    // Preserve values/RNG order and one owner. Only its physical storage changes.
    if (projection && c.projection_layout == "linear") value = value.t().contiguous().t();
    value.set_requires_grad(true); f.owners.push_back(value); return value;
  };
  const auto base = t.points-t.forced;
  for (Index side = 0; side < 2; ++side) {
    for (Index r = 0; r < base; ++r) g.regions.push_back({t.budget, true, true, "proposal", "lh-count-affect-v1"});
    g.regions.push_back({t.forced});
    for (Index v = 0; v < n; ++v) {
      const auto region = v < t.forced ? base : v < t.points ? v-t.forced : (v-t.points)/t.local;
      tide::Node node{side*(base+1)+region};
      node.memory = "lh-fiber-attention-all-softmax-repeat-v1";
      node.query_heads = node.kv_heads = 4;
      node.clear = true; node.full = "lh-silu-rms-v1"; node.readout = "norm-fp64-v1";
      node.state_clock = {period, 0, t.layers}; node.emit_period = period;
      node.emission = c.emission == "slot" ? "slot_affine" : "broadcast";
      g.nodes.push_back(node);
    }
  }
  tide::Node read{static_cast<Index>(g.regions.size())};
  read.memory = "lh-fiber-attention-all-softmax-repeat-v1";
  read.query_heads = read.kv_heads = 4; read.full = "lh-identity-identity-v1";
  read.state_clock = {period, t.layers, 1}; g.nodes.push_back(read); g.regions.push_back({1});
  g.inputs = {0}; g.outputs = {body}; g.layout = tide::PortLayout{}; g.source_domain = tide::SourceDomain{};
  std::vector<Index> logical_in(body+1), logical_out(body+1), physical_in(body+1);
  std::vector<std::vector<Index>> rows(body);
  auto add = [&](Index source, Index target, Index logical_target, Index logical_source, Index phase, Index delay) {
    g.edges.push_back({source, target, delay});
    g.layout->edge_target.push_back(physical_in[target]++);
    g.layout->edge_source.push_back(rows[source].size());
    g.source_domain->edge_target.push_back(logical_target);
    rows[source].push_back(logical_source); g.nodes[source].emit_phases.push_back(phase);
  };
  for (const auto& edge : t.edges) {
    const auto input = logical_in[edge.target]++, output = logical_out[edge.source]++;
    for (Index phase = 0; phase < t.layers; ++phase)
      add(edge.source, edge.target, input, output, phase, phase == t.layers-1 ? 2 : 1);
  }
  for (Index phase = 0; phase < t.layers; ++phase) add(n, body, phase, -1, phase, t.layers-phase);
  logical_in[body] = t.layers;
  g.layout->input = {physical_in[0]++}; g.source_domain->input = {logical_in[0]++}; g.layout->output = {0};
  for (Index v = 0; v <= body; ++v) {
    tide::NodeWeights w{zero, dummy, zero, zero};
    w.kernel = tide::make_fiber_attention_kernel(g.nodes[v], logical_in[v], c.attention_packing,
                                                c.fiber_pooling, c.fiber_cache, c.attention_layout);
    // Non-learned schema scaffolding is shared and excluded from the model count.
    w.extra["fiber_qkv"] = parameter({width, 3*width}, true, true);
    w.extra["fiber_out"] = parameter({width, width}, true, true);
    w.extra["fiber_qkv_bias"] = at::zeros({3*width}, opts);
    w.extra["fiber_out_bias"] = zero;
    w.extra["fiber_decay"] = at::full({}, .01, opts);
    w.extra["fiber_pool"] = parameter({logical_in[v]}, false);
    if (v < body) {
      w.extra["lh_norm_weight"] = parameter({width}, false);
      auto weight = logical_out[v] ? parameter({logical_out[v]*width, width}) : at::Tensor();
      if (c.emission == "row") {
        if (weight.defined()) w.extra["row_emit_weight"] = weight;
        w.full_kernel = row_emit(rows[v], g.nodes[v].emit_phases, period, logical_out[v]);
      } else {
        // The comparison path uses the existing independent per-slot Full kernel.
        for (size_t slot = 0; slot < rows[v].size(); ++slot) {
          const auto logical = rows[v][slot];
          w.extra["emit_w_"+std::to_string(slot)] = logical < 0 ? identity : weight.slice(0, logical*width, (logical+1)*width).t();
          w.extra["emit_b_"+std::to_string(slot)] = zero;
        }
      }
    }
    m.nodes.push_back(std::move(w));
  }
  m.input_scale = m.output_scale = {one};
  m.agg_scale.assign(g.edges.size(), one); m.edge_scale = m.agg_scale;
  f.embedding = parameter({c.vocab, width}); f.head = parameter({c.vocab, width});
  Index count = 0; for (const auto& p : f.owners) count += p.numel();
  const auto expected = (4*(body+1)+static_cast<Index>(t.edges.size()))*width*width
      +body*width+static_cast<Index>(t.edges.size())+1+t.layers+2*c.vocab*width;
  if (count != expected) throw std::logic_error("scale parameter accounting mismatch");
  f.inventory = {{"parameters", double(count)}, {"parameter_bytes", double(count)*(c.runtime.dtype == at::kDouble ? 8 : 4)},
    {"parameter_owners", double(f.owners.size())}, {"static_nodes_per_cortex", double(n)},
    {"body_nodes", double(body)}, {"pdg_nodes", double(g.nodes.size())}, {"logical_edges", double(t.edges.size())},
    {"physical_edges", double(g.edges.size())}, {"body_ticks_per_token", double(t.layers)},
    {"nominal_leaf_activation", double(t.budget)/t.local}};
  return f;
}
} // namespace pdg_scale
