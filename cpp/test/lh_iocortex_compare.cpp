#include "lh_iocortex_fixture.h"
#include "lh_cache_oracle.h"
#include "tide/fiber_attention.h"
#include "tide/lazy_add.h"
#include <set>

namespace lh_iocortex {
namespace AL = AccumulateLocal;
void close(const at::Tensor& a, const at::Tensor& b, const std::string& message) {
  const bool fp64 = a.scalar_type() == at::kDouble;
  require(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type()
          && at::allclose(a, b, fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6), message);
}
Cache original_cache(const PtrBatchPtrBaseHidden& ptr, Index b, at::TensorOptions opts) {
  if (auto h = dynamic_cast<AL::BatchPtrTensorHidden*>(ptr.get()))
    return {{"hidden", (AL::TensorHidden::multi_batch_forward_mode ? h->cache[b] : h->hptrs[b]->data).clone()}};
  auto h = dynamic_cast<AL::BatchPtrKVHidden*>(ptr.get());
  require(h != nullptr, "unknown original IOCortex hidden type");
  return lh_oracle::cache(*h, b, opts);
}
ObservedSelector::ObservedSelector(Index batch, const std::shared_ptr<GraphConfig>& cfg, const Policy& p,
                                   const VPtrBatchPtrBaseHidden& h, at::TensorOptions options)
    : BaseSelector(batch, 2, cfg), original(batch, 2, cfg, p.lead, p.original_mode == 2), hidden(h), opts(options) {}
VPtrBatchSignals ObservedSelector::select(const VPtrBatchSignals& inputs) {
  Tick tick;
  for (Index v = 0; v < static_cast<Index>(inputs.size()); ++v) if (inputs[v]) {
    for (Index row = 0; row < inputs[v]->sampleids.numel(); ++row) {
      const auto b = inputs[v]->sampleids[row].item<Index>();
      tick.proposals[{b, v}] = inputs[v]->x[row].clone();
      tick.caches[{b, v}] = original_cache(hidden.at(v), b, opts);
    }
  }
  tick.selected = original.select(inputs);  // The real original heap/tensor implementation.
  ticks.push_back(tick);
  return tick.selected;  // Original post-selection mutates these BatchSignals.
}
tide::State empty_state(const tide::Graph& g, const tide::Model& m, Index v) {
  if (g.nodes[v].memory == "lh-add-repeat-v1") return {at::zeros_like(m.nodes[v].bias)};
  const auto slots = g.incoming_ports.offsets[v+1]-g.incoming_ports.offsets[v];
  return tide::make_fiber_attention_kernel(g.nodes[v], slots)->initial(m.nodes[v]);
}
void compare_cache(const tide::NodeWeights& w, const tide::State& s, Index cut,
                   const Cache& expected, const std::string& where) {
  if (expected.count("hidden")) close(tide::decode_add_repeat(w, s, cut), expected.at("hidden"), where+" Add hidden");
  else {
    for (const auto& name : {"key", "value"}) close(s.slots.at(name), expected.at(name), where+" "+name);
    close(tide::decode_fiber_bias(w, s, cut), expected.at("log_bias"), where+" log bias");
  }
}
void compare_states(const Fixture& f, const tide::Continuation& q,
                    const VPtrBatchPtrBaseHidden& ih, const VPtrBatchPtrBaseHidden& oh) {
  for (Index v = 0; v < 2*f.n; ++v) for (Index b = 0; b < Fixture::batch; ++b) {
    auto it = q.states.find({b, v}); auto s = it == q.states.end() ? empty_state(f.body, f.model, v) : it->second;
    auto expected = original_cache((v < f.n ? ih : oh).at(v%f.n), b, f.opts);
    compare_cache(f.model.nodes[v], s, q.cut, expected, "body b="+std::to_string(b)+" v="+std::to_string(v));
  }
}
void compare_counts(const Fixture& f, const tide::Continuation& q,
                    const ObservedSelector& is, const ObservedSelector& os) {
  const Index base = f.cfg->base_num;
  for (Index c = 0; c < 2; ++c) {
    const auto& s = (c ? os : is).original;
    for (Index r = 0; r < base; ++r) for (Index b = 0; b < Fixture::batch; ++b)
      for (Index slot = 0; slot < f.cfg->localnum+(f.policy.lead ? 1 : 0); ++slot) {
        const auto local = f.policy.lead && slot == 0 ? s.s+r : s.point_s+r*f.cfg->localnum+slot-(f.policy.lead ? 1 : 0);
        const auto hi = q.history.find({b, c*(base+1)+r});
        for (const auto& name : {"selected", "affected"}) {
          const bool selected = std::string(name) == "selected";
          const auto& tensor = selected ? s.selectcountT : s.affectcountT;
          const auto& tensors = selected ? s.selectcounts : s.affectcounts;
          Index expected = s.select_by_tensor ? (tensors[r].defined() ? tensors[r][b][slot].item<Index>() : 0)
                                              : tensor[r][b][slot].item<Index>();
          Index actual = 0;
          if (hi != q.history.end()) {
            const auto& values = hi->second.node_maps.at(name); auto vi = values.find(c*f.n+local);
            if (vi != values.end()) actual = vi->second;
          }
          require(actual == expected, "IOCortex selector history mismatch");
        }
      }
  }
}
namespace {
Rows selected_rows(const VPtrBatchSignals& selected) {
  Rows result;
  for (Index v = 0; v < static_cast<Index>(selected.size()); ++v) if (selected[v])
    for (Index row = 0; row < selected[v]->sampleids.numel(); ++row)
      result[{selected[v]->sampleids[row].item<Index>(), v}] = selected[v]->x[row];
  return result;
}
using Wires = std::map<tide::Owner, at::Tensor>;  // (sample, physical edge)
Wires wire_values(const Fixture& f, const VPtrBatchSignals& iacts, const VPtrBatchSignals& oacts) {
  Wires expected;
  for (Index block = 0; block < 4; ++block) {
    auto& original = *f.blocks[block];
    const auto signals = original.BaseEmitToEdges(Fixture::batch, block%2 ? oacts : iacts);
    for (Index edge = 0; edge < original.A.nnz; ++edge) if (signals[edge])
      for (Index row = 0; row < signals[edge]->sampleids.numel(); ++row)
        expected[{signals[edge]->sampleids[row].item<Index>(), f.edge_offsets[block]+edge}] = signals[edge]->x[row];
  }
  return expected;
}
void check_atom(const Fixture& f, const tide::Atom& a, Index send, Wires& expected) {
  require(a.kind == 1 && a.position == send && a.time == send+1
          && a.node == f.body.edges.at(a.source).target, "IOCortex wire coordinates mismatch");
  auto it = expected.find({a.batch, a.source}); require(it != expected.end(), "IOCortex extra/duplicate wire");
  close(a.value, it->second, "IOCortex signaling payload mismatch"); expected.erase(it);
}
}  // namespace
void compare_trace(const Fixture& f, const tide::Result& result, const ObservedSelector& is,
                   const ObservedSelector& os, Index start) {
  std::map<std::tuple<Index, Index, Index>, const tide::Event*> events;
  for (const auto& e : result.trace)
    require(events.emplace(std::make_tuple(e.time, e.batch, e.node), &e).second, "duplicate IOCortex event");
  std::map<Index, Wires> messages;
  std::map<tide::Owner, at::Tensor> outputs;
  for (Index t = start; t < result.continuation.cut; ++t) {
    for (Index c = 0; c < 2; ++c) {
      const auto& tick = (c ? os : is).ticks.at(t); auto selected = selected_rows(tick.selected);
      for (const auto& [owner, proposal] : tick.proposals) {
        auto key = std::make_tuple(t, owner.first, c*f.n+owner.second);
        auto ei = events.find(key); require(ei != events.end(), "missing IOCortex candidate");
        const auto& e = *ei->second; auto active = selected.find(owner);
        require(e.active == (active != selected.end()), "IOCortex selected route mismatch");
        close(e.proposal, proposal, "IOCortex candidate proposal mismatch");
        auto actual_norm = at::norm(e.proposal.to(at::kDouble));
        auto expected_norm = at::norm(proposal.to(at::kDouble));
        close(e.descriptor, actual_norm, "IOCortex FP64 descriptor computation mismatch");
        // Upstream FP32 arithmetic remains FP32 even when Read accumulates in
        // FP64. Bound norm disagreement by the actual candidate perturbation;
        // the selected route above must still agree exactly.
        const auto perturbation = at::norm(e.proposal.to(at::kDouble)-proposal.to(at::kDouble)).item<double>();
        require(std::abs(e.descriptor.item<double>()-expected_norm.item<double>())
                  <= perturbation+1e-10+1e-8*expected_norm.item<double>(), "IOCortex norm perturbation bound exceeded");
        compare_cache(f.model.nodes[e.node], e.proposed_state, t+1, tick.caches.at(owner), "candidate");
        if (e.active) close(e.full, active->second, "IOCortex activation/norm mismatch");
        events.erase(ei);
      }
      if (c) for (const auto& [owner, value] : selected) if (owner.second == 0) outputs[{t, owner.first}] = value;
    }
    messages[t] = wire_values(f, is.ticks[t].selected, os.ticks[t].selected);
  }
  require(events.empty(), "extra IOCortex candidate");
  for (const auto& a : result.messages) {
    require(messages.count(a.position), "IOCortex message send time outside window");
    check_atom(f, a, a.position, messages.at(a.position));
  }
  for (const auto& [time, rows] : messages) require(rows.empty(), "missing IOCortex emission");
  for (const auto& o : result.outputs) {
    auto it = outputs.find({o.time, o.batch});
    require(o.port == 0 && it != outputs.end(), "extra/duplicate IOCortex output");
    close(o.value, it->second, "IOCortex output value mismatch"); outputs.erase(it);
  }
  require(outputs.empty(), "missing IOCortex output");
}
void compare_pending(const Fixture& f, const tide::Continuation& q,
                     const VPtrBatchSignals& iacts, const VPtrBatchSignals& oacts) {
  auto expected = wire_values(f, iacts, oacts);
  for (const auto& a : q.pending) check_atom(f, a, q.cut-1, expected);
  require(expected.empty(), "missing IOCortex pending message");
}
void compare_continuation(const tide::Continuation& a, const tide::Continuation& b) {
  require(a.identity == b.identity && a.cut == b.cut && a.batch_size == b.batch_size && a.ledger == b.ledger,
          "IOCortex continuation identity/clock/ledger mismatch");
  require(a.states.size() == b.states.size() && a.history.size() == b.history.size() && a.pending.size() == b.pending.size(),
          "IOCortex continuation membership mismatch");
  for (const auto& [owner, s] : a.states) {
    const auto& other = b.states.at(owner);
    require(s.last_time == other.last_time && s.observations == other.observations && s.slots.size() == other.slots.size(),
            "IOCortex continuation state clock mismatch");
    close(s.value, other.value, "IOCortex continuation state value mismatch");
    for (const auto& [name, value] : s.slots) close(value, other.slots.at(name), "IOCortex continuation slot mismatch");
  }
  for (const auto& [owner, h] : a.history) {
    const auto& other = b.history.at(owner);
    require(h.last_time == other.last_time && h.scalars == other.scalars && h.node_maps == other.node_maps
            && h.tensors.empty() && other.tensors.empty(), "IOCortex continuation history mismatch");
  }
  // Own key values: Atom::key() itself borrows references to its record.
  std::map<std::tuple<Index, Index, Index, Index, Index, Index>, at::Tensor> pending;
  for (const auto& atom : b.pending) pending.emplace(atom.key(), atom.value);
  for (const auto& atom : a.pending) {
    auto it = pending.find(atom.key()); require(it != pending.end(), "IOCortex continuation pending identity mismatch");
    close(atom.value, it->second, "IOCortex continuation pending value mismatch"); pending.erase(it);
  }
  require(pending.empty(), "IOCortex continuation pending duplicate");
}
}  // namespace lh_iocortex
