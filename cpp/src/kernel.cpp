#include "tide/kernel.h"
#include "tide/full.h"
#include "tide/aggregate.h"
#include "tide/read.h"
#include "tide/next.h"
#include "tide/region.h"
#include "tide/fiber_attention.h"

namespace tide {
std::vector<State> StateKernel::batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                                    const std::vector<Index>& times, const ContentViews& views) const {
  std::vector<State> states;
  for (size_t i = 0; i < old.size(); ++i) states.push_back(step(w, old[i], views[i].with_value(h[i]), times[i]));
  return states;
}
std::vector<State> StateKernel::sequence(const NodeWeights& w, const State& initial, const Tensor& h,
                                       const std::vector<Index>& times, const ContentViews& views) const {
  State state = initial;
  std::vector<State> states;
  for (size_t i = 0; i < times.size(); ++i) {
    state = step(w, state, views[i].with_value(h[i]), times[i]); states.push_back(state);
  }
  return states;
}
State StateKernel::reset(const State& state) const {
  auto result = state; result.value = state.value * 0;
  for (auto& [name, value] : result.slots) value = value * 0;
  return result;
}
Tensor affine_scan(Tensor a, Tensor b, const Tensor& initial) {
  const auto n = b.size(0);
  for (Index stride = 1; stride < n; stride *= 2) {
    b = at::cat({b.slice(0, 0, stride), b.slice(0, stride) + a.slice(0, stride) * b.slice(0, 0, n - stride)});
    a = at::cat({a.slice(0, 0, stride), a.slice(0, stride) * a.slice(0, 0, n - stride)});
  }
  return b + a * initial;
}
std::shared_ptr<const StateKernel> make_attention_kernel(Index, Index, Index);
void configure_model(const Graph& g, Model& m) {
  if (g.nodes.size() != m.nodes.size()) throw std::invalid_argument("node weight count mismatch");
  if (m.regions.empty()) m.regions.resize(g.regions.size());
  if (m.regions.size() != g.regions.size()) throw std::invalid_argument("region weight count mismatch");
  for (size_t r = 0; r < m.regions.size(); ++r)
    if (!m.regions[r].kernel) m.regions[r].kernel = make_region_kernel(g.regions[r]);
  for (size_t i = 0; i < m.nodes.size(); ++i) {
    if (!m.nodes[i].kernel) {
      const auto& n = g.nodes[i];
      m.nodes[i].kernel = !n.identity && is_fiber_attention_profile(n.memory)
                        ? make_fiber_attention_kernel(n, g.incoming_ports.offsets[i+1]-g.incoming_ports.offsets[i])
                        : !n.identity && n.memory == "attention" ? make_attention_kernel(n.query_heads, n.kv_heads, n.window)
                                                              : make_state_kernel(n.identity ? "identity" : n.memory);
    }
    m.nodes[i].full_kind = g.nodes[i].identity ? "identity" : g.nodes[i].full;
    if (!m.nodes[i].full_kernel) m.nodes[i].full_kernel = make_full_kernel(g.nodes[i]);
    if (!m.nodes[i].next_kernel) m.nodes[i].next_kernel = make_next_kernel(g.nodes[i]);
    if (!m.nodes[i].read_kernel) m.nodes[i].read_kernel = make_read_kernel(g.nodes[i]);
    if (!m.nodes[i].aggregate_kernel) m.nodes[i].aggregate_kernel = make_aggregate_kernel(g.nodes[i]);
  }
}
}  // namespace tide
