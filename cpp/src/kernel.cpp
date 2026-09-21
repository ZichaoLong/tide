#include "tide/kernel.h"
#include "tide/full.h"

namespace tide {
std::vector<State> StateKernel::batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                                    const std::vector<Index>& times, const FiberViews& fibers) const {
  std::vector<State> states;
  for (size_t i = 0; i < old.size(); ++i) states.push_back(step(w, old[i], h[i], times[i], *fibers[i]));
  return states;
}
std::vector<State> StateKernel::sequence(const NodeWeights& w, const State& initial, const Tensor& h,
                                       const std::vector<Index>& times, const FiberViews& fibers) const {
  State state = initial;
  std::vector<State> states;
  for (size_t i = 0; i < times.size(); ++i) {
    state = step(w, state, h[i], times[i], *fibers[i]); states.push_back(state);
  }
  return states;
}
Tensor StateKernel::read(const NodeWeights& w, const State&, const State& proposal, const Tensor&,
                         Index, const std::vector<Atom>&) const { return (proposal.value * w.read).sum(-1); }
Tensor StateKernel::read_batch(const NodeWeights& w, const std::vector<State>& old, const std::vector<State>& proposals,
                               const Tensor& h, const std::vector<Index>& times, const FiberViews& fibers) const {
  std::vector<Tensor> values;
  for (size_t i = 0; i < proposals.size(); ++i) values.push_back(read(w, old[i], proposals[i], h[i], times[i], *fibers[i]));
  return at::stack(values);
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
  for (size_t i = 0; i < m.nodes.size(); ++i) {
    if (!m.nodes[i].kernel) {
      const auto& n = g.nodes[i];
      m.nodes[i].kernel = !n.identity && n.memory == "attention" ? make_attention_kernel(n.query_heads, n.kv_heads, n.window)
                                                              : make_state_kernel(n.identity ? "identity" : n.memory);
    }
    m.nodes[i].full_kind = g.nodes[i].identity ? "identity" : g.nodes[i].full;
    if (!m.nodes[i].full_kernel) m.nodes[i].full_kernel = make_full_kernel(g.nodes[i]);
  }
}
}  // namespace tide
