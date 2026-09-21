#include "tide/stream.h"
#include "tide/frontier.h"
#include "tide/next.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace {
using namespace tide;
class ClockNext : public NextKernel {
 public:
  State step(const NodeWeights& w, const NextInput& r) const override {
    const auto& gain = w.extra.at("next_gain");
    auto value = r.old.value + gain*(r.comparison.value+r.time*r.content.value+(r.active ? r.control : -r.control));
    Tensor source;
    for (const auto& s : r.content.sources) {
      auto term = (s.slot+1)*(s.atom.position+1)*s.atom.value*s.scale;
      source = source.defined() ? source+term : term;
    }
    return {value, r.time, r.old.observations+1, {{"memory", r.old.slots.at("memory")+gain*source}}};
  }
  void validate_weights(const NodeWeights& w) const override {
    if (!w.extra.count("next_gain")) throw std::invalid_argument("missing Next gain");
  }
};
class BrokenNext final : public ClockNext {
 public:
  State step(const NodeWeights& w, const NextInput& r) const override {
    auto state = ClockNext::step(w, r); state.last_time = r.time+1; return state;
  }
};
}  // namespace

void check_next_programs(const at::TensorOptions& options) {
  using namespace tide;
  for (bool clear : {false, true}) for (bool encoded : {false, true}) for (int algorithm : {0, 1, 2}) {
    Graph g; g.nodes = {{0, clear}, {0}, {0}}; g.regions = {{1, true, false, "content"}};
    for (auto& n : g.nodes) { n.memory = "ssm"; n.next_state = "clock-next-v1"; }
    g.inputs = {0, 1}; g.outputs = {0, 1};
    Model m;
    auto gain = at::full({}, 2, options).set_requires_grad(true);
    auto program = std::make_shared<ClockNext>();
    for (int v = 0; v < 3; ++v) {
      NodeWeights w{at::zeros({1}, options), at::zeros({1, 1}, options), at::zeros({1}, options), at::zeros({1}, options)};
      w.next_kernel = program; w.extra["next_gain"] = gain;
      for (const auto name : {"ssm_dt", "ssm_b", "ssm_c"}) w.extra[name] = at::zeros({1, 1}, options);
      w.extra["ssm_a"] = at::zeros({1}, options); w.extra["ssm_skip"] = at::zeros({1}, options).set_requires_grad(true);
      m.nodes.push_back(std::move(w));
    }
    m.input_scale = m.output_scale = {at::ones({}, options), at::ones({}, options)};
    if (encoded) {
      for (Index port = 0; port < 2; ++port) {
        g.nodes.push_back({port+1, false, true}); g.regions.push_back({1});
        g.edges.push_back({port+3, port, 1}); g.inputs[port] = port+3; g.origins.push_back({port, port, 3});
        m.nodes.push_back({at::zeros({1}, options), at::zeros({1, 1}, options), at::zeros({1}, options), at::zeros({1}, options)});
      }
      m.agg_scale = m.edge_scale = {at::ones({}, options), at::ones({}, options)};
    }
    g.compile();
    auto x = at::tensor({1., 3., 2., 0.}, options).reshape({2, 2}).set_requires_grad(true);
    auto other = at::ones({2, 2}, options).set_requires_grad(true);
    auto idle = at::full({1}, 7, options).set_requires_grad(true);
    Continuation q; q.identity = g.identity; q.batch_size = 2;
    q.states[{0, 2}] = {idle, -1, 0, {{"memory", idle}}};
    std::vector<External> xs;
    for (Index b = 0; b < 2; ++b) for (Index v = 0; v < 2; ++v) for (Index i = 0; i < 2; ++i)
      xs.push_back({b, v, i, 3*i+(encoded ? 0 : 1), (b == 0 ? x : other)[i][v].reshape({1})});
    Options runtime; runtime.packed = algorithm != 0; runtime.workers = algorithm == 0 ? 1 : 3;
    auto execute = [&] { return algorithm == 2 ? Frontier(g, m, runtime).run(q, xs, 5, 5)
                                              : Streaming(g, m, runtime).run(q, xs, 5, 5); };
    auto result = execute();
    Tensor loss = at::zeros({}, options);
    for (Index v = 0; v < 2; ++v) {
      const auto& state = result.continuation.states.at({0, v});
      loss = loss+state.value.sum()+state.slots.at("memory").sum();
      if (state.last_time != 4 || state.observations != 2) throw std::runtime_error("custom Next clocks mismatch");
    }
    if (!at::equal(loss, at::full({}, clear ? 10 : 40, options))) throw std::runtime_error("custom Next forward mismatch");
    auto grad = torch::autograd::grad({loss}, {x, gain, other, idle, m.nodes[0].extra.at("ssm_skip"), m.nodes[1].extra.at("ssm_skip")},
                                     {}, false, false, true);
    auto expected = at::tensor({clear ? 0. : 4., 4., clear ? 0. : 12., 12.}, options).reshape({2, 2});
    if (!at::equal(grad[0], expected) || !at::equal(grad[1], at::full_like(gain, clear ? 5 : 20))
        || grad[2].defined() || grad[3].defined()
        || !at::equal(grad[4], at::full({1}, clear ? 0 : 6, options)) || !at::equal(grad[5], at::full({1}, 6, options)))
      throw std::runtime_error("custom Next isolated VJP mismatch");
    for (const auto& e : result.trace) if (e.node == 2) throw std::runtime_error("idle node executed Next");
    if (!at::equal(result.continuation.states.at({0, 2}).value, idle)) throw std::runtime_error("idle state changed");
    if (algorithm == 2 && !encoded && (result.stats["state_blocks"] != 0 || result.stats["full_blocks"] == 0))
      throw std::runtime_error("custom Next was incorrectly prefilled");
    {
      at::NoGradGuard guard;
      auto inference = execute();
      for (Index v = 0; v < 2; ++v) {
        const auto& a = result.continuation.states.at({0, v}); const auto& b = inference.continuation.states.at({0, v});
        if (!at::equal(a.value, b.value) || !at::equal(a.slots.at("memory"), b.slots.at("memory")))
          throw std::runtime_error("custom Next inference mismatch");
      }
    }
    m.nodes[0].next_kernel = std::make_shared<BrokenNext>();
    bool rejected = false;
    try { execute(); } catch (const std::invalid_argument& e) {
      if (std::string(e.what()).find("Next returned invalid state clocks") == std::string::npos) throw;
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("future Next clock was accepted");
  }
}
