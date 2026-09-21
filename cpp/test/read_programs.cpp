#include "tide/stream.h"
#include "tide/frontier.h"
#include "tide/read.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace {
using namespace tide;
class ClockRead final : public ReadKernel {
 public:
  Tensor step(const NodeWeights& w, const ReadInput& r) const override {
    auto value = r.content.contributions[0].value.sum() + r.time;
    for (const auto& s : r.content.sources) value = value + (s.atom.position+1)*s.atom.value.sum();
    if (r.state) value = value + r.state->value.sum() + r.state->last_time + r.state->observations;
    return w.extra.at("read_gain") * value;
  }
  void validate_weights(const NodeWeights& w) const override {
    if (!w.extra.count("read_gain")) throw std::invalid_argument("missing Read gain");
  }
};
}  // namespace

void check_read_programs(const at::TensorOptions& options) {
  using namespace tide;
  for (const std::string mode : {"content", "old", "proposal"}) for (int algorithm : {0, 1, 2}) {
    Graph g; g.nodes = {{0}}; g.nodes[0].readout = "clock-read-v1";
    g.regions = {{1, true, true, mode}}; g.inputs = {0}; g.outputs = {0}; g.compile();
    Model m; m.nodes = {{at::zeros({2}, options).set_requires_grad(true), at::zeros({2, 2}, options),
                       at::zeros({2}, options), at::ones({2}, options)}};
    auto gain = at::full({}, 2, options).set_requires_grad(true);
    m.nodes[0].extra["read_gain"] = gain; m.nodes[0].read_kernel = std::make_shared<ClockRead>();
    m.input_scale = m.output_scale = {at::ones({}, options)};
    auto x = at::ones({2}, options).set_requires_grad(true), y = at::full({2}, 3, options).set_requires_grad(true);
    auto initial = at::full({2}, 2, options).set_requires_grad(true), other = at::zeros({2}, options).set_requires_grad(true);
    Continuation q; q.identity = g.identity; q.batch_size = 2;
    q.states[{0, 0}] = {initial}; q.states[{1, 0}] = {other};
    std::vector<External> xs{{0, 0, 0, 1, x}, {0, 0, 1, 4, y}, {1, 0, 0, 1, other}, {1, 0, 1, 4, other}};
    Options runtime; runtime.workers = algorithm == 0 ? 1 : 2; runtime.packed = algorithm != 0;
    auto execute = [&] { return algorithm == 2 ? Frontier(g, m, runtime).run(q, xs, 5, 5)
                                              : Streaming(g, m, runtime).run(q, xs, 5, 5); };
    auto result = execute();
    auto loss = result.trace[0].descriptor + result.trace[2].descriptor;
    const double expected = mode == "content" ? 54 : mode == "old" ? 72 : 94;
    if (!at::equal(loss, at::full({}, expected, options))) throw std::runtime_error("custom Read mode forward mismatch");
    auto grad = torch::autograd::grad({loss}, {x, y, initial, m.nodes[0].decay, gain, other}, {}, false, false, true);
    const auto dx = mode == "content" ? 4. : mode == "old" ? 6. : 7.;
    if (!at::equal(grad[0], at::full_like(x, dx)) || !at::equal(grad[1], at::full_like(y, mode == "proposal" ? 8 : 6))
        || !at::equal(grad[4], at::full_like(gain, expected/2)) || grad[5].defined())
      throw std::runtime_error("custom Read mode isolated VJP mismatch");
    if (mode == "content") {
      if (grad[2].defined() || grad[3].defined()) throw std::runtime_error("content Read exposed state");
    } else if (!at::equal(grad[2], at::full_like(initial, mode == "old" ? 3 : 1.5))
               || !at::equal(grad[3], at::full_like(initial, mode == "old" ? 1 : 2.5)))
      throw std::runtime_error("custom Read state VJP mismatch");
    if (algorithm && result.stats.at("read_scalar_batch_steps") != 4) throw std::runtime_error("Read fallback count mismatch");
    at::NoGradGuard guard;
    auto inference = execute();
    if (!at::equal(inference.trace[0].descriptor + inference.trace[2].descriptor, loss))
      throw std::runtime_error("custom Read inference mismatch");
    if (inference.stats["semantic_read_replays"] != 0) throw std::runtime_error("inference replayed Read");
  }
}
