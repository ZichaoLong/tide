#include "portable_torch/runtime.hpp"
#include "tide/aggregate.h"
#include "tide/stream.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
class ClockFiber final : public tide::AggregateKernel {
 public:
  tide::AggregateResult step(const tide::NodeWeights& w, const tide::AggregateInput& input) const override {
    if (input.slots != 3) throw std::invalid_argument("custom Aggregate slot domain");
    for (size_t i = 1; i < input.sources.size(); ++i)
      if (input.sources[i].atom.key() < input.sources[i-1].atom.key()) throw std::runtime_error("noncanonical source view");
    tide::AggregateResult result;
    for (const auto& source : input.sources) {
      const auto& atom = source.atom;
      auto term = ((source.slot+1)*atom.value*source.scale + (atom.position+atom.kind)) * (input.time+1) * w.extra.at("gain");
      result.value = result.value.defined() ? result.value + term : term;
      result.contributions.push_back({source.slot, term});
    }
    std::sort(result.contributions.begin(), result.contributions.end(), [](const auto& a, const auto& b) { return a.slot < b.slot; });
    return result;
  }
  void validate_weights(const tide::NodeWeights& w, tide::Index slots) const override {
    if (slots != 3 || !w.extra.count("gain")) throw std::invalid_argument("custom Aggregate weights");
  }
};

void check_origins(const at::TensorOptions& options) {
  tide::Graph direct; direct.nodes = {{0}}; direct.nodes[0].aggregation = "clock-fiber-v1";
  direct.edges = {{0, 0, 3}}; direct.regions = {{1}}; direct.inputs = {0, 0}; direct.outputs = {0}; direct.compile();
  tide::Model m;
  m.nodes = {{at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::ones({2}, options)}};
  auto gain = at::full({}, 2, options).set_requires_grad(true);
  m.nodes[0].extra["gain"] = gain; m.nodes[0].aggregate_kernel = std::make_shared<ClockFiber>();
  m.input_scale = {at::ones({}, options), at::ones({}, options)};
  m.agg_scale = m.edge_scale = m.output_scale = {at::ones({}, options)};
  auto encoded = direct; encoded.nodes.push_back({1, false, true}); encoded.regions.push_back({1});
  encoded.edges.insert(encoded.edges.end(), {{1, 0, 1}, {1, 0, 1}}); encoded.inputs = {1};
  encoded.layout = tide::PortLayout{{0, 0, 1}, {2, 0, 1}, {0}, {1}};
  encoded.origins = {{1, 0, 3}, {2, 1, 3}}; encoded.compile();
  if (!direct.origin_index.empty() || encoded.origin_index.size() != encoded.edges.size())
    throw std::runtime_error("optional origin index allocation mismatch");
  auto em = m;
  em.nodes.push_back({at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::zeros({2}, options)});
  em.input_scale.resize(1);
  em.agg_scale = em.edge_scale = {at::ones({}, options), at::ones({}, options), at::ones({}, options)};
  auto x = at::ones({2}, options).set_requires_grad(true), y = at::full({2}, 3, options).set_requires_grad(true);
  tide::Options runtime; runtime.packed = true; runtime.workers = 2;
  tide::Streaming reference(direct, m, runtime), embedding(encoded, em, runtime);
  tide::Continuation q; q.identity = direct.identity;
  auto a = reference.run(q, {{0, 0, 0, 1, x}, {0, 1, 0, 1, x}, {0, 0, 1, 4, y}, {0, 1, 1, 4, y}}, 5, 5);
  q.identity = encoded.identity;
  auto b = embedding.run(q, {{0, 0, 0, 0, x}, {0, 0, 1, 3, y}}, 5, 5);
  for (const auto* result : {&a, &b}) {
    auto loss = result->outputs[0].value.sum() + result->outputs[1].value.sum();
    if (!at::equal(loss, at::full({}, 1004, options))) throw std::runtime_error("source origin changed Aggregate value");
    auto gradient = torch::autograd::grad({loss}, {x, y, gain});
    if (!at::equal(gradient[0], at::full_like(x, 372)) || !at::equal(gradient[1], at::full_like(y, 30))
        || !at::equal(gradient[2], at::full_like(gain, 862))) throw std::runtime_error("source origin changed Aggregate VJP");
  }
}
}
int main(int argc, char** argv) {
  try {
    auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    auto device = portable_torch::resolve_device(args);
    if (args.dtype != at::kFloat && args.dtype != at::kDouble) throw std::invalid_argument("FP32/FP64 required");
    if (!args.output_dir.empty() && std::filesystem::exists(args.output_dir)) throw std::invalid_argument("output exists");
    at::set_num_threads(1);
    auto options = at::TensorOptions().device(device).dtype(args.dtype);
    tide::Graph g; g.nodes = {{0}}; g.nodes[0].aggregation = "clock-fiber-v1";
    g.edges = {{0, 0, 1}}; g.regions = {{1}}; g.inputs = {0, 0}; g.outputs = {0};
    g.layout = tide::PortLayout{{0}, {2}, {1, 0}, {1}}; g.compile();
    tide::Model m;
    m.nodes = {{at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::ones({2}, options)}};
    auto gain = at::full({}, 2, options).set_requires_grad(true);
    m.nodes[0].extra["gain"] = gain; m.nodes[0].aggregate_kernel = std::make_shared<ClockFiber>();
    m.input_scale = {at::ones({}, options), at::ones({}, options)};
    m.agg_scale = m.edge_scale = m.output_scale = {at::ones({}, options)};
    auto x = at::ones({2}, options).set_requires_grad(true);
    auto y = at::full({2}, 3, options).set_requires_grad(true);
    auto unused = at::ones({2}, options).set_requires_grad(true);
    tide::Continuation q; q.identity = g.identity; q.batch_size = 2;
    tide::Options runtime; runtime.packed = true; runtime.workers = 2;
    tide::Streaming engine(g, m, runtime);
    auto result = engine.run(q, {{0, 0, 0, 0, x}, {0, 1, 0, 1, y}, {1, 0, 0, 0, unused}, {1, 1, 0, 1, unused}}, 2, 2);
    auto loss = result.outputs[0].value.sum() + result.outputs[2].value.sum();
    if (!at::equal(loss, at::full({}, 136, options)) || result.stats.at("aggregate_scalar_fallback_steps") != 4)
      throw std::runtime_error("custom Aggregate forward/fallback mismatch");
    auto gradients = torch::autograd::grad({loss}, {x, y, unused, gain}, {}, false, false, true);
    if (!at::equal(gradients[0], at::full_like(x, 52)) || !at::equal(gradients[1], at::full_like(y, 4))
        || gradients[2].defined() || !at::equal(gradients[3], at::full_like(gain, 116)))
      throw std::runtime_error("custom Aggregate isolated VJP mismatch");
    const auto& terms = result.trace[2].contributions;
    if (terms.size() != 2 || terms[0].slot != 0 || terms[1].slot != 2)
      throw std::runtime_error("custom Aggregate contribution identity mismatch");
    check_origins(options);
    const std::string report = "custom-aggregate-kernel: passed\n";
    if (!args.output_dir.empty()) {
      if (!std::filesystem::create_directories(args.output_dir)) throw std::runtime_error("failed to create output");
      std::ofstream file(std::filesystem::path(args.output_dir) / "result.txt"); file << report;
      if (!file) throw std::runtime_error("failed to write result");
    }
    std::cout << report; return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
