#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (args.dtype != at::kFloat && args.dtype != at::kDouble) throw std::invalid_argument("graph smoke supports float32/float64 only");
    const std::filesystem::path output(args.output_dir);
    if (!args.output_dir.empty() && std::filesystem::exists(output)) throw std::invalid_argument("output path already exists");
    portable_torch::seed_runtime(device, args.seed);
    at::set_num_threads(1);
    auto options = at::TensorOptions().device(device).dtype(args.dtype);
    tide::Graph graph;
    graph.nodes = {{0, false, false}}; graph.edges = {{0, 0, 2}};
    graph.regions = {{1, true, true}}; graph.inputs = {0}; graph.outputs = {0}; graph.compile();
    tide::Model model;
    model.nodes = {{at::zeros({2}, options), at::eye(2, options) * 0.2,
                    at::zeros({2}, options), at::ones({2}, options)}};
    model.input_scale = model.agg_scale = model.edge_scale = model.output_scale = {at::ones({}, options)};
    tide::Continuation q; q.identity = graph.identity;
    auto x = at::ones({2}, options); x.set_requires_grad(true);
    tide::Streaming engine(graph, model, {});
    auto result = engine.run(q, {{0, 0, 0, 0, x}}, 5, 5);
    auto loss = result.continuation.states.at({0, 0}).value.sum() + result.continuation.pending.at(0).value.sum();
    for (const auto& o : result.outputs) loss = loss + o.value.sum();
    auto gradient = torch::autograd::grad({loss}, {x}).at(0);
    if (result.outputs.size() != 3 || !at::isfinite(gradient).all().item<bool>()) throw std::runtime_error("invalid graph/gradient smoke result");
    std::ostringstream report;
    report.precision(17);
    report << "{\"device\":\"cpu\",\"resolution_reason\":\"" << portable_torch::resolution_reason(args, device)
           << "\",\"dtype\":\"" << portable_torch::dtype_name(args.dtype) << "\",\"outputs\":" << result.outputs.size()
           << ",\"loss\":" << loss.item<double>() << ",\"gradient_sum\":" << gradient.sum().item<double>() << "}\n";
    if (!args.output_dir.empty()) {
      if (!std::filesystem::create_directories(output)) throw std::runtime_error("failed to create output directory");
      std::ofstream file(output / "result.json"); file << report.str();
      if (!file) throw std::runtime_error("failed to write result");
    }
    std::cout << report.str();
    return 0;
  } catch (const c10::Error& e) {
    std::cerr << e.what_without_backtrace() << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
  return 2;
}
