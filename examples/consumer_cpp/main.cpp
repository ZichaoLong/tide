#include <tide/stream.h>
#include <portable_torch/runtime.hpp>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv, true);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (args.dtype != at::kFloat && args.dtype != at::kDouble) throw std::invalid_argument("FP32/FP64 required");
    at::set_num_threads(1);
    portable_torch::seed_runtime(device, args.seed);
    {
      auto options = at::TensorOptions().dtype(args.dtype).device(device);
      tide::Graph graph;
      graph.nodes = {{0}}; graph.edges = {{0, 0, 2}};
      graph.regions = {{1}}; graph.inputs = graph.outputs = {0}; graph.compile();
      tide::Model model;
      model.nodes = {{at::zeros({2}, options), at::eye(2, options) * .2,
                      at::zeros({2}, options), at::ones({2}, options)}};
      model.input_scale = model.agg_scale = model.edge_scale = model.output_scale = {at::ones({}, options)};
      tide::Continuation state; state.identity = graph.identity;
      auto input = at::ones({2}, options); input.set_requires_grad(true);
      tide::Streaming executor(graph, model, {});
      auto first = executor.run(state, {{0,0,0,0,input}}, 2, 2);
      auto second = executor.run(first.continuation, {}, 5, 5);
      auto whole = executor.run(state, {{0,0,0,0,input}}, 5, 5);
      if (whole.outputs.size() != 3 || second.outputs.size() != 2 ||
          !at::allclose(whole.outputs.back().value, second.outputs.back().value))
        throw std::runtime_error("C++ consumer chunk mismatch");
      auto loss = whole.outputs.back().value.square().sum();
      auto grad = torch::autograd::grad({loss}, {input}).at(0);
      if (!at::isfinite(grad).all().item<bool>() || grad.abs().sum().item<double>() == 0)
        throw std::runtime_error("C++ consumer gradient mismatch");
    }
    portable_torch::synchronize(device);
    std::cout << "{\"state\":\"passed\",\"dtype\":\"" << portable_torch::dtype_name(args.dtype)
              << "\",\"device\":\"" << device.str() << "\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
