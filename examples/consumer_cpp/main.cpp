#include <tide/stream.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>

int main() {
  try {
    at::set_num_threads(1);
    for (auto dtype : {at::kFloat, at::kDouble}) {
      auto options = at::TensorOptions().dtype(dtype).device(at::kCPU);
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
    std::cout << "{\"state\":\"passed\",\"dtype\":[\"float32\",\"float64\"],\"backend\":\"cpu\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
