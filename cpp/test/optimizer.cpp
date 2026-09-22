#include "portable_torch/runtime.hpp"
#include "tide/optimizer.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tide;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void close(const Tensor& actual, const Tensor& expected, const std::string& message) {
  const auto atol = actual.scalar_type() == at::kDouble ? 1e-12 : 2e-6;
  const auto rtol = actual.scalar_type() == at::kDouble ? 1e-11 : 2e-5;
  require(actual.sizes() == expected.sizes() && at::allclose(actual, expected, rtol, atol), message);
}

Tensor parameter(at::TensorOptions options, std::initializer_list<double> values) {
  return at::tensor(values, options).set_requires_grad(true);
}

void set_gradient(const Tensor& value, const Tensor& gradient) { value.mutable_grad() = gradient; }

void check_registry(at::TensorOptions options) {
  ParameterRegistry registry;
  auto shared = parameter(options, {1., -2.});
  auto other = parameter(options, {.5});
  registry.add("z.shared", shared);
  registry.add("a.shared", shared);
  registry.add("other", other);
  const auto owners = registry.owners();
  require(owners.size() == 2 && owners[0].canonical == "a.shared" && owners[0].aliases.size() == 2,
          "canonical owner/alias partition mismatch");
  require(registry.canonical_name("z.shared") == "a.shared", "alias did not resolve to canonical owner");
  require(registry.alias_partitions()[0][0] == "a.shared" && registry.alias_partitions()[0][1] == "z.shared",
          "alias order mismatch");
  bool duplicate = false;
  try { registry.add("a.shared", other); } catch (const std::invalid_argument&) { duplicate = true; }
  require(duplicate, "duplicate parameter name was accepted");

  OptimizerGroup group;
  group.parameters = {"z.shared", "other"};  // Alias spelling is normalized once.
  group.lr = .1; group.momentum = .8; group.weight_decay = .02;
  SGD optimizer(registry, {group});
  require(optimizer.layout().groups[0][0] == "a.shared", "optimizer lost canonical order");
  set_gradient(shared, at::ones_like(shared));
  set_gradient(other, at::zeros_like(other));  // Connected zero is still an update.
  optimizer.step();
  close(shared, at::tensor({.898, -2.096}, options), "SGD first shared update mismatch");
  close(other, at::tensor({.499}, options), "SGD connected-zero update mismatch");
  require(optimizer.state().size() == 2 && optimizer.state().at("a.shared").momentum_buffer.defined(),
          "SGD momentum ownership mismatch");
  optimizer.zero_grad();
  require(!shared.grad().defined() && !other.grad().defined(), "SGD set-to-none did not preserve None gradients");

  ParameterRegistry adam_registry;
  auto adam_value = parameter(options, {1., -2.});
  auto absent = parameter(options, {.25});
  adam_registry.add("weight", adam_value);
  adam_registry.add("absent", absent);
  OptimizerGroup adam_group;
  adam_group.parameters = {"weight", "absent"};
  adam_group.lr = .01; adam_group.weight_decay = .1; adam_group.eps = 1e-5;
  AdamW adam(adam_registry, {adam_group});
  set_gradient(adam_value, at::tensor({.5, -.25}, options));
  adam.step();
  const auto decayed = at::tensor({.999, -1.998}, options);
  const auto first_direction = at::tensor({.5, -.25}, options)
      / (at::tensor({.5, .25}, options) + 1e-5);
  close(adam_value, decayed - .01 * first_direction, "AdamW first update mismatch");
  require(adam.state().size() == 1 && adam.state().at("weight").step == 1 && !adam.state().count("absent"),
          "AdamW None gradient incorrectly acquired state");
}

void check_model_view(at::TensorOptions options) {
  Model model;
  auto weight = parameter(options, {1., 2.});
  model.nodes.resize(2);
  model.nodes[0].decay = weight;
  model.nodes[0].weight = at::zeros({2, 2}, options);
  model.nodes[0].bias = at::zeros({2}, options);
  model.nodes[0].read = at::zeros({2}, options);
  model.nodes[1] = model.nodes[0];
  auto region_gain = parameter(options, {.3});
  RegionWeights region;
  region.extra["gain"] = region_gain;
  model.regions = {region};
  model.input_scale = {at::ones({}, options).set_requires_grad(true)};
  auto registry = model.parameters();
  require(registry.contains("nodes.0.decay") && registry.contains("nodes.1.decay")
          && registry.contains("regions.0.gain"),
          "Model parameter view omitted trainable aliases");
  require(registry.canonical_name("nodes.1.decay") == "nodes.0.decay",
          "Model parameter view did not preserve cross-node sharing");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kFloat && args.dtype != at::kDouble))
      throw std::invalid_argument("optimizer check requires CPU FP32/FP64");
    if (!args.output_dir.empty() && std::filesystem::exists(args.output_dir))
      throw std::invalid_argument("output exists");
    at::set_num_threads(1);
    const auto options = at::TensorOptions().device(device).dtype(args.dtype);
    check_registry(options);
    check_model_view(options);
    const std::string report = "named-parameter-optimizer: passed; aliases, None/zero gradients, SGD and AdamW\n";
    if (!args.output_dir.empty()) {
      if (!std::filesystem::create_directories(args.output_dir)) throw std::runtime_error("failed to create output");
      std::ofstream file(std::filesystem::path(args.output_dir) / "result.txt");
      file << report;
      if (!file) throw std::runtime_error("failed to write result");
    }
    std::cout << report;
    return 0;
  } catch (const c10::Error& error) {
    std::cerr << error.what_without_backtrace() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
  }
  return 2;
}
