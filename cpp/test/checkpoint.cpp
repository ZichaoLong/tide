#include "portable_torch/runtime.hpp"
#include "tide/checkpoint.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using namespace tide;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void same_tensor(const Tensor& actual, const Tensor& expected, const std::string& message) {
  require(actual.defined() == expected.defined(), message + " definedness");
  if (actual.defined()) require(at::equal(actual, expected), message);
}

void same_state(const std::map<std::string, OptimizerState>& actual,
                const std::map<std::string, OptimizerState>& expected) {
  require(actual.size() == expected.size(), "optimizer state owner count mismatch");
  for (const auto& [name, state] : expected) {
    const auto it = actual.find(name);
    require(it != actual.end(), "optimizer state owner missing: " + name);
    require(it->second.step == state.step, "optimizer step mismatch: " + name);
    same_tensor(it->second.momentum_buffer, state.momentum_buffer, "momentum mismatch: " + name);
    same_tensor(it->second.exp_avg, state.exp_avg, "exp_avg mismatch: " + name);
    same_tensor(it->second.exp_avg_sq, state.exp_avg_sq, "exp_avg_sq mismatch: " + name);
    same_tensor(it->second.max_exp_avg_sq, state.max_exp_avg_sq, "max_exp_avg_sq mismatch: " + name);
  }
}

struct Fixture {
  ParameterRegistry registry;
  Tensor shared;
  Tensor other;
};

Fixture make_fixture(at::ScalarType dtype, double offset = 0.0) {
  const auto options = at::TensorOptions().dtype(dtype).device(at::kCPU);
  Fixture result;
  result.shared = at::tensor({1.0 + offset, -2.0 + offset}, options).set_requires_grad(true);
  result.other = at::tensor({0.5 + offset}, options).set_requires_grad(true);
  result.registry.add("graph.z.shared", result.shared);
  result.registry.add("graph.a.shared", result.shared);
  result.registry.add("graph.other", result.other);
  return result;
}

OptimizerGroup group(const std::vector<std::string>& names, double lr, double momentum,
                     double weight_decay, double eps = 1e-8, bool amsgrad = false) {
  OptimizerGroup result;
  result.parameters = names;
  result.lr = lr;
  result.momentum = momentum;
  result.weight_decay = weight_decay;
  result.eps = eps;
  result.amsgrad = amsgrad;
  return result;
}

template <typename Optimizer>
void seed_state(Fixture& fixture, Optimizer& optimizer, bool adam) {
  fixture.shared.mutable_grad() = at::tensor({1.0, -0.25}, fixture.shared.options());
  fixture.other.mutable_grad().reset();  // Structural absence: no state is created.
  optimizer.step();
  optimizer.zero_grad();
  fixture.shared.mutable_grad() = at::zeros_like(fixture.shared);
  fixture.other.mutable_grad() = at::zeros_like(fixture.other);  // Connected zero: state is created.
  optimizer.step();
  optimizer.zero_grad();
  require(optimizer.state().count("graph.a.shared") == 1, "shared owner state was not created");
  require(optimizer.state().count("graph.other") == 1, "connected-zero state was not created");
  if (adam) {
    require(optimizer.state().at("graph.other").exp_avg.defined(), "AdamW zero state is absent");
  } else {
    require(optimizer.state().at("graph.other").momentum_buffer.defined(), "SGD zero state is absent");
  }
}

template <typename Optimizer>
void same_groups(const Optimizer& actual, const Optimizer& expected) {
  require(actual.layout().class_name == expected.layout().class_name, "optimizer class mismatch");
  require(actual.layout().groups == expected.layout().groups, "optimizer group order mismatch");
  require(actual.groups().size() == expected.groups().size(), "optimizer group count mismatch");
  for (size_t i = 0; i < expected.groups().size(); ++i) {
    const auto& a = actual.groups()[i];
    const auto& e = expected.groups()[i];
    require(a.lr == e.lr && a.weight_decay == e.weight_decay && a.momentum == e.momentum
                && a.dampening == e.dampening && a.beta1 == e.beta1 && a.beta2 == e.beta2
                && a.eps == e.eps && a.nesterov == e.nesterov && a.amsgrad == e.amsgrad
                && a.maximize == e.maximize,
            "optimizer group options were not restored");
  }
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(output), "failed to write test checkpoint");
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "failed to read test checkpoint");
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

template <typename Optimizer>
void check_round_trip(at::ScalarType dtype, const std::filesystem::path& root, bool adam) {
  auto source = make_fixture(dtype);
  const auto groups = std::vector<OptimizerGroup>{
      group({"graph.z.shared"}, .1, .8, .02, adam ? 1e-5 : 1e-8, adam),
      group({"graph.other"}, .03, .6, .0, adam ? 1e-5 : 1e-8, false)};
  Optimizer optimizer(source.registry, groups);
  seed_state(source, optimizer, adam);
  const auto expected_shared = source.shared.detach().clone();
  const auto expected_other = source.other.detach().clone();
  const auto expected_groups = optimizer;
  const auto expected_state = optimizer.state();
  const auto path = root / (adam ? "adamw.ckpt" : "sgd.ckpt");
  Checkpoint::save(path, source.registry, &optimizer, "graph-v1");

  auto restored = make_fixture(dtype, 10.0);
  auto different_groups = groups;
  different_groups[0].lr = .777;
  different_groups[1].weight_decay = .44;
  Optimizer restored_optimizer(restored.registry, different_groups);
  Checkpoint::load(path, restored.registry, &restored_optimizer, "graph-v1");
  same_tensor(restored.shared, expected_shared, "shared value did not round-trip");
  same_tensor(restored.other, expected_other, "other value did not round-trip");
  same_groups(restored_optimizer, expected_groups);
  same_state(restored_optimizer.state(), expected_state);
  require(restored.registry.alias_partitions() == source.registry.alias_partitions(),
          "alias topology changed during round-trip");

  // The next update must use the restored options and moments, not the
  // intentionally different options used to construct the destination.
  source.shared.mutable_grad() = at::tensor({.25, -.75}, source.shared.options());
  source.other.mutable_grad() = at::tensor({.5}, source.other.options());
  restored.shared.mutable_grad() = source.shared.grad().clone();
  restored.other.mutable_grad() = source.other.grad().clone();
  optimizer.step();
  restored_optimizer.step();
  same_tensor(restored.shared, source.shared, "shared next update mismatch");
  same_tensor(restored.other, source.other, "other next update mismatch");
  same_state(restored_optimizer.state(), optimizer.state());

  // A graph identity mismatch and an alias-topology mismatch both fail before
  // touching either the values or the optimizer continuation.
  auto identity_target = make_fixture(dtype, 3.0);
  Optimizer identity_optimizer(identity_target.registry, groups);
  const auto identity_before = identity_target.shared.clone();
  const auto identity_state_before = identity_optimizer.state();
  bool identity_failed = false;
  try { Checkpoint::load(path, identity_target.registry, &identity_optimizer, "different-graph"); }
  catch (const std::exception&) { identity_failed = true; }
  require(identity_failed, "identity mismatch was accepted");
  same_tensor(identity_target.shared, identity_before, "identity failure mutated values");
  same_state(identity_optimizer.state(), identity_state_before);

  ParameterRegistry wrong_registry;
  auto wrong_shared = at::zeros({2}, at::TensorOptions().dtype(dtype).device(at::kCPU)).set_requires_grad(true);
  auto wrong_other = at::zeros({1}, at::TensorOptions().dtype(dtype).device(at::kCPU)).set_requires_grad(true);
  wrong_registry.add("graph.a.shared", wrong_shared);
  wrong_registry.add("graph.other", wrong_other);
  Optimizer wrong_optimizer(wrong_registry, {group({"graph.a.shared"}, .1, .8, .02, adam ? 1e-5 : 1e-8, adam),
                                             group({"graph.other"}, .03, .6, 0.0, adam ? 1e-5 : 1e-8, false)});
  bool alias_failed = false;
  try { Checkpoint::load(path, wrong_registry, &wrong_optimizer, "graph-v1"); }
  catch (const std::exception&) { alias_failed = true; }
  require(alias_failed, "alias topology mismatch was accepted");

  // Corruption and truncation are rejected by checksum/preflight and leave a
  // live destination unchanged.
  const auto good = read_bytes(path);
  for (int kind = 0; kind < 2; ++kind) {
    auto damaged = good;
    if (kind == 0) damaged.at(12) ^= 0x80;
    else damaged.resize(damaged.size() - 1);
    const auto damaged_path = root / (adam ? (kind == 0 ? "adamw-corrupt.ckpt" : "adamw-short.ckpt")
                                           : (kind == 0 ? "sgd-corrupt.ckpt" : "sgd-short.ckpt"));
    write_bytes(damaged_path, damaged);
    auto damaged_target = make_fixture(dtype, 4.0);
    Optimizer damaged_optimizer(damaged_target.registry, groups);
    const auto before_value = damaged_target.shared.clone();
    const auto before_state = damaged_optimizer.state();
    bool failed = false;
    try { Checkpoint::load(damaged_path, damaged_target.registry, &damaged_optimizer, "graph-v1"); }
    catch (const std::exception&) { failed = true; }
    require(failed, "damaged checkpoint was accepted");
    same_tensor(damaged_target.shared, before_value, "damaged load mutated values");
    same_state(damaged_optimizer.state(), before_state);
  }

  bool overwrite_failed = false;
  try { Checkpoint::save(path, source.registry, &optimizer, "graph-v1"); }
  catch (const std::exception&) { overwrite_failed = true; }
  require(overwrite_failed, "checkpoint save overwrote an existing target");
  require(read_bytes(path) == good, "failed overwrite changed the existing checkpoint");
}

void check_weights_only(at::ScalarType dtype, const std::filesystem::path& root) {
  auto source = make_fixture(dtype);
  const auto path = root / "weights-only.ckpt";
  Checkpoint::save(path, source.registry, nullptr, "weights-v1");
  auto restored = make_fixture(dtype, 6.0);
  Checkpoint::load(path, restored.registry, nullptr, "weights-v1");
  same_tensor(restored.shared, source.shared, "weights-only shared mismatch");
  same_tensor(restored.other, source.other, "weights-only other mismatch");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kFloat && args.dtype != at::kDouble))
      throw std::invalid_argument("checkpoint check requires CPU FP32/FP64");
    if (!args.output_dir.empty() && std::filesystem::exists(args.output_dir))
      throw std::invalid_argument("output exists");
    at::set_num_threads(1);
    const auto root = args.output_dir.empty()
        ? std::filesystem::temp_directory_path() / ("tidegraph-checkpoint-" + std::to_string(::getpid()))
        : std::filesystem::path(args.output_dir);
    std::filesystem::create_directories(root);
    check_round_trip<SGD>(args.dtype, root, false);
    check_round_trip<AdamW>(args.dtype, root, true);
    check_weights_only(args.dtype, root);
    if (args.output_dir.empty()) std::filesystem::remove_all(root);
    const std::string report = "native-checkpoint: passed; schema, aliases, SGD/AdamW state, corruption and exclusive publication\n";
    if (!args.output_dir.empty()) {
      std::ofstream file(root / "result.txt");
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
