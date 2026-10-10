#include "device_backend.h"
#include "device_program.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tuple>

namespace {
using tide::device_online::DeviceProgram;
using at::Tensor;
void require(bool condition, const char* reason) {
  if (!condition) throw std::runtime_error(reason);
}
template<class F> void rejects(F fn) {
  bool rejected = false;
  try { fn(); } catch (const std::logic_error&) { rejected = true; }
  require(rejected, "control lifecycle/shape guard accepted invalid call");
}
int64_t check(at::Device device) {
  // All storage is fixed. These are control values, not floating payloads.
  const auto longs = at::TensorOptions().dtype(at::kLong).device(device);
  const auto bools = longs.dtype(at::kBool), ints = longs.dtype(at::kInt);
  auto count=at::zeros({1},longs), stop=at::zeros({1},longs), step=at::ones({1},longs);
  auto iterations=at::zeros({1},longs), capacity=at::ones({1},longs), one=at::ones({1},longs);
  auto unfinished=at::zeros({1},bools), room=at::zeros({1},bools), ready=at::zeros({1},bools);
  auto index=at::zeros({1},ints), again=at::zeros({1},ints), exhausted=at::zeros({1},ints);
  portable_torch::synchronize(device);
  DeviceProgram program(device);
  rejects([&] { program.run(); });
  const auto head=program.label(), body=program.label(), end=program.label();
  program.mark(head);
  program.less(count, stop, unfinished);
  program.less(iterations, capacity, room);
  program.logical_and(unfinished, room, ready);
  program.cast_index(ready, index);
  program.branch(index, {end, body});
  program.mark(body);
  program.add(count, step);
  program.add(iterations, one);
  program.branch(again, {head});
  program.mark(end);
  program.less(count, stop, unfinished);
  program.cast_index(unfinished, exhausted);
  program.finish();
  rejects([&] { program.finish(); });
  rejects([&] { program.mark(head); });
  rejects([&] { program.run(0); });
  int64_t cases = 0;
  auto run = [&](int64_t target, int64_t increment, int64_t budget, int64_t expected,
                 int64_t expected_steps, int64_t expected_exhausted) {
    stop.fill_(target); step.fill_(increment); capacity.fill_(budget); iterations.zero_();
    portable_torch::synchronize(device); program.run();
    // Export only at the completed model boundary. Never feeds a device branch.
    require(count.to(at::kCPU).item<int64_t>() == expected, "control count/int64 mismatch");
    require(iterations.to(at::kCPU).item<int64_t>() == expected_steps, "device chose wrong loop count");
    require(exhausted.to(at::kCPU).item<int32_t>() == expected_exhausted, "capacity status mismatch");
    ++cases;
  };
  run(5, 1, 16, 5, 5, 0);
  run(5, 1, 16, 5, 0, 0);       // Empty continuation leaves the state unchanged.
  run(11, 2, 2, 9, 2, 1);        // Explicit capacity exhaustion; state is resumable.
  run(11, 2, 2, 11, 1, 0);
  run(20, 3, 0, 11, 0, 1);       // No work fits; no hidden extra iteration.
  run(20, 3, 8, 20, 3, 0);
  const int64_t large=(int64_t(1)<<55)+17;
  count.fill_(large);
  run(large+10, 2, 3, large+6, 3, 1);
  run(large+10, 2, 3, large+10, 2, 0);
  const auto bytes = program.workspace_bytes();
  require(bytes > 0, "control workspace owners were not retained");
  program.close(); program.close();
  rejects([&] { program.run(); });
  std::cout << "device-control: passed cases=" << cases << " exact_int64=true continuation=true"
            << " zero_iterations=true capacity_status=true host_decisions_per_iteration=0"
            << " retained_workspace_bytes=" << bytes << '\n';
  return cases;
}
}  // namespace
int main(int argc, char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    const auto args = portable_torch::parse_cli(argc, argv, true);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    if (args.device_spec == "auto" || args.dtype != at::kFloat)
      throw std::invalid_argument("control check requires explicit NPU and float32 runtime (control buffers are int64/bool/int32)");
    if (!args.output_dir.empty() && std::filesystem::exists(args.output_dir))
      throw std::invalid_argument("output directory already exists");
    const auto device = portable_torch::resolve_device(args);
    if (device.type() != tide::device_online::resident_device_type) throw std::invalid_argument("control check requires NPU");
    at::set_num_threads(1); at::set_num_interop_threads(1);
    const auto cases = check(device);
    runtime.close();
    if (!args.output_dir.empty()) {
      if (!std::filesystem::create_directories(args.output_dir)) throw std::runtime_error("cannot create output");
      std::ofstream file(std::filesystem::path(args.output_dir)/"result.json");
      file << "{\"schema\":\"tide-device-control-v1\",\"state\":\"passed\",\"cases\":" << cases
           << ",\"scope\":\"single-device control only; no graph/queue/training/performance claim\"}\n";
      if (!file) throw std::runtime_error("cannot write result");
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
