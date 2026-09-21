#include "streaming.h"
#include "metrics_jsonl_writer.h"
#include <ATen/Parallel.h>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>

namespace {
double peak_rss_bytes() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss);
#else
  return static_cast<double>(usage.ru_maxrss)*1024;
#endif
}
tide::Options options(const tide_bench::Config& c, bool trace) {
  tide::Options result; result.workers = c.workers; result.packed = c.packed; result.trace = trace;
  return result;
}
}
int main(int argc, char** argv) {
  using namespace tide_bench;
  try {
    auto c = parse(argc, argv);
    if (c.runtime.help) {
      portable_torch::print_usage(std::cout, argv[0]);
      std::cout << "Benchmark: --run-id ID --nodes N --active-rings N --batch N --width N --ticks N\n"
                   "  --api cursor|functional --workers N --packed 0|1 --warmup N --repetitions N\n";
      return 0;
    }
    const auto device = portable_torch::resolve_device(c.runtime);
    if (!device.is_cpu() || (c.runtime.dtype != at::kFloat && c.runtime.dtype != at::kDouble))
      throw std::invalid_argument("streaming benchmark requires CPU FP64/FP32");
    at::set_num_threads(1); at::set_num_interop_threads(1); at::NoGradGuard guard;
    portable_torch::seed_runtime(device, c.runtime.seed);
    if (!std::filesystem::create_directory(c.runtime.output_dir))
      throw std::invalid_argument("benchmark output directory must be new");
    const auto started = Clock::now();
    portable_experiment::MetricsJsonlWriter writer(std::filesystem::path(c.runtime.output_dir)/"metrics.jsonl", c.run_id);
    const auto setup_start = Clock::now();
    auto f = fixture(c);
    tide::Streaming engine(std::move(f.graph), std::move(f.model), options(c, false));
    const auto setup = seconds(setup_start);

    const auto validation_start = Clock::now();
    Config anchor = c; anchor.nodes = 8*c.rings; anchor.api = "functional";
    anchor.packed = false; anchor.workers = 1;
    tide::Result expected;
    {
      auto base = fixture(anchor);
      tide::Streaming scalar(std::move(base.graph), std::move(base.model), options(anchor, true));
      auto target = fixture(c);
      tide::Streaming traced(std::move(target.graph), std::move(target.model), options(c, true));
      auto reference = execute(scalar, anchor, base.inputs);
      auto actual = execute(traced, c, target.inputs);
      check_work(reference, anchor); check_work(actual, c); compare(actual.result, reference.result, true);
      expected = std::move(reference.result);
    }
    expected.trace.clear(); expected.messages.clear();
    const auto validation = seconds(validation_start);
    for (Index repeat = -c.warmup; repeat < c.repetitions; ++repeat) {
      auto sample = execute(engine, c, f.inputs);
      if (sample.result.continuation.identity != engine.graph().identity) throw std::runtime_error("benchmark identity changed");
      check_work(sample, c); compare(sample.result, expected, false);
      if (repeat < 0) continue;
      std::map<std::string, double> metrics{
        {"perf/construction_seconds", setup}, {"perf/validation_seconds", validation}, {"perf/reset_seconds", sample.reset},
        {"perf/advance_seconds", sample.advance}, {"perf/snapshot_seconds", sample.snapshot},
        {"perf/candidate_events_per_second", c.ticks*c.rings*c.batch/sample.advance},
        {"memory/process_peak_rss_bytes", peak_rss_bytes()},
        {"memory/graph_identity_bytes", static_cast<double>(engine.graph().identity.size())},
        {"work/cached_states", static_cast<double>(sample.result.continuation.states.size())},
        {"work/pending_messages", static_cast<double>(sample.result.continuation.pending.size())},
        {"work/output_rows", static_cast<double>(sample.result.outputs.size())}};
      double checksum = 0;
      for (const auto& output : sample.result.outputs) checksum += output.value.to(at::kDouble).sum().item<double>();
      metrics["check/output_checksum"] = checksum;
      double states = 0, pending = 0;
      for (const auto& [owner, state] : sample.result.continuation.states) states += state.value.to(at::kDouble).sum().item<double>();
      for (const auto& atom : sample.result.continuation.pending) pending += atom.value.to(at::kDouble).sum().item<double>();
      metrics["check/state_checksum"] = states; metrics["check/pending_checksum"] = pending;
      for (const auto& [name, value] : sample.result.stats) metrics["work/"+name] = value;
      writer.Write(repeat, metrics, seconds(started), {{"api", c.api}, {"nodes", c.nodes}, {"packed", c.packed}, {"workers", c.workers}});
    }
    std::cout << "streaming benchmark: parity passed; " << c.repetitions << " recorded repetitions\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
