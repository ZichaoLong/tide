#include "tide/operator_work.h"
#include "scale.h"
#include "../bench/metrics_jsonl_writer.h"
#include "tide/dense.h"
#include "portable_torch/threads.hpp"
#include <ATen/Parallel.h>
#include <ATen/Version.h>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>

namespace {
double rss() {
  rusage r{}; if (getrusage(RUSAGE_SELF, &r)) throw std::runtime_error("getrusage failed");
  return double(r.ru_maxrss)*1024;
}
}
int main(int argc, char** argv) {
  using namespace pdg_scale;
  try {
    auto c = parse(argc, argv);
    if (c.runtime.help) {
      portable_torch::print_usage(std::cout, argv[0]);
      std::cout << "PDG scale: --topology FILE --run-id ID --width N --batch N --steps N --warmup N\n"
        "  --workers N --head-workers N --threads N --packed 0|1 --grad 0|1 --emission row|slot --vocab N --check 0|1 --profile 0|1\n";
      std::cout << "  --parallel-regions 0|1 --compact-events 0|1\n";
      return 0;
    }
    auto device = portable_torch::resolve_device(c.runtime);
    if (!device.is_cpu() || (c.runtime.dtype != at::kFloat && c.runtime.dtype != at::kDouble))
      throw std::invalid_argument("PDG scale requires CPU FP32/FP64");
    at::set_num_threads(c.threads); at::set_num_interop_threads(1);
    auto topology = read_topology(c.topology);
    if (!std::filesystem::create_directory(c.runtime.output_dir)) throw std::invalid_argument("new output directory required");
    const auto started = Clock::now();
    portable_experiment::MetricsJsonlWriter writer(std::filesystem::path(c.runtime.output_dir)/"metrics.jsonl", c.run_id);
    if (c.check) { check(c, topology); std::cout << "CHECK passed: scalar slot / scalar row / packed row / parallel packed row\n" << std::flush; }
    portable_torch::seed_runtime(device, c.runtime.seed);
    at::AutoGradMode grad(c.grad);
    const auto construction_start = Clock::now();
    auto f = fixture(c, topology);
    tide::Options options; options.workers = c.workers; options.packed = c.packed; options.trace = false;
    options.profile = c.profile;
    options.parallel_regions = c.parallel_regions; options.compact_events = c.compact_events;
    tide::Streaming engine(std::move(f.graph), std::move(f.model), options);
    tide::DenseLinear head(c.head_workers);
    const auto construction = seconds(construction_start);
    const auto runtime_threads = portable_torch::thread_metrics();
    std::cout << "MODEL parameters=" << std::fixed << f.inventory.at("parameters") << " construction_seconds=" << construction
              << " grad=" << c.grad << " workers=" << c.workers << " threads=" << at::get_num_threads() << '\n'
              << at::get_parallel_info() << portable_torch::blas_description() << '\n' << std::flush;
    tide::Continuation q; q.identity = engine.graph().identity; q.batch_size = c.batch;
    tide::StreamingCursor cursor(engine, std::move(q));
    const auto period = topology.layers+1;
    // Fixed external token IDs keep input preparation identical across schedules.
    // Persistent cache/history spans all steps, including the warmup prefix.
    at::Tensor previous_logits;
    for (Index token = 0; token < c.steps; ++token) {
      tide::work::reset(c.work_count);
      const auto begin = Clock::now();
      previous_logits = at::Tensor();
      auto ids = at::remainder(at::arange(c.batch, at::TensorOptions().dtype(at::kLong))*3+token*7, c.vocab);
      auto embeddings = f.embedding.index_select(0, ids);
      std::vector<tide::External> inputs;
      for (Index b = 0; b < c.batch; ++b) inputs.push_back({b, 0, token, token*period, embeddings[b]});
      const auto advance_begin = Clock::now();
      auto result = cursor.advance(inputs, (token+1)*period, (token+1)*period);
      const auto advance_end = Clock::now();
      const auto advance = std::chrono::duration<double>(advance_end-advance_begin).count();
      // Missing readout is an explicit zero row, as in sparse token readout.
      std::vector<at::Tensor> hidden(c.batch, at::zeros({c.width}, embeddings.options()));
      for (const auto& output : result.outputs) hidden.at(output.batch) = output.value;
      tide::work::linear(tide::work::HeadCalls, c.batch, c.width, c.vocab);
      previous_logits = head.run(at::stack(hidden), f.head);
      const auto end = Clock::now();
      const auto elapsed = std::chrono::duration<double>(end-begin).count();
      if (!at::isfinite(previous_logits).all().item<bool>() || previous_logits.requires_grad() != c.grad)
        throw std::runtime_error("nonfinite logits or incorrect grad mode");
      std::map<std::string, double> metrics{{"perf/token_seconds", elapsed}, {"perf/advance_seconds", advance},
        {"perf/ms_per_sample_token", elapsed*1000/c.batch}, {"perf/sample_tokens_per_second", c.batch/elapsed},
        {"perf/construction_seconds", construction}, {"memory/process_peak_rss_bytes", rss()},
        {"check/logits_sum", previous_logits.detach().to(at::kDouble).sum().item<double>()},
        {"check/logits_requires_grad", previous_logits.requires_grad() ? 1. : 0.},
        {"work/readout_rows", double(result.outputs.size())}};
      if (c.work_count) {
        tide::work::add(tide::work::BodyCandidates, result.stats["candidate_events"]-result.outputs.size());
        tide::work::add(tide::work::BodySelected, result.stats["selected_events"]-result.outputs.size());
        auto work = tide::work::metrics(); metrics.insert(work.begin(), work.end());
      }
      if (c.profile) {
        metrics["profile/input_seconds"] = std::chrono::duration<double>(advance_begin-begin).count();
        metrics["profile/head_seconds"] = std::chrono::duration<double>(end-advance_end).count();
      }
      for (const auto& [key, value] : f.inventory) metrics["model/"+key] = value;
      metrics.insert(runtime_threads.begin(), runtime_threads.end());
      metrics["runtime/head_workers"] = c.head_workers;
      for (const auto& [key, value] : result.stats) {
        if (key.rfind("profile_", 0) == 0)
          metrics["profile/"+key.substr(8, key.size()-11)+"_seconds"] = value*1e-9;
        else metrics["work/"+key] = value;
      }
      if (result.stats["update_calls"]) metrics["work/mean_rows_per_update_call"] = double(result.stats["candidate_events"])/result.stats["update_calls"];
      writer.Write(token, metrics, seconds(started), {{"phase", std::string(token < c.warmup ? "warmup" : "measure")},
        {"emission", c.emission}, {"packed", c.packed}, {"grad", c.grad}, {"profile", c.profile}});
      std::cout << "STEP " << token << " ms/sample-token=" << elapsed*1000/c.batch
                << " candidates=" << result.stats["candidate_events"] << " edges=" << result.stats["visited_edges"] << '\n' << std::flush;
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
