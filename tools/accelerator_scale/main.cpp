#include "execution.h"
#include "../../cpp/bench/metrics_jsonl_writer.h"
#include <ATen/Parallel.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

namespace {
double rss() { rusage r{}; if (getrusage(RUSAGE_SELF, &r)) throw std::runtime_error("getrusage failed"); return double(r.ru_maxrss)*1024; }
}
int main(int argc, char** argv) {
  using namespace pdg_scale;
  try {
    Index count = 1; bool seen = false, seen_policy = false, seen_transport = false;
    std::string policy = "memory", transport = "resident";
    std::vector<char*> common{argv[0]};
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--transport") {
        if (seen_transport || ++i == argc) throw std::invalid_argument("duplicate/missing --transport");
        seen_transport = true; transport = argv[i];
        if (transport != "resident" && transport != "host") throw std::invalid_argument("--transport requires resident or host");
        continue;
      }
      if (std::string(argv[i]) == "--placement") {
        if (seen_policy || ++i == argc) throw std::invalid_argument("duplicate/missing --placement");
        seen_policy = true; policy = argv[i];
        if (policy != "memory" && policy != "locality") throw std::invalid_argument("--placement requires memory or locality");
        continue;
      }
      if (std::string(argv[i]) != "--devices") { common.push_back(argv[i]); continue; }
      if (seen || ++i == argc) throw std::invalid_argument("duplicate/missing --devices");
      seen = true; std::string value = argv[i];
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("--devices requires a positive integer");
      count = std::stoll(value);
    }
    auto c = parse(common.size(), common.data());
    if (c.runtime.help) {
      portable_torch::print_usage(std::cout, argv[0]);
      std::cout << "Historical PDG scale options plus --devices 1..16 --placement memory|locality. Explicit CPU coordination/FP64 Read,\n"
                   "--transport resident|host: NPU state/messages or CPU bridge baseline. FP64 Read remains on CPU.\n"
                   "Inference or grad-forward only; no checkpoint import.\n";
      return 0;
    }
    if (c.runtime.dtype != at::kFloat || c.emission != "row" || !c.packed || c.fiber_pooling != "event"
        || c.profile || c.operator_profile || c.work_count || c.head_workers != 1 || count < 1 || count > 16)
      throw std::invalid_argument("scale placement requires FP32, packed row Emit, event pooling, head-workers1 and profiling disabled");
    if (c.runtime.device_spec == "auto") throw std::invalid_argument("benchmark requires an explicit backend");
    // Initialize the vendor queue policy before resolving/initializing NPU.
    // An explicitly supplied value remains available for stack qualification.
    if (c.runtime.device_spec.rfind("npu",0)==0 && !std::getenv("TASK_QUEUE_ENABLE"))
      if (setenv("TASK_QUEUE_ENABLE","0",0)) throw std::runtime_error("cannot select NPU task queue policy");
    const auto device = portable_torch::resolve_device(c.runtime);
    if (!device.is_cpu() && device.type() != c10::DeviceType::PrivateUse1)
      throw std::invalid_argument("this placement client is qualified for CPU/NPU only");
    at::set_num_threads(c.threads); at::set_num_interop_threads(1);
    auto topology = read_topology(c.topology);
    if (!std::filesystem::create_directories(c.runtime.output_dir)) throw std::invalid_argument("new output directory required");
    const bool resident = transport == "resident";
    if (c.check) accelerator_scale::check(c, topology, device, count, policy, resident);
    const auto started = Clock::now();
    portable_experiment::MetricsJsonlWriter writer(std::filesystem::path(c.runtime.output_dir)/"metrics.jsonl", c.run_id);
    portable_torch::seed_runtime(at::Device(at::kCPU), c.runtime.seed);
    at::AutoGradMode grad(c.grad);
    const auto setup_start = Clock::now();
    auto f = fixture(c, topology);
    auto placement = accelerator_scale::place(f, device, count, policy, resident);
    tide::Options options; options.workers = c.workers; options.packed = true; options.trace = false;
    options.full_autograd = c.full_autograd; options.aggregate_autograd = c.aggregate_autograd;
    options.parallel_regions = c.parallel_regions; options.compact_events = c.compact_events;
    options.defer_state_release = c.defer_state_release; options.packed_sources = c.packed_sources; options.batch_next = c.batch_next;
    accelerator_scale::Execution cursor(std::move(f.graph), std::move(f.model), options, c.batch, placement);
    accelerator_scale::synchronize(placement);
    const auto construction = seconds(setup_start);
    std::cout << "MODEL parameters=" << std::fixed << f.inventory.at("parameters")
              << " devices=" << count << " construction_seconds=" << construction << '\n' << std::flush;
    std::ofstream mapping(std::filesystem::path(c.runtime.output_dir)/"placement.json");
    mapping << "{\"node_shards\":[";
    for (size_t i = 0; i < placement.node_device.size(); ++i) mapping << (i ? "," : "") << placement.node_device[i];
    mapping << "],\"device_resident_state\":" << (resident ? "true" : "false") << ",\"cpu_fp64_read\":true,\"policy\":\"" << policy
            << "\",\"physical_edges\":" << placement.edges << ",\"cut_edges\":" << placement.cut_edges
            << ",\"node_load_limit_bytes\":" << placement.node_load_limit << ",\"devices\":" << count
            << ",\"embedding_device_index\":" << f.embedding.device().index()
            << ",\"head_device_index\":" << f.head.device().index() << ",\"parameter_bytes\":[";
    for(size_t i=0;i<placement.parameter_bytes.size();++i) mapping << (i?",":"") << placement.parameter_bytes[i];
    mapping << "]}\n";
    mapping.close(); if (!mapping) throw std::runtime_error("placement publication failed");
    at::Tensor previous_logits;
    for (Index token = 0; token < c.steps; ++token) {
      previous_logits = at::Tensor();
      auto ids = at::remainder(at::arange(c.batch, at::TensorOptions().dtype(at::kLong))*3+token*7, c.vocab);
      accelerator_scale::synchronize(placement);
      accelerator_scale::transfers.reset(); accelerator_scale::reset_memory(placement);
      const auto begin = Clock::now();
      auto embeddings = accelerator_scale::embed(f.embedding, ids, !resident);
      std::vector<tide::External> inputs;
      for (Index b = 0; b < c.batch; ++b) inputs.push_back({b,0,token,token*(topology.layers+1),embeddings[b]});
      auto result = cursor.advance(inputs, (token+1)*(topology.layers+1), (token+1)*(topology.layers+1));
      std::vector<at::Tensor> hidden(c.batch, at::zeros({c.width}, embeddings.options()));
      for (const auto& output : result.outputs) hidden.at(output.batch) = output.value;
      previous_logits = accelerator_scale::project(at::stack(hidden), f.head, !resident);
      accelerator_scale::synchronize(placement);
      const auto elapsed = seconds(begin);
      if (!at::isfinite(previous_logits).all().item<bool>() || previous_logits.requires_grad() != c.grad)
        throw std::runtime_error("nonfinite logits or incorrect grad-forward mode");
      auto metrics = accelerator_scale::transfers.metrics();
      auto mem = accelerator_scale::memory(placement); metrics.insert(mem.begin(),mem.end());
      metrics.insert({{"perf/token_seconds",elapsed}, {"perf/ms_per_sample_token",elapsed*1000/c.batch},
        {"perf/sample_tokens_per_second",c.batch/elapsed}, {"perf/construction_seconds",construction},
        {"memory/process_peak_rss_bytes",rss()}, {"check/logits_sum",previous_logits.detach().to(at::kCPU).to(at::kDouble).sum().item<double>()},
        {"check/logits_requires_grad",c.grad ? 1. : 0.}, {"runtime/devices",double(count)},
        {"runtime/workers",double(c.workers)}, {"runtime/aten_threads",double(at::get_num_threads())},
        {"runtime/effective_device_workers",double(resident?std::min(c.workers,count):c.workers)},
        {"runtime/npu_task_queue",device.is_cpu()?-1.:double(std::atoi(std::getenv("TASK_QUEUE_ENABLE")))},
        {"work/readout_rows",double(result.outputs.size())}});
      for (const auto& [key,value] : f.inventory) metrics["model/"+key] = value;
      for (const auto& [key,value] : result.stats) metrics["work/"+key] = value;
      writer.Write(token,metrics,seconds(started),{{"phase",std::string(token<c.warmup ? "warmup" : "measure")},
        {"memory",c.memory},{"grad",c.grad},{"transport",transport},
        {"partition_policy",policy}});
      std::cout << "STEP " << token << " seconds=" << elapsed << " rows=" << result.outputs.size() << '\n' << std::flush;
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
