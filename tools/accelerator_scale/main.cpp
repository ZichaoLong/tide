#include "execution.h"
#include "training.h"
#include "../../cpp/bench/metrics_jsonl_writer.h"
#include <ATen/Parallel.h>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/resource.h>

namespace {
bool used_npu = false;
double rss() { rusage r{}; if (getrusage(RUSAGE_SELF, &r)) throw std::runtime_error("getrusage failed"); return double(r.ru_maxrss)*1024; }
}
int run(int argc, char** argv) {
  using namespace pdg_scale;
  try {
    Index count = 1; bool seen = false, seen_policy = false, seen_transport = false, seen_vjp = false;
    std::string policy = "memory", transport = "resident", vjp_policy = "strict";
    accelerator_scale::Scoring scoring;
    accelerator_scale::TrainingConfig training;
    accelerator_scale::ProfileConfig profile;
    std::string ranking_device="cpu", event_device="cpu";
    bool reference_fp64 = false;
    double check_atol=-1., check_rtol=-1.;
    bool reference_float32=false;
    std::set<std::string> scoring_flags;
    std::vector<char*> common{argv[0]};
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if(flag=="--backward-threads" || flag=="--optimizer-threads") {
        if(!scoring_flags.insert(flag).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+flag);
        const std::string value=argv[i];
        if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
          throw std::invalid_argument("positive phase thread integer required");
        const auto n=std::stoll(value);if(n<1 || n>160)throw std::invalid_argument("phase threads require1..160");
        if(flag=="--backward-threads")training.backward_threads=n;else training.optimizer_threads=n;
        continue;
      }
      if(flag=="--profile-step" || flag=="--profile-phase" || flag=="--profile-output") {
        if(!scoring_flags.insert(flag).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+flag);
        const std::string value=argv[i];
        if(flag=="--profile-step") {
          if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("profile step requires a nonnegative integer");
          profile.step=std::stoll(value);
        } else if(flag=="--profile-phase")profile.phase=value;
        else profile.output=value;
        continue;
      }
      if(flag=="--reference-payload-dtype") {
        if(!scoring_flags.insert(flag).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+flag);
        const std::string value=argv[i];
        if(value!="matched" && value!="float32")throw std::invalid_argument("reference payload requires matched or float32");
        reference_float32=value=="float32";continue;
      }
      if(flag=="--check-atol" || flag=="--check-rtol" || flag=="--loss-scale") {
        if(!scoring_flags.insert(flag).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+flag);
        size_t used=0;const std::string value=argv[i];const auto number=std::stod(value,&used);
        if(used!=value.size() || !std::isfinite(number) || number<=0)throw std::invalid_argument("positive finite "+flag+" required");
        if(flag=="--check-atol")check_atol=number;
        else if(flag=="--check-rtol")check_rtol=number;
        else training.loss_scale=number;
        continue;
      }
      if(flag=="--training-steps" || flag=="--training-warmup" || flag=="--optimizer" || flag=="--learning-rate") {
        if(!scoring_flags.insert(flag).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+flag);
        const std::string value=argv[i];
        if(flag=="--optimizer")training.optimizer=value;
        else if(flag=="--learning-rate") {
          size_t used=0;training.learning_rate=std::stod(value,&used);
          if(used!=value.size())throw std::invalid_argument("invalid learning rate");
        }else {
          if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("nonnegative training integer required");
          if(flag=="--training-steps")training.steps=std::stoll(value);else training.warmup=std::stoll(value);
        }
        continue;
      }
      if (flag == "--read-device" || flag == "--control-device" || flag == "--read-dtype" || flag == "--reference-read-dtype" || flag == "--ranking-device" || flag == "--event-device") {
        if (!scoring_flags.insert(flag).second || ++i == argc) throw std::invalid_argument("duplicate/missing "+flag);
        const std::string value = argv[i];
        if (flag == "--read-device") scoring.read_device = value;
        else if (flag == "--control-device") scoring.control_device = value;
        else if (flag == "--ranking-device") ranking_device = value;
        else if (flag == "--event-device") event_device = value;
        else if (flag == "--read-dtype") {
          if (value != "float32" && value != "float64") throw std::invalid_argument("--read-dtype requires float32 or float64");
          scoring.dtype = value == "float32" ? at::kFloat : at::kDouble;
        } else {
          if (value != "matched" && value != "float64") throw std::invalid_argument("--reference-read-dtype requires matched or float64");
          reference_fp64 = value == "float64";
        }
        continue;
      }
      if (std::string(argv[i]) == "--vjp-policy") {
        if(seen_vjp || ++i==argc)throw std::invalid_argument("duplicate/missing --vjp-policy");
        seen_vjp=true;vjp_policy=argv[i];
        if(vjp_policy!="strict" && vjp_policy!="basis-conditioned")
          throw std::invalid_argument("--vjp-policy requires strict or basis-conditioned");
        continue;
      }
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
      std::cout << "Historical PDG scale options plus --devices 1..16 --placement memory|locality. configurable event scheduling/ranking,\n"
                   "--transport resident|host: NPU state/messages or CPU bridge baseline.\n"
                   "--read-device cpu|model --read-dtype float64|float32 --control-device cpu|model (defaults cpu,float64,cpu).\n"
                   "model uses node/region-owner device; NPU requires FP32 Read/controls.\n"
                   "--ranking-device cpu|model --event-device cpu|model (defaults cpu,cpu).\n"
                   "--dtype float32|float16: payload precision; FP16 requires resident transport.\n"
                   "--reference-payload-dtype matched|float32: same-dtype oracle or cross-precision diagnostic.\n"
                   "--check-atol X --check-rtol X: FP16 only; default 1e-3/2e-2, routes stay exact.\n"
                   "--loss-scale X: static training scale (FP32 1, FP16 128); FP32 master/slots/loss.\n"
                   "--training-steps N --training-warmup N --optimizer sgd|adamw --learning-rate X.\n"
                   "--head-workers N: CPU output-column workers (default1); NPU requires1.\n"
                   "--backward-threads N --optimizer-threads N: CPU training phase ATen/BLAS threads (default1).\n"
                   "--profile-step N --profile-phase token|forward|backward|optimizer --profile-output NEW: scoped CANN trace.\n"
                   "Training: --grad 1 --warmup 0; each update uses --steps tokens from empty state.\n"
                   "--check 1 --vjp-policy strict|basis-conditioned --reference-read-dtype matched|float64.\n";
      return 0;
    }
    if ((c.runtime.dtype != at::kFloat && c.runtime.dtype != at::kHalf) || c.emission != "row" || !c.packed || c.fiber_pooling != "event"
        || c.profile || c.operator_profile || c.work_count || count < 1 || count > 16)
      throw std::invalid_argument("scale placement requires FP32/FP16, packed row Emit, event pooling and operator profiling disabled");
    if (c.runtime.device_spec == "auto") throw std::invalid_argument("benchmark requires an explicit backend");
    // Initialize the vendor queue policy before resolving/initializing NPU.
    // An explicitly supplied value remains available for stack qualification.
    if (c.runtime.device_spec.rfind("npu",0)==0 && !std::getenv("TASK_QUEUE_ENABLE"))
      if (setenv("TASK_QUEUE_ENABLE","0",0)) throw std::runtime_error("cannot select NPU task queue policy");
    const bool half=c.runtime.dtype==at::kHalf;
    c.reference_float32=reference_float32;
    if(reference_float32 && (!half || !c.check))throw std::invalid_argument("FP32 reference requires FP16 --check 1");
    if(half && transport!="resident")throw std::invalid_argument("FP16 requires resident transport");
    if((check_atol>=0 || check_rtol>=0) && (!half || !c.check))
      throw std::invalid_argument("custom tolerances require --dtype float16 --check 1");
    c.check_atol=half?(check_atol>=0?check_atol:1e-3):1e-6;
    c.check_rtol=half?(check_rtol>=0?check_rtol:2e-2):1e-5;
    if(half && !scoring_flags.count("--loss-scale"))training.loss_scale=128.;
    if(!training.steps && scoring_flags.count("--loss-scale"))throw std::invalid_argument("--loss-scale requires training");
    c.runtime.allow_npu_float16=half; // Qualified client capability, not a public-session fallback.
    const auto device = portable_torch::resolve_device(c.runtime);
    used_npu = device.type()==c10::DeviceType::PrivateUse1;
    if (!device.is_cpu() && device.type() != c10::DeviceType::PrivateUse1)
      throw std::invalid_argument("this placement client is qualified for CPU/NPU only");
    if (!device.is_cpu() && c.head_workers != 1)
      throw std::invalid_argument("NPU requires head-workers1; use device matrix kernels");
    if(!device.is_cpu() && (training.backward_threads!=1 || training.optimizer_threads!=1))
      throw std::invalid_argument("phase thread options require CPU training");
    at::set_num_threads(c.threads); at::set_num_interop_threads(1);
    auto topology = read_topology(c.topology);
    if(seen_vjp && !c.check)throw std::invalid_argument("--vjp-policy requires --check 1");
    if (scoring_flags.count("--reference-read-dtype") && !c.check)
      throw std::invalid_argument("--reference-read-dtype requires --check 1");
    const bool resident = transport == "resident";
    scoring.validate(device, resident);training.validate(c);
    profile.validate(device,c.steps,training.steps);
    if((ranking_device!="cpu" && ranking_device!="model") || (event_device!="cpu" && event_device!="model")
        || (!resident && (ranking_device!="cpu" || event_device!="cpu")))
      throw std::invalid_argument("ranking/event device must be cpu|model; model requires resident transport");
    if(ranking_device=="model" && !device.is_cpu() && scoring.dtype!=at::kFloat)
      throw std::invalid_argument("NPU ranking requires FP32 descriptors; no implicit conversion");
    const auto device_contexts = accelerator_scale::initialize_devices(device, count);
    if (!std::filesystem::create_directories(c.runtime.output_dir)) throw std::invalid_argument("new output directory required");
    if (c.check) accelerator_scale::check(c, topology, device, count, policy, resident,vjp_policy=="basis-conditioned",scoring,reference_fp64,ranking_device,event_device);
    if(c.check && training.steps) {
      accelerator_scale::Placement policies;policies.resident=resident;policies.policy=policy;
      policies.scoring=scoring;policies.ranking_device=ranking_device;policies.event_device=event_device;
      accelerator_scale::check_training(c,topology,device,count,policies,training);
    }
    const auto started = Clock::now();
    portable_experiment::MetricsJsonlWriter writer(std::filesystem::path(c.runtime.output_dir)/"metrics.jsonl", c.run_id);
    portable_torch::seed_runtime(at::Device(at::kCPU), c.runtime.seed);
    at::AutoGradMode grad(c.grad);
    const auto setup_start = Clock::now();
    auto f = fixture(c, topology);
    accelerator_scale::configure_scoring(f, scoring);
    auto placement = accelerator_scale::place(f, device, count, policy, resident);
    placement.scoring = scoring;placement.ranking_device=ranking_device;placement.event_device=event_device;
    placement.profile=profile;
    tide::Options options; options.workers = c.workers; options.packed = true; options.trace = false;
    options.full_autograd = c.full_autograd; options.aggregate_autograd = c.aggregate_autograd;
    options.parallel_regions = c.parallel_regions; options.compact_events = c.compact_events;
    options.defer_state_release = c.defer_state_release; options.packed_sources = c.packed_sources; options.batch_next = c.batch_next;

    accelerator_scale::synchronize(placement);
    const auto construction = seconds(setup_start);
    std::cout << "MODEL parameters=" << std::fixed << f.inventory.at("parameters")
              << " devices=" << count << " construction_seconds=" << construction << '\n' << std::flush;
    std::ofstream mapping(std::filesystem::path(c.runtime.output_dir)/"placement.json");
    mapping << "{\"dtype\":\"" << portable_torch::dtype_name(c.runtime.dtype) << "\",\"node_shards\":[";
    for (size_t i = 0; i < placement.node_device.size(); ++i) mapping << (i ? "," : "") << placement.node_device[i];
    mapping << "],\"device_resident_state\":" << (resident ? "true" : "false")
            << ",\"cpu_fp64_read\":" << (scoring.read_device=="cpu" && scoring.dtype==at::kDouble ? "true" : "false")
            << ",\"read_device\":\"" << scoring.read_device << "\",\"read_dtype\":\"" << scoring.dtype_name()
            << "\",\"control_device\":\"" << scoring.control_device << "\",\"event_scheduler\":\"" << event_device << "\",\"ranking\":\"" << ranking_device << "\",\"policy\":\"" << policy
            << "\",\"physical_edges\":" << placement.edges << ",\"cut_edges\":" << placement.cut_edges
            << ",\"node_load_limit_bytes\":" << placement.node_load_limit << ",\"devices\":" << count
            << ",\"embedding_device_index\":" << int(f.embedding.device().index())
            << ",\"head_device_index\":" << int(f.head.device().index()) << ",\"parameter_bytes\":[";
    for(size_t i=0;i<placement.parameter_bytes.size();++i) mapping << (i?",":"") << placement.parameter_bytes[i];
    mapping << "]}\n";
    mapping.close(); if (!mapping) throw std::runtime_error("placement publication failed");
    if(training.steps) {
      accelerator_scale::train(c,topology,f,placement,options,training,writer,construction);
      return 0;
    }
    accelerator_scale::Execution cursor(std::move(f.graph),std::move(f.model),options,c.batch,placement);
    tide::DenseLinear head(c.head_workers);
    at::Tensor previous_logits;
    for (Index token = 0; token < c.steps; ++token) {
      previous_logits = at::Tensor();
      auto ids = at::remainder(at::arange(c.batch, at::TensorOptions().dtype(at::kLong))*3+token*7, c.vocab);
      accelerator_scale::synchronize(placement);
      accelerator_scale::transfers.reset(); accelerator_scale::reset_memory(placement);
      accelerator_scale::ProfileScope trace(profile,placement.devices,token,"token");
      const auto begin = Clock::now();
      auto embeddings = accelerator_scale::embed(f.embedding, ids, !resident);
      std::vector<tide::External> inputs;
      for (Index b = 0; b < c.batch; ++b) inputs.push_back({b,0,token,token*(topology.layers+1),embeddings[b]});
      auto result = cursor.advance(inputs, (token+1)*(topology.layers+1), (token+1)*(topology.layers+1));
      std::vector<at::Tensor> hidden(c.batch, at::zeros({c.width}, embeddings.options()));
      for (const auto& output : result.outputs) hidden.at(output.batch) = output.value;
      previous_logits = accelerator_scale::project(at::stack(hidden), f.head, !resident, &head);
      accelerator_scale::synchronize(placement);
      const auto elapsed = seconds(begin);
      trace.finish();
      if (!at::isfinite(previous_logits).all().item<bool>() || previous_logits.requires_grad() != c.grad)
        throw std::runtime_error("nonfinite logits or incorrect grad-forward mode");
      auto metrics = accelerator_scale::transfers.metrics();
      auto mem = accelerator_scale::memory(placement); metrics.insert(mem.begin(),mem.end());
      metrics.insert({{"perf/token_seconds",elapsed}, {"perf/ms_per_sample_token",elapsed*1000/c.batch},
        {"perf/sample_tokens_per_second",c.batch/elapsed}, {"perf/construction_seconds",construction},
        {"memory/process_peak_rss_bytes",rss()}, {"check/logits_sum",previous_logits.detach().to(at::kCPU).to(at::kDouble).sum().item<double>()},
        {"check/logits_requires_grad",c.grad ? 1. : 0.}, {"runtime/devices",double(count)},
        {"runtime/workers",double(c.workers)}, {"runtime/aten_threads",double(at::get_num_threads())},
        {"runtime/head_workers",double(c.head_workers)},
        {"runtime/effective_device_workers",double(resident && !device.is_cpu()?std::min(c.workers,count):c.workers)},
        {"runtime/npu_task_queue",device.is_cpu()?-1.:double(std::atoi(std::getenv("TASK_QUEUE_ENABLE")))},
        {"work/readout_rows",double(result.outputs.size())}});
      for (const auto& [key,value] : f.inventory) metrics["model/"+key] = value;
      for (const auto& [key,value] : result.stats) metrics["work/"+key] = value;
      writer.Write(token,metrics,seconds(started),{{"phase",std::string(token<c.warmup ? "warmup" : "measure")},
        {"dtype",portable_torch::dtype_name(c.runtime.dtype)},{"memory",c.memory},{"grad",c.grad},{"transport",transport},
        {"partition_policy",policy},{"read_device",scoring.read_device},{"read_dtype",scoring.dtype_name()},
        {"control_device",scoring.control_device},{"event_scheduler",event_device},{"ranking_device",ranking_device}});
      std::cout << "STEP " << token << " seconds=" << elapsed << " rows=" << result.outputs.size() << '\n' << std::flush;
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
int main(int argc,char** argv) {
  const auto code=run(argc,argv);
  // All benchmark tensors and worker pools have been destroyed. Release the
  // vendor runtime while its process-static event/cache managers still exist.
  try {if(used_npu)accelerator_scale::finalize();}
  catch(const std::exception& error) {std::cerr << "NPU finalization: " << error.what() << '\n';return 1;}
  return code;
}
