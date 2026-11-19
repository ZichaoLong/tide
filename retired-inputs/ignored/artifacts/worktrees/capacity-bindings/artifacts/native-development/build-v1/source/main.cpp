#include "bundle.h"
#include "runtime.h"
#include "build_identity.h"
#include <torch/optim/adamw.h>
#include <torch/serialize.h>
#include <torch/autograd.h>
#include <torch/version.h>
#include <ATen/Parallel.h>
#include <ATen/ThreadLocalState.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <sys/resource.h>
#include <sched.h>

using namespace tide_native;
namespace {
struct Options {
  std::string device, dtype = "float32", mode = "train", streams = "same";
  std::filesystem::path input, output, resume;
  int workers = 1, inner = 1, repeats = 1, warmup = 0;
};
int integer(const std::string& text, int lower, int upper) {
  TORCH_CHECK(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
              "expected a nonnegative integer, got ", text);
  auto result = std::stoll(text);
  TORCH_CHECK(result >= lower && result <= upper, "integer outside allowed range: ", text);
  return static_cast<int>(result);
}
Options parse(int argc, char** argv) {
  Options o;
  std::map<std::string, std::string> flags;
  for (int i = 1; i < argc; ++i) {
    std::string name = argv[i], value;
    const auto equal = name.find('=');
    if (equal != std::string::npos) { value = name.substr(equal + 1); name.resize(equal); }
    else { TORCH_CHECK(i + 1 < argc, "missing argument for ", name); value = argv[++i]; }
    TORCH_CHECK(flags.emplace(name, value).second, "duplicate option: ", name);
  }
  for (const auto& item : flags) {
    const auto& k = item.first;
    const auto& v = item.second;
    if (k == "--device") o.device = v;
    else if (k == "--dtype") o.dtype = v;
    else if (k == "--mode") o.mode = v;
    else if (k == "--streams") o.streams = v;
    else if (k == "--input-dir") o.input = v;
    else if (k == "--output-dir") o.output = v;
    else if (k == "--resume") o.resume = v;
    else if (k == "--workers") o.workers = integer(v, 1, 4);
    else if (k == "--inner-threads") o.inner = integer(v, 1, 256);
    else if (k == "--repeats") o.repeats = integer(v, 1, 10000);
    else if (k == "--warmup") o.warmup = integer(v, 0, 10000);
    else if (k != "--device-index") TORCH_CHECK(false, "unknown argument: ", k);
  }
  TORCH_CHECK(!o.device.empty(), "--device is required before tensor/runtime initialization");
  if (flags.count("--device-index")) {
    TORCH_CHECK(o.device == "npu", "--device-index requires unsuffixed --device npu");
    o.device += ":" + std::to_string(integer(flags.at("--device-index"), 0, 32767));
  }
  if (o.device == "npu") o.device = "npu:0";
  if (o.device.rfind("npu:", 0) == 0) integer(o.device.substr(4), 0, 32767);
  else TORCH_CHECK(o.device == "cpu" || o.device == "auto", "unsupported --device ", o.device);
  TORCH_CHECK(o.dtype == "float32" || o.dtype == "bfloat16", "--dtype must be float32 or bfloat16");
  TORCH_CHECK(o.mode == "inference" || o.mode == "grad_forward" || o.mode == "train", "invalid --mode");
  TORCH_CHECK(o.streams == "same" || o.streams == "pool", "--streams must be same or pool");
  TORCH_CHECK(o.workers == 1 || o.workers == 2 || o.workers == 4, "--workers must be 1, 2 or 4");
  TORCH_CHECK(!o.input.empty() && !o.output.empty(), "--input-dir and --output-dir are required");
  TORCH_CHECK(!std::filesystem::exists(o.output), "--output-dir must not exist");
  TORCH_CHECK(o.resume.empty() || o.mode == "train", "--resume requires train mode");
  return o;
}

// A persistent pool makes scheduling overhead part of samples without including
// thread creation. Each dispatch installs the caller's complete TLS separately.
class Pool {
 public:
  explicit Pool(int workers) : workers_(workers) {
    if (workers == 1) return;
    for (int i = 0; i < workers; ++i) threads_.emplace_back([this, i] {
      size_t seen = 0;
      while (true) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          start_.wait(lock, [&] { return stopping_ || generation_ != seen; });
          if (stopping_) return;
          seen = generation_;
        }
        try {
          for (int task; (task = next_.fetch_add(1)) < count_;) function_(task, i);
        } catch (...) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!error_) error_ = std::current_exception();
          next_.store(count_);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (++finished_ == workers_) done_.notify_one();
      }
    });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    start_.notify_all();
    for (auto& thread : threads_) thread.join();
  }
  void run(int count, const std::function<void(int, int)>& function) {
    if (workers_ == 1) { for (int i = 0; i < count; ++i) function(i, 0); return; }
    std::unique_lock<std::mutex> lock(mutex_);
    function_ = function; count_ = count; next_ = 0; finished_ = 0; error_ = nullptr;
    ++generation_;
    start_.notify_all();
    done_.wait(lock, [&] { return finished_ == workers_; });
    if (error_) std::rethrow_exception(error_);
  }
 private:
  int workers_, count_ = 0, finished_ = 0;
  size_t generation_ = 0;
  bool stopping_ = false;
  std::atomic<int> next_{0};
  std::mutex mutex_;
  std::condition_variable start_, done_;
  std::vector<std::thread> threads_;
  std::function<void(int, int)> function_;
  std::exception_ptr error_;
};

struct Run {
  Region region;
  torch::optim::AdamW optimizer;
  std::vector<Tensors> input;
  std::vector<at::Tensor> output;
  int64_t progress = 0;
  Run(const Tensors& source, const Runtime& runtime, bool gradients)
      : region(source, runtime.device()), optimizer(region.parameters(),
          torch::optim::AdamWOptions(.001).eps(.001).weight_decay(.01)) {
    for (int64_t task = 0; task < source.at("z").size(0); ++task) {
      Tensors values;
      for (const auto& name : {"z", "score", "w", "target"}) {
        values[name] = source.at(name)[task].to(runtime.device()).clone();
        if (std::string(name) != "target") values[name].set_requires_grad(gradients);
      }
      input.push_back(std::move(values));
    }
    output.resize(input.size());
  }
  void restore(const std::filesystem::path& path) {
    torch::serialize::InputArchive root, parameters, optim;
    root.load_from(path.string(), region.values.at("readout").device());
    root.read("parameters", parameters);
    torch::NoGradGuard guard;
    for (auto& item : region.values) {
      at::Tensor value;
      parameters.read(item.first, value);
      TORCH_CHECK(value.sizes() == item.second.sizes(), "checkpoint parameter shape mismatch");
      item.second.copy_(value);
    }
    root.read("optimizer", optim); optimizer.load(optim);
    at::Tensor step; root.read("progress", step); progress = step.item<int64_t>();
  }
  void checkpoint(const std::filesystem::path& path) {
    torch::serialize::OutputArchive root, parameters, optim;
    for (const auto& item : region.values) parameters.write(item.first, item.second.detach().cpu());
    root.write("parameters", parameters);
    optimizer.save(optim); root.write("optimizer", optim);
    root.write("progress", at::scalar_tensor(progress, at::TensorOptions().dtype(at::kLong)));
    auto stage = path.string() + ".staging";
    root.save_to(stage); publish_staged(stage, path);
  }
  void iteration(const Options& o, Pool& pool, const Runtime& runtime,
                 const std::vector<std::optional<c10::Stream>>& streams) {
    if (o.mode == "train") {
      optimizer.zero_grad(true);
      for (auto& task : input) for (const auto& name : {"z", "w", "score"}) task.at(name).mutable_grad() = at::Tensor();
    }
    at::ThreadLocalState caller_tls;
    pool.run(static_cast<int>(input.size()), [&](int task, int worker) {
      at::ThreadLocalStateGuard tls(caller_tls);
      c10::DeviceGuard device(runtime.device());
      c10::OptionalStreamGuard stream(streams[worker]);
      at::AutoGradMode grad(o.mode != "inference");
      auto& t = input[task];
      auto y = region.forward(t.at("z"), t.at("score"), t.at("w"), o.dtype == "bfloat16");
      TORCH_CHECK(y.requires_grad() == (o.mode != "inference"), "worker grad TLS mismatch");
      output[task] = y;
      if (o.mode == "train") {
        AutocastGuard disabled(runtime.device().type(), false);
        auto loss = (y * t.at("target")).sum() / static_cast<double>(y.numel() * input.size());
        loss.backward();
      }
    });
    // In the optional pool-stream probe, explicitly finish every task before
    // the main-stream optimizer consumes shared gradients. Same-stream runs
    // rely on ordered submission and the outer timing synchronization.
    if (o.streams == "pool") runtime.synchronize();
    if (o.mode == "train") { optimizer.step(); ++progress; }
  }
  Tensors results(const Options& o) {
    Tensors result;
    for (size_t i = 0; i < output.size(); ++i) {
      result["output." + std::to_string(i)] = output[i];
      if (o.mode == "train") for (const auto& name : {"z", "w", "score"})
        result["grad." + std::to_string(i) + "." + name] = input[i].at(name).grad();
    }
    for (const auto& item : region.values) {
      result["parameter." + item.first] = item.second;
      if (o.mode == "train") {
        TORCH_CHECK(item.second.grad().defined(), "missing shared parameter gradient: ", item.first);
        result["grad.parameter." + item.first] = item.second.grad();
        auto& state = dynamic_cast<torch::optim::AdamWParamState&>(*optimizer.state().at(item.second.unsafeGetTensorImpl()));
        result["moment1." + item.first] = state.exp_avg();
        result["moment2." + item.first] = state.exp_avg_sq();
      }
    }
    return result;
  }
};
void validate(const Tensors& values) {
  TORCH_CHECK(values.size() == 13, "fixture needs exactly nine parameters and z/score/w/target");
  const auto shape = values.at("z").sizes();
  TORCH_CHECK(shape.size() == 5 && shape[0] <= 64, "z must have five axes and at most 64 tasks");
  auto scalar = shape.vec(); scalar.pop_back();
  for (const auto& name : {"score", "w", "target"})
    TORCH_CHECK(values.at(name).sizes().vec() == scalar, "fixture shape mismatch: ", name);
  TORCH_CHECK((values.at("w") >= 0).all().item<bool>(), "priors must be nonnegative");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Tide regional scoring subpath (not a complete Tide model)\n"
        "--device cpu|auto|npu[:N] --dtype float32|bfloat16 --input-dir DIR --output-dir NEW_DIR\n"
        "--mode inference|grad_forward|train --workers 1|2|4 --inner-threads N\n"
        "--warmup N --repeats N --streams same|pool --resume CHECKPOINT\n"
        "auto deliberately resolves to CPU; NPU requires an explicit choice and NPU build.\n";
      return 0;
    }
    auto o = parse(argc, argv);
    at::set_num_threads(o.inner); at::set_num_interop_threads(1);
    auto source = read_bundle(o.input); validate(source);
    Runtime runtime(o.device);
    auto streams = runtime.streams(o.workers, o.streams == "pool");
    Pool pool(o.workers);
    if (o.warmup) {
      Run warm(source, runtime, o.mode != "inference");
      if (!o.resume.empty()) warm.restore(o.resume);
      runtime.synchronize();
      for (int i = 0; i < o.warmup; ++i) warm.iteration(o, pool, runtime, streams);
      runtime.synchronize();
    }
    Run measured(source, runtime, o.mode != "inference");
    if (!o.resume.empty()) measured.restore(o.resume);
    const auto initial_progress = measured.progress;
    std::vector<double> samples;
    for (int i = 0; i < o.repeats; ++i) {
      runtime.synchronize();
      auto start = std::chrono::steady_clock::now();
      measured.iteration(o, pool, runtime, streams);
      runtime.synchronize();
      samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    TORCH_CHECK(std::filesystem::create_directories(o.output), "failed to create new output directory");
    write_bundle(o.output / "tensors", measured.results(o));
    if (o.mode == "train") measured.checkpoint(o.output / "checkpoint.pt");
    int64_t parameters = 0;
    for (const auto& p : measured.region.parameters()) parameters += p.numel();
    struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
    std::ostringstream json;
    json.precision(17);
    json << "{\n\"schema\":1,\"state\":\"completed\",\"client\":\"standalone-libtorch\","
         << "\"scope\":\"regional Attention+FFN scoring subpath\",\"build_backend\":" << json_string(TIDE_NATIVE_BACKEND)
         << ",\"build_revision\":" << json_string(TIDE_BUILD_REVISION) << ",\"build_dirty\":" << (TIDE_BUILD_DIRTY ? "true" : "false")
         << ",\"compiler\":" << json_string(TIDE_BUILD_COMPILER) << ",\"build_type\":" << json_string(TIDE_BUILD_TYPE)
         << ",\"arch\":" << json_string(TIDE_BUILD_ARCH) << ",\"cxx11_abi\":" << _GLIBCXX_USE_CXX11_ABI
         << ",\"torch_version\":" << json_string(TORCH_VERSION) << ",\"torch_cmake\":" << json_string(TIDE_TORCH_PREFIX)
         << ",\"npu_cmake\":" << json_string(TIDE_NPU_PREFIX) << ",\"requested_device\":" << json_string(o.device)
         << ",\"device\":" << json_string(runtime.device().str()) << ",\"resolution_reason\":"
         << json_string(o.device == "auto" ? "intentional_auto_cpu" : "explicit_device")
         << ",\"dtype\":" << json_string(o.dtype) << ",\"mode\":" << json_string(o.mode)
         << ",\"workers\":" << o.workers << ",\"intra_threads\":" << at::get_num_threads()
         << ",\"interop_threads\":" << at::get_num_interop_threads() << ",\"streams\":" << json_string(o.streams)
         << ",\"stream_ids\":[";
    for (size_t i = 0; i < streams.size(); ++i) { if (i) json << ','; json << (streams[i] ? std::to_string(streams[i]->id()) : "null"); }
    json << "],\"task_shape\":[";
    for (int i = 0; i < source.at("z").dim(); ++i) { if (i) json << ','; json << source.at("z").size(i); }
    json << "],\"parameters\":" << parameters << ",\"progress_before\":" << initial_progress << ",\"progress_after\":" << measured.progress
         << ",\"warmup\":" << o.warmup << ",\"samples_ms\":[";
    for (size_t i = 0; i < samples.size(); ++i) { if (i) json << ','; json << samples[i]; }
    json << "],\"cpu_affinity\":[";
    cpu_set_t affinity; CPU_ZERO(&affinity); sched_getaffinity(0, sizeof(affinity), &affinity);
    bool first = true;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &affinity)) { if (!first) json << ','; first = false; json << i; }
    json << "],\"process_peak_rss_bytes\":" << usage.ru_maxrss * 1024
         << ",\"arithmetic\":{\"parameter_gradient_optimizer\":\"float32\",\"projection\":" << json_string(o.dtype)
         << ",\"attention_products_normalization_scalar_readout\":\"float32\",\"autocast_cache\":false}"
         << ",\"device_operator_placement\":\"not_profiled\",\"timing_boundary\":\"synchronized region tasks; train includes zero_grad, backward, AdamW; excludes fixture/model load, pool creation and serialization\""
         << ",\"sample_policy\":\"fixed inputs; train parameters and AdamW continue; warmup uses a discarded owner\",\"optimizer\":{\"name\":\"AdamW\",\"lr\":0.001,\"eps\":0.001,\"weight_decay\":0.01},\"argv\":[";
    for (int i = 0; i < argc; ++i) { if (i) json << ','; json << json_string(argv[i]); }
    json << "]}\n";
    atomic_text(o.output / "result.json", json.str());
    std::cout << json.str();
    runtime.synchronize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "tide-regional: " << error.what() << '\n';
    return 1;
  }
}
