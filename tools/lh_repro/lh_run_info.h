#pragma once
// Diagnostics only. Called outside the original IOCortexNet::think timer.
#include <ATen/Parallel.h>
#include <ATen/Version.h>
#include <torch/version.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <sys/resource.h>
#include <sys/utsname.h>

namespace LHRunInfo {
using Clock = std::chrono::steady_clock;

inline void line(const char* kind, const json& value) {
    std::cout << kind << " " << value.dump() << std::endl;
}

inline bool arguments(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--device" &&
        std::string(argv[2]) == "cpu" && std::string(argv[3]) == "--dtype" &&
        std::string(argv[4]) == "float32") return true;
    std::cerr << "Usage: " << argv[0] << " --device cpu --dtype float32\n";
    return false;
}

inline json resources() {
    json out = json::object();
    struct rusage r {};
    if (getrusage(RUSAGE_SELF, &r) == 0) {
        out["peak_rss_gib_so_far"] = double(r.ru_maxrss) / (1024.0*1024.0); // Linux KiB
        out["cpu_user_seconds"] = r.ru_utime.tv_sec + r.ru_utime.tv_usec / 1e6;
        out["cpu_system_seconds"] = r.ru_stime.tv_sec + r.ru_stime.tv_usec / 1e6;
    }
    std::ifstream status("/proc/self/status");
    std::string text;
    while (std::getline(status, text)) {
        std::istringstream input(text);
        std::string key, value;
        input >> key >> value;
        if (key == "VmRSS:") out["rss_gib"] = std::stod(value)/(1024.0*1024.0);
        if (key == "Threads:") out["process_threads"] = std::stoll(value);
        if (key == "Cpus_allowed_list:") out["cpu_affinity"] = value;
    }
    return out;
}

inline void environment() {
    struct utsname u {};
    const bool ok = uname(&u) == 0;
    json env = {{"lh_revision", "a10fdb1883fccd63ec21e36cc9cffa294c63c9e2"},
        {"torch_headers", TORCH_VERSION}, {"compiler", __VERSION__},
        {"arch", ok ? u.machine : "unknown"}, {"kernel", ok ? u.release : "unknown"},
        {"aten_threads", at::get_num_threads()}, {"interop_threads", at::get_num_interop_threads()},
        {"device", "cpu"}, {"dtype", "float32"}, {"seed_policy", "original unseeded RNG"},
        {"backward", false}, {"optimizer", false}, {"node_parallel", true}};
#ifdef NOGRAD
    env["mode"] = "nograd";
#else
    env["mode"] = "grad-forward";
#endif
#ifdef _GLIBCXX_USE_CXX11_ABI
    env["cxx11_abi"] = _GLIBCXX_USE_CXX11_ABI;
#endif
    for (const char* name : {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "KMP_AFFINITY"}) {
        const char* value = std::getenv(name);
        env["thread_env"][name] = value ? json(value) : json(nullptr);
    }
    env["parallel_info"] = at::get_parallel_info();
    env["torch_build_config"] = at::show_config();
    line("LH_ENV", env);
}

inline void model(IOCortexNet& net, const GraphData& graph, const json& cfg,
                  int64_t batch, int64_t selectnum) {
    json out = {{"config", cfg}, {"batch", batch}, {"selectnum", selectnum},
        {"static_nodes_per_cortex", net.inet->sourceN}, {"attention_mode", "CROSSBATCH"},
        {"initial_kv_capacity", AL::BatchPtrKVHidden::init_cache_size},
        {"grad_enabled", at::GradMode::is_enabled()},
        {"edges", {{"inputA", graph.inputA.nnz}, {"outputA", graph.outputA.nnz},
                   {"ioA", graph.ioA.nnz}, {"oiA", graph.oiA.nnz}}}};
    int64_t count = 0, bytes = 0;
    for (const auto& parameter : net.parameters()) {
        if (parameter.device().type() != torch::kCPU || parameter.scalar_type() != torch::kFloat32)
            throw std::runtime_error("this reproduction requires CPU FP32 parameters");
        count += parameter.numel();
        bytes += parameter.numel()*parameter.element_size();
    }
    for (const auto& child : net.named_children()) {
        int64_t n = 0;
        for (const auto& parameter : child.value()->parameters()) n += parameter.numel();
        out["registered_parameters_by_child"][child.key()] = n;
    }
    int64_t root_count = 0;
    for (const auto& parameter : net.parameters(false)) root_count += parameter.numel();
    out["registered_parameters_at_root"] = root_count;
    out["registered_parameter_elements"] = count;
    out["registered_parameter_gib"] = bytes/double(int64_t(1)<<30);
    out["resources_after_construction"] = resources();
    line("LH_MODEL", out);
}

struct Times {
    int64_t batch;
    std::vector<double> ms;
    explicit Times(int64_t b) : batch(b) {}
    void step(int64_t t, Clock::time_point start, Clock::time_point end, const Tensor& logits) {
        const double elapsed = std::chrono::duration<double, std::milli>(end-start).count();
        ms.push_back(elapsed);
        json out = {{"token_index", t}, {"outer_think_ms_per_batch", elapsed},
            {"outer_ms_per_sample_token", elapsed/batch}, {"resources", resources()},
            {"logits_requires_grad", logits.requires_grad()},
            {"logits_shape", {logits.size(0), logits.size(1)}}};
        line("LH_STEP", out);
    }
    ~Times() {
        if (ms.empty()) return;
        const double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
        line("LH_DONE", {{"steps", ms.size()}, {"batch", batch},
            {"outer_mean_ms_per_sample_token", sum/(batch*ms.size())},
            {"outer_aggregate_tokens_per_second", 1000.0*batch*ms.size()/sum},
            {"resources", resources()}, {"backward", false}});
    }
};
} // namespace LHRunInfo
