#include "profiling.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#if PORTABLE_TORCH_ENABLE_NPU
#include <dlfcn.h>
#include <third_party/acl/inc/acl/acl_prof.h>
#endif

namespace accelerator_scale {
void ProfileConfig::validate(at::Device device, int64_t tokens, int64_t updates) const {
  if (!enabled()) {
    if (!output.empty() || phase != "token") throw std::invalid_argument("profiling requires --profile-step");
    return;
  }
  if (device.type() != c10::DeviceType::PrivateUse1 || output.empty())
    throw std::invalid_argument("profiling requires NPU and a new --profile-output directory");
  if ((updates && (phase != "forward" && phase != "backward" && phase != "optimizer"))
      || (!updates && phase != "token") || step >= (updates ? updates : tokens))
    throw std::invalid_argument("profile phase/step is outside the declared workload");
  if (std::filesystem::exists(output)) throw std::invalid_argument("new profiling directory required");
}
#if PORTABLE_TORCH_ENABLE_NPU
namespace {
void checked(aclError error, const char* name) {
  if (error != ACL_SUCCESS) throw std::runtime_error(std::string(name)+" failed: "+std::to_string(error));
}
template<class T> T symbol(void* library, const char* name) {
  auto address = dlsym(library, name);
  if (!address) throw std::runtime_error(std::string("missing CANN profiler API: ")+name);
  return reinterpret_cast<T>(address);
}
}
struct ProfileScope::Impl {
  void* library = nullptr;
  aclprofConfig* config = nullptr;
  bool initialized = false, started = false;
  decltype(&aclprofInit) init = nullptr;
  decltype(&aclprofCreateConfig) create = nullptr;
  decltype(&aclprofStart) start = nullptr;
  decltype(&aclprofStop) stop = nullptr;
  decltype(&aclprofDestroyConfig) destroy = nullptr;
  decltype(&aclprofFinalize) finalize = nullptr;
  Impl(const ProfileConfig& c, const std::vector<at::Device>& devices) {
    library = dlopen("libmsprofiler.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(std::string("cannot load CANN profiler: ")+dlerror());
    try {
      init=symbol<decltype(init)>(library,"aclprofInit");
      create=symbol<decltype(create)>(library,"aclprofCreateConfig");
      start=symbol<decltype(start)>(library,"aclprofStart");
      stop=symbol<decltype(stop)>(library,"aclprofStop");
      destroy=symbol<decltype(destroy)>(library,"aclprofDestroyConfig");
      finalize=symbol<decltype(finalize)>(library,"aclprofFinalize");
      if (!std::filesystem::create_directories(c.output)) throw std::runtime_error("new profiling directory required");
      checked(init(c.output.c_str(), c.output.size()), "aclprofInit"); initialized=true;
      std::vector<uint32_t> ids;
      for (auto device : devices) ids.push_back(static_cast<uint32_t>(device.index()));
      config=create(ids.data(), ids.size(), ACL_AICORE_NONE, nullptr,
                    ACL_PROF_TASK_TIME | ACL_PROF_RUNTIME_API | ACL_PROF_AICPU | ACL_PROF_ACL_API);
      if (!config) throw std::runtime_error("aclprofCreateConfig failed");
      checked(start(config), "aclprofStart"); started=true;
      std::cout << "PROFILE begin phase=" << c.phase << " step=" << c.step << '\n' << std::flush;
    } catch (...) { cleanup(); throw; }
  }
  void cleanup() noexcept {
    if (started && stop) stop(config);
    if (config && destroy) destroy(config);
    if (initialized && finalize) finalize();
    if (library) dlclose(library);
    started=false;config=nullptr;initialized=false;library=nullptr;
  }
  ~Impl() { cleanup(); }
  void finish() {
    if (started) { checked(stop(config), "aclprofStop"); started=false; }
    if (config) { checked(destroy(config), "aclprofDestroyConfig"); config=nullptr; }
    if (initialized) { checked(finalize(), "aclprofFinalize"); initialized=false; }
    std::cout << "PROFILE complete\n" << std::flush;
  }
};
#else
struct ProfileScope::Impl {};
#endif
ProfileScope::ProfileScope(const ProfileConfig& c, const std::vector<at::Device>& devices,
                           int64_t step, const std::string& phase) {
  if (!c.enabled() || c.step != step || c.phase != phase) return;
#if PORTABLE_TORCH_ENABLE_NPU
  impl_=std::make_unique<Impl>(c,devices);
#else
  throw std::invalid_argument("this build has no CANN profiler");
#endif
}
ProfileScope::~ProfileScope() = default;
void ProfileScope::finish() {
#if PORTABLE_TORCH_ENABLE_NPU
  if (impl_) { impl_->finish(); impl_.reset(); }
#endif
}
}  // namespace accelerator_scale
