#include "runtime.h"
#ifdef TIDE_WITH_NPU
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

namespace tide_native {
AutocastGuard::AutocastGuard(at::DeviceType device, bool enabled)
    : device_(device), enabled_(at::autocast::is_autocast_enabled(device)),
      cache_(at::autocast::is_autocast_cache_enabled()),
      dtype_(at::autocast::get_autocast_dtype(device)) {
  at::autocast::set_autocast_cache_enabled(false);
  at::autocast::set_autocast_dtype(device, at::kBFloat16);
  at::autocast::set_autocast_enabled(device, enabled);
}
AutocastGuard::~AutocastGuard() {
  at::autocast::set_autocast_enabled(device_, enabled_);
  at::autocast::set_autocast_dtype(device_, dtype_);
  at::autocast::set_autocast_cache_enabled(cache_);
}
Runtime::Runtime(const std::string& requested) {
  if (requested == "cpu" || requested == "auto") return;
#ifdef TIDE_WITH_NPU
  if (requested.rfind("npu:", 0) == 0) {
    torch_npu::init_npu(requested);
    initialized_ = true;
    device_ = at::Device(requested);
    return;
  }
#endif
  TORCH_CHECK(false, "device ", requested,
              " is unavailable in this build; select CPU or build with TIDE_NATIVE_BACKEND=NPU");
}
Runtime::~Runtime() {
#ifdef TIDE_WITH_NPU
  if (initialized_) {
    try { torch_npu::finalize_npu(); } catch (...) {}
  }
#endif
}
void Runtime::synchronize() const {
#ifdef TIDE_WITH_NPU
  if (initialized_) torch::npu::synchronize(device_.index());
#endif
}
std::vector<std::optional<c10::Stream>> Runtime::streams(int workers, bool pool) const {
  std::vector<std::optional<c10::Stream>> result(workers);
#ifdef TIDE_WITH_NPU
  if (initialized_) {
    auto stream = c10_npu::getCurrentNPUStream(device_.index()).unwrap();
    for (auto& item : result) item = pool
        ? c10_npu::getStreamFromPool(false, device_.index()).unwrap() : stream;
    return result;
  }
#endif
  TORCH_CHECK(!pool, "pool streams require an NPU build and explicit NPU device");
  return result;
}
}  // namespace tide_native
