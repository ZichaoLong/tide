#include "portable_torch/runtime.hpp"
#include "tide/device.h"
#include <ATen/Context.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <charconv>
#include <mutex>
#include <stdexcept>
#if PORTABLE_TORCH_ENABLE_CUDA
#include <torch/cuda.h>
#endif
#if TIDE_NPU_STANDALONE
#include <torch_npu/torch_npu.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#endif

namespace portable_torch {
namespace {
int index_of(const std::string& request, const std::string& backend) {
  if (request == backend) return 0;
  const auto text = request.substr(backend.size()+1);
  int index = -1;
  const auto result = std::from_chars(text.data(), text.data()+text.size(), index);
  if (result.ec != std::errc{} || result.ptr != text.data()+text.size() || index < 0 || index >= 32768)
    throw std::invalid_argument("invalid nonnegative logical device index");
  return index;
}
#if TIDE_NPU_STANDALONE
struct NpuShutdown {
  ~NpuShutdown() { try { torch_npu::finalize_npu(); } catch (...) {} }
};
torch::Device initialize_npu(int index) {
  if (index >= c10_npu::device_count()) throw std::runtime_error("NPU logical device is unavailable");
  static std::once_flag initialized;
  std::call_once(initialized, [index] {
    torch_npu::init_npu(static_cast<c10::DeviceIndex>(index));
    static NpuShutdown shutdown;
  });
  auto device = torch::Device(c10::DeviceType::PrivateUse1, index);
  c10::impl::VirtualGuardImpl(device.type()).setDevice(device);
  return device;
}
#endif
}  // namespace

torch::Device resolve_device(const RuntimeOptions& options) {
  const auto& request = options.device_spec;
  if (request == "cpu") return torch::Device(torch::kCPU);
  if (request == "auto") {
#if PORTABLE_TORCH_ENABLE_CUDA
    if (torch::cuda::is_available() && torch::cuda::device_count()) return torch::Device(torch::kCUDA, 0);
#endif
#if TIDE_NPU_STANDALONE
    if (c10_npu::device_count()) {
      if (options.dtype != torch::kFloat32) throw std::invalid_argument("NPU requires float32");
      return initialize_npu(0);
    }
#endif
    return torch::Device(torch::kCPU);
  }
  if (request == "cuda" || request.rfind("cuda:", 0) == 0) {
    const int index = index_of(request, "cuda");
#if PORTABLE_TORCH_ENABLE_CUDA
    if (!torch::cuda::is_available() || index >= torch::cuda::device_count())
      throw std::runtime_error("CUDA was explicitly requested but the logical device is unavailable");
    return torch::Device(torch::kCUDA, index);
#else
    (void)index;
    throw std::runtime_error("CUDA requested but this binary was compiled without CUDA support");
#endif
  }
  if (request == "npu" || request.rfind("npu:", 0) == 0) {
    const int index = index_of(request, "npu");
#if TIDE_NPU_STANDALONE
    if (options.dtype != torch::kFloat32) throw std::invalid_argument("NPU requires float32");
    return initialize_npu(index);
#else
    (void)index;
    throw std::runtime_error("NPU requested without a standalone libtorch_npu runtime; rebuild with TIDE_BACKEND=NPU and TIDE_NPU_RUNTIME=standalone");
#endif
  }
  throw std::invalid_argument("unsupported device: " + request);
}

std::string resolution_reason(const RuntimeOptions& options, const torch::Device& device) {
  const std::string name = device.is_cpu() ? "cpu" : device.is_cuda() ? "cuda" : "npu";
  if (options.device_spec != "auto") return "explicit:" + name;
  return device.is_cpu() ? "auto:no-accelerator-selected-cpu" : "auto:single-visible-" + name;
}

void seed_runtime(const torch::Device& device, std::uint64_t seed) {
  for (const auto& target : device.is_cpu() ? std::vector<torch::Device>{device}
                                          : std::vector<torch::Device>{torch::Device(torch::kCPU), device}) {
    auto generator =
#if TIDE_NPU_STANDALONE
      target.type() == c10::DeviceType::PrivateUse1
        ? at_npu::detail::getDefaultNPUGenerator(target.index()) :
#endif
        at::globalContext().defaultGenerator(target);
    std::lock_guard<std::mutex> lock(generator.mutex());
    generator.set_current_seed(seed);
  }
}

void synchronize(const torch::Device& device) {
  if (device.is_cpu()) return;
#if PORTABLE_TORCH_ENABLE_CUDA
  if (device.is_cuda()) { torch::cuda::synchronize(device.index()); return; }
#endif
#if TIDE_NPU_STANDALONE
  if (device.type() == c10::DeviceType::PrivateUse1) { torch::npu::synchronize(device.index()); return; }
#endif
  throw std::runtime_error("synchronization unavailable for " + device.str());
}
const char* compiled_backend() noexcept { return tide::execution_backend(); }
std::string dtype_name(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32: return "float32";
    case torch::kFloat64: return "float64";
    case torch::kFloat16: return "float16";
    case torch::kBFloat16: return "bfloat16";
    default: return c10::toString(dtype);
  }
}
}  // namespace portable_torch
