#include "tide/device.h"

namespace tide {
bool supported_payload(const Tensor& value) {
  if (!value.defined() || value.layout() != at::kStrided
      || (value.scalar_type() != at::kFloat && value.scalar_type() != at::kDouble)) return false;
  if (value.device().is_cpu()) return true;
#if PORTABLE_TORCH_ENABLE_CUDA
  if (value.device().is_cuda()) return true;
#endif
#if PORTABLE_TORCH_ENABLE_NPU
  if (value.device().type() == c10::DeviceType::PrivateUse1 && value.scalar_type() == at::kFloat) return true;
#endif
  return false;
}
bool supported_kernel_payload(const Tensor& value) {
  if (supported_payload(value)) return true;
  if (!value.defined() || value.layout()!=at::kStrided || value.scalar_type()!=at::kHalf) return false;
  if (value.device().is_cpu()) return true;
#if PORTABLE_TORCH_ENABLE_CUDA
  if (value.device().is_cuda()) return true;
#endif
#if PORTABLE_TORCH_ENABLE_NPU
  if (value.device().type()==c10::DeviceType::PrivateUse1) return true;
#endif
  return false;
}
const char* execution_backend() noexcept {
#if PORTABLE_TORCH_ENABLE_CUDA
  return "cuda";
#elif PORTABLE_TORCH_ENABLE_NPU
  return "npu";
#else
  return "cpu";
#endif
}
}  // namespace tide
