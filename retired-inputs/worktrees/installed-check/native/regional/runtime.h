#pragma once
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <c10/core/Stream.h>
#include <optional>
#include <string>
#include <vector>

namespace tide_native {
// Restore every setting: workers must not leak a previous task's policy.
class AutocastGuard {
 public:
  AutocastGuard(at::DeviceType device, bool enabled);
  ~AutocastGuard();
 private:
  at::DeviceType device_;
  bool enabled_, cache_;
  at::ScalarType dtype_;
};
class Runtime {
 public:
  explicit Runtime(const std::string& requested);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  const at::Device& device() const { return device_; }
  void synchronize() const;
  std::vector<std::optional<c10::Stream>> streams(int workers, bool pool) const;
 private:
  at::Device device_{at::kCPU};
  bool initialized_ = false;
};
}  // namespace tide_native
