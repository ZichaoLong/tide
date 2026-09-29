#pragma once
#include <ATen/core/Tensor.h>
#include <memory>
#include <string>
#include <vector>

namespace accelerator_scale {
struct ProfileConfig {
  int64_t step = -1;
  std::string phase = "token", output;
  bool enabled() const { return step >= 0; }
  void validate(at::Device, int64_t tokens, int64_t updates) const;
};
// Optional, phase-scoped CANN adapter. No vendor dependency in CPU builds.
class ProfileScope {
 public:
  ProfileScope(const ProfileConfig&, const std::vector<at::Device>&, int64_t step,
               const std::string& phase);
  ~ProfileScope();
  void finish();
  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace accelerator_scale
