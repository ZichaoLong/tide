#pragma once
#include <ATen/ATen.h>
#include <functional>
#include <memory>
#include <vector>

namespace accelerator_scale {
class GraphReplay {
 public:
  explicit GraphReplay(at::Device);
  explicit GraphReplay(const std::vector<at::Device>&);
  ~GraphReplay();
  GraphReplay(const GraphReplay&)=delete;
  GraphReplay& operator=(const GraphReplay&)=delete;
  void capture(const std::function<void()>&);
  void replay();
  void synchronize();
  std::pair<size_t,int64_t> peer_inventory() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace accelerator_scale
