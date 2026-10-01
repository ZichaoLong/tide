#pragma once
#include <ATen/ATen.h>
#include <string>
#include <vector>
namespace tide_flow {
// Benchmark-owned process counters. No reads inside graph/event/kernel loops.
// Phase boundaries reset allocator peaks; callers must own those counters.
class MemoryRecord {
 public:
  explicit MemoryRecord(std::vector<at::Device>);
  void capture(const std::string& phase,bool reset_peak=true);
  std::string json() const;
 private:
  std::vector<at::Device> devices_;
  std::vector<std::string> phases_;
};
}
