#pragma once
#include <ATen/ATen.h>
#include <string>
#include <vector>
namespace tide_flow {
struct DeviceMemoryInfo {at::Device device;int64_t free,total,allocated;};
DeviceMemoryInfo device_memory_info(at::Device);
// Benchmark-owned process counters. No reads inside graph/event/kernel loops.
// Phase boundaries reset allocator peaks; callers must own those counters.
class MemoryRecord {
 public:
  explicit MemoryRecord(std::vector<at::Device>);
  void capture(const std::string& phase,bool reset_peak=true);
  std::string json() const;
  const std::vector<int64_t>& peak_growth() const {return peak_growth_;}
 private:
  std::vector<at::Device> devices_;
  std::vector<std::string> phases_;
  std::vector<int64_t> initial_,peak_growth_;
};
}
