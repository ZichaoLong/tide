#include "memory.h"
#include "consumer.h"
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#if PORTABLE_TORCH_ENABLE_NPU
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#endif
#if PORTABLE_TORCH_ENABLE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif
namespace tide_flow {
namespace {
void reset(at::Device d) {
#if PORTABLE_TORCH_ENABLE_NPU
  if(d.type()==c10::DeviceType::PrivateUse1){c10_npu::NPUCachingAllocator::resetPeakStats(d.index());return;}
#endif
#if PORTABLE_TORCH_ENABLE_CUDA
  if(d.is_cuda()){c10::cuda::CUDACachingAllocator::resetPeakStats(d.index());return;}
#endif
  throw std::invalid_argument("consumer allocator counters unavailable for explicit device");
}
template<class Stats> void fields(std::ostream& out,const Stats& s) {
  out<<",\"allocated_bytes\":"<<s.allocated_bytes[0].current
     <<",\"peak_allocated_bytes\":"<<s.allocated_bytes[0].peak
     <<",\"reserved_bytes\":"<<s.reserved_bytes[0].current
     <<",\"peak_reserved_bytes\":"<<s.reserved_bytes[0].peak;
}
void device_fields(std::ostream& out,at::Device d) {
#if PORTABLE_TORCH_ENABLE_NPU
  if(d.type()==c10::DeviceType::PrivateUse1){fields(out,c10_npu::NPUCachingAllocator::getDeviceStats(d.index()));return;}
#endif
#if PORTABLE_TORCH_ENABLE_CUDA
  if(d.is_cuda()){fields(out,c10::cuda::CUDACachingAllocator::getDeviceStats(d.index()));return;}
#endif
  throw std::invalid_argument("consumer allocator counters unavailable for explicit device");
}
}
MemoryRecord::MemoryRecord(std::vector<at::Device> devices) {
  std::set<std::string> seen;
  for(auto d:devices)if(!d.is_cpu()) {
    if(d.index()<0||!seen.insert(d.str()).second)throw std::invalid_argument("memory record needs distinct logical devices");
    devices_.push_back(d);reset(d);
  }
  capture("initial",false);
}
void MemoryRecord::capture(const std::string& phase,bool reset_peak) {
  struct rusage usage{};
  if(getrusage(RUSAGE_SELF,&usage))throw std::runtime_error("cannot read consumer peak RSS");
#ifdef __APPLE__
  const int64_t rss=usage.ru_maxrss;
#else
  const int64_t rss=int64_t(usage.ru_maxrss)*1024;
#endif
  std::ostringstream out;out<<"{\"phase\":"<<quoted(phase)<<",\"cpu_peak_rss_bytes\":"<<rss<<",\"devices\":[";
  bool first=true;for(auto d:devices_) {
    if(!first)out<<',';first=false;out<<"{\"device\":"<<quoted(d.str());device_fields(out,d);out<<'}';
  }
  out<<"]}";phases_.push_back(out.str());
  if(reset_peak)for(auto d:devices_)reset(d);
}
std::string MemoryRecord::json() const {
  std::ostringstream out;out<<"{\"schema\":\"tide-consumer-memory-v1\","
    "\"scope\":\"per-process allocator per logical accelerator; excludes untracked vendor/driver memory; CPU RSS is process-lifetime peak\","
    "\"sampling\":\"phase boundaries outside step timers; allocator peaks reset after construction and warmup; initial setup included in construction\",\"phases\":[";
  bool first=true;for(const auto& phase:phases_){if(!first)out<<',';first=false;out<<phase;}out<<"]}";return out.str();
}
}
