#include "memory.h"
#include "consumer.h"
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <c10/core/DeviceGuard.h>
#if PORTABLE_TORCH_ENABLE_NPU
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <third_party/acl/inc/acl/acl.h>
#include <dlfcn.h>
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
int64_t device_fields(std::ostream& out,at::Device d) {
#if PORTABLE_TORCH_ENABLE_NPU
  if(d.type()==c10::DeviceType::PrivateUse1){const auto s=c10_npu::NPUCachingAllocator::getDeviceStats(d.index());fields(out,s);return s.allocated_bytes[0].peak;}
#endif
#if PORTABLE_TORCH_ENABLE_CUDA
  if(d.is_cuda()){const auto s=c10::cuda::CUDACachingAllocator::getDeviceStats(d.index());fields(out,s);return s.allocated_bytes[0].peak;}
#endif
  throw std::invalid_argument("consumer allocator counters unavailable for explicit device");
}
}
DeviceMemoryInfo device_memory_info(at::Device d) {
  if(d.index()<0)throw std::invalid_argument("memory admission needs explicit logical device");
  c10::DeviceGuard guard(d);size_t free=0,total=0;int64_t allocated=0;
#if PORTABLE_TORCH_ENABLE_NPU
  if(d.type()==c10::DeviceType::PrivateUse1) {
    // Resolve the already initialized runtime, rather than searching for or
    // loading another toolkit library. The matched SDK supplies the signature.
    auto get_memory=reinterpret_cast<decltype(&aclrtGetMemInfo)>(dlsym(RTLD_DEFAULT,"aclrtGetMemInfo"));
    if(!get_memory||get_memory(ACL_HBM_MEM,&free,&total)!=ACL_SUCCESS)throw std::runtime_error("cannot query NPU driver free memory");
    allocated=c10_npu::NPUCachingAllocator::getDeviceStats(d.index()).allocated_bytes[0].current;
    return {d,int64_t(free),int64_t(total),allocated};
  }
#endif
#if PORTABLE_TORCH_ENABLE_CUDA
  if(d.is_cuda()) {
    if(cudaMemGetInfo(&free,&total)!=cudaSuccess)throw std::runtime_error("cannot query CUDA driver free memory");
    allocated=c10::cuda::CUDACachingAllocator::getDeviceStats(d.index()).allocated_bytes[0].current;
    return {d,int64_t(free),int64_t(total),allocated};
  }
#endif
  throw std::invalid_argument("driver memory admission unavailable for device");
}
MemoryRecord::MemoryRecord(std::vector<at::Device> devices) {
  std::set<std::string> seen;
  for(auto d:devices)if(!d.is_cpu()) {
    if(d.index()<0||!seen.insert(d.str()).second)throw std::invalid_argument("memory record needs distinct logical devices");
    devices_.push_back(d);reset(d);
    initial_.push_back(device_memory_info(d).allocated);peak_growth_.push_back(0);
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
  bool first=true;for(size_t i=0;i<devices_.size();++i) {
    auto d=devices_[i];if(!first)out<<',';first=false;out<<"{\"device\":"<<quoted(d.str());
    peak_growth_[i]=std::max(peak_growth_[i],device_fields(out,d)-initial_[i]);out<<'}';
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
