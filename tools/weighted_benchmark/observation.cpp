#include "observation.h"
#include "portable_torch/runtime.hpp"
#include <iostream>
#include <sys/resource.h>
#include <system_error>
#if PORTABLE_TORCH_ENABLE_NPU
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#endif
#if PORTABLE_TORCH_ENABLE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif
namespace weighted_benchmark {
namespace {
template<class Stats> void fields(Json& row,const Stats& s){
  row["allocated_bytes"]=s.allocated_bytes[0].current;
  row["reserved_bytes"]=s.reserved_bytes[0].current;
  row["peak_allocated_bytes"]=s.allocated_bytes[0].peak;
  row["peak_reserved_bytes"]=s.reserved_bytes[0].peak;
}
void device_memory(Json& row,at::Device device,bool reset){
  if(device.is_cpu())return;
  row["device"]=device.str();
#if PORTABLE_TORCH_ENABLE_NPU
  if(device.type()==c10::DeviceType::PrivateUse1){
    fields(row,c10_npu::NPUCachingAllocator::getDeviceStats(device.index()));
    if(reset)c10_npu::NPUCachingAllocator::resetPeakStats(device.index());return;
  }
#endif
#if PORTABLE_TORCH_ENABLE_CUDA
  if(device.is_cuda()){
    fields(row,c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()));
    if(reset)c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());return;
  }
#endif
  throw std::invalid_argument("benchmark allocator unavailable for requested device");
}
}
void runtime_probe(at::Device device){
  const auto x=at::ones({1,1},at::TensorOptions().dtype(at::kFloat).device(device));
  const auto y=at::matmul(x,x);portable_torch::synchronize(device);
  std::cout<<"TIDE_WEIGHTED_RUNTIME_READY_V1\n"<<std::flush;
}
void capture_memory(Json& report,at::Device device,const std::string& phase,bool reset){
  rusage usage{};if(getrusage(RUSAGE_SELF,&usage))throw std::system_error(errno,std::generic_category(),"getrusage");
#ifdef __APPLE__
  const auto rss=int64_t(usage.ru_maxrss);
#else
  const auto rss=int64_t(usage.ru_maxrss)*1024;
#endif
  if(!report.contains("memory"))report["memory"]={
    {"schema","tide-weighted-memory-v1"},
    {"scope","process allocator only; excludes untracked vendor/driver memory; RSS is process-lifetime peak"},
    {"sampling","outside step timers; peaks reset after runtime_ready, construction and warmup"},
    {"phases",Json::array()}};
  Json row{{"phase",phase},{"process_lifetime_peak_rss_bytes",rss}};device_memory(row,device,reset);
  report["memory"]["phases"].push_back(row);report["process_lifetime_peak_rss_bytes"]=rss;
}
}
