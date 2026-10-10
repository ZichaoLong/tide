#include "program_internal.h"
#include <chrono>
#include <stdexcept>

namespace tide::device_online {
namespace cuda_backend {
void check(cudaError_t status,const char* operation) {
  if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void check_blas(cublasStatus_t status,const char* operation) {
  if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(operation)+": cuBLAS status "+std::to_string(status));
}
Type type(at::ScalarType type) {
  switch(type){case at::kFloat:return F32;case at::kHalf:return F16;case at::kLong:return I64;
    case at::kInt:return I32;case at::kByte:return U8;case at::kBool:return BOOL;
    default:throw std::invalid_argument("CUDA resident dtype unavailable");}
}
template<class Query> void wait(Query query,int32_t ms) {
  if(ms<=0)throw std::invalid_argument("CUDA completion timeout must be positive");
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(ms);
  for(;;) {
    const auto status=query();
    if(status==cudaSuccess)return;
    if(status!=cudaErrorNotReady)check(status,"query CUDA completion");
    if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("CUDA resident completion timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
void wait_event(cudaEvent_t event,int32_t ms){wait([=]{return cudaEventQuery(event);},ms);}
void wait_stream(cudaStream_t stream,int32_t ms){wait([=]{return cudaStreamQuery(stream);},ms);}
}
void check_device_launch(int status,const char* operation) {cuda_backend::check(cudaError_t(status),operation);}
void validate_kernel_device(c10::Device device) {
  if(!device.is_cuda()||device.index()<0)throw std::invalid_argument("CUDA resident requires an explicit CUDA device");
  c10::cuda::CUDAGuard guard(device);
  int driver=0;cuda_backend::check(cudaDriverGetVersion(&driver),"query CUDA driver");
  if(driver<12080)throw std::invalid_argument("CUDA resident requires driver support for CUDA 12.8 conditional SWITCH graphs");
  cudaDeviceProp properties{};cuda_backend::check(cudaGetDeviceProperties(&properties,device.index()),"query CUDA properties");
  if(properties.major<8)throw std::invalid_argument("CUDA resident profile requires compute capability 8.0 or newer");
}
}
