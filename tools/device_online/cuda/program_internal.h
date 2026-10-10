#pragma once
#include "../device_program.h"
#include "../control_flow.h"
#include "numeric.h"
#include "portable_torch/runtime.hpp"
#include <cublas_v2.h>
#include <c10/cuda/CUDAGuard.h>
#include <thread>
#include <limits>

namespace tide::device_online {
struct DeviceProgram::Impl {
  at::Device device;
  portable_torch::RuntimeResource resource;
  const std::thread::id thread=std::this_thread::get_id();
  cudaStream_t stream=nullptr;
  cudaEvent_t input_ready=nullptr,complete=nullptr,preceding=nullptr;
  cudaGraph_t graph=nullptr;
  cudaGraphExec_t executable=nullptr;
  cublasHandle_t blas=nullptr;
  std::vector<ControlInstruction> instructions;
  std::vector<std::function<void(cudaStream_t)>> operations;
  std::vector<at::Tensor> indexes,owners;
  at::Tensor error,workspace;
  size_t labels=0;
  bool finished=false,closed=false,failed=false,in_flight=false,completion_recorded=false;
  int64_t workspace_limit=std::numeric_limits<int64_t>::max();
  explicit Impl(at::Device);
  void initialize();
  void check() const;
  void building() const;
  void buffer(const at::Tensor&) const;
  void append(std::function<void(cudaStream_t)>,const std::vector<at::Tensor>&);
  void element(cuda_backend::Op,const at::Tensor&,const at::Tensor&,const at::Tensor&);
  void release();
};
namespace cuda_backend {
void check(cudaError_t,const char*);
void check_blas(cublasStatus_t,const char*);
Type type(at::ScalarType);
void wait_event(cudaEvent_t,int32_t milliseconds);
void wait_stream(cudaStream_t,int32_t milliseconds);
}
}
