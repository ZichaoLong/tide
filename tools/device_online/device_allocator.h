#pragma once
#include "device_backend.h"
#if TIDE_RESIDENT_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
namespace tide::device_online {namespace allocator=c10::cuda::CUDACachingAllocator;}
#else
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
namespace tide::device_online {namespace allocator=c10_npu::NPUCachingAllocator;}
#endif
