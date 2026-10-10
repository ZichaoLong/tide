#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tide::device_online::cuda_backend {
// CUDA numerical kernels use explicit storage and no dynamic allocation.
// Dense matrix multiplication is submitted to cuBLAS separately.
enum Type { F32,F16,I64,I32,U8,BOOL };
enum Op { Copy,Add,Mul,Div,Equal,Less,And,Softplus,Sigmoid,Tanh,Relu,Silu,
          TanhBackward,SiluBackward,ReluBackward,Cast };
struct Layout {
  int rank=0;
  int64_t size[8]{},a[8]{},b[8]{};
  int64_t count=0;
};
void elementwise(cudaStream_t,Op,const void*,Type,const void*,Type,void*,Type,Layout);
void reduce(cudaStream_t,const void*,Type,void*,int64_t outer,int64_t axis,int64_t inner,bool softmax);
void normalize(cudaStream_t,const void*,Type,void*,float*,int64_t rows,int64_t width,double epsilon,bool rms);
void permute(cudaStream_t,const void*,Type,void*,Layout);
void index(cudaStream_t,const void*,Type,const int64_t*,void*,int64_t outer,int64_t source_axis,
           int64_t target_axis,int64_t inner,bool copy,int32_t* error);
void route(cudaStream_t,const int32_t*,const int32_t*,int32_t targets,int32_t blocks,
           cudaGraphConditionalHandle loop,cudaGraphConditionalHandle pc,int32_t* error);
}
