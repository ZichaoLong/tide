#include "weighted_aggregate_math.h"
extern "C" __global__ __aicore__ void tide_weighted_aggregate_norm_vjp(GM_ADDR normalized,GM_ADDR rstd,
    GM_ADDR gradient,GM_ADDR source_roots,GM_ADDR input_gradient,GM_ADDR row_errors,GM_ADDR error,
    int64_t rows,int64_t width,int64_t d){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_aggregate_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)error);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum())
    ((__gm__ int32_t*)row_errors)[row]=reverse_norm((__gm__ float*)normalized+row*width*d,(__gm__ float*)rstd+row*width,
      (__gm__ float*)gradient+row*width*d,(__gm__ uint8_t*)source_roots+row*width,width,d,
      (__gm__ float*)input_gradient+row*width*(d+1));
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
