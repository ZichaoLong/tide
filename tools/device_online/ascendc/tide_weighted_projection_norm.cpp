#include "weighted_projection_norm.h"
extern "C" __global__ __aicore__ void tide_weighted_projection_norm(GM_ADDR normal,GM_ADDR rstd,
    GM_ADDR gradient,GM_ADDR roots,GM_ADDR error,int64_t rows,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<float> cache;cache.SetGlobalBuffer((__gm__ float*)normal);
  AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_projection_device::reverse_norm((__gm__ float*)normal,(__gm__ float*)rstd,
    (__gm__ float*)gradient,(__gm__ uint8_t*)roots,(__gm__ int32_t*)error,rows,width);
  AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
