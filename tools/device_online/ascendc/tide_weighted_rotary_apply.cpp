#include "weighted_rotary_math.h"
extern "C" __global__ __aicore__ void tide_weighted_rotary_apply(GM_ADDR input,GM_ADDR cosine,GM_ADDR sine,GM_ADDR roots,GM_ADDR output,GM_ADDR error,int64_t rows,int64_t capacity,int64_t width,int64_t pairs,int64_t reverse){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_rotary_device::apply((__gm__ float*)input,(__gm__ float*)cosine,(__gm__ float*)sine,(__gm__ uint8_t*)roots,(__gm__ float*)output,(__gm__ int32_t*)error,rows,capacity,width,pairs,reverse);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
