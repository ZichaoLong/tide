#include "weighted_rotary_reduce.h"
extern "C" __global__ __aicore__ void tide_weighted_rotary_unfold(GM_ADDR cosine,GM_ADDR sine,GM_ADDR quadrant,GM_ADDR error,int64_t count){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_rotary_device::unfold((__gm__ float*)cosine,(__gm__ float*)sine,(__gm__ int32_t*)quadrant,(__gm__ int32_t*)error,count);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
