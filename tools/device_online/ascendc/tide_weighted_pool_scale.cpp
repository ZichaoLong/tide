#include "weighted_pool_math.h"
extern "C" __global__ __aicore__ void tide_weighted_pool_scale(GM_ADDR w,GM_ADDR total,GM_ADDR prob,GM_ADDR meta,GM_ADDR coeff,GM_ADDR amplitude,GM_ADDR error,int64_t rows,int64_t capacity,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_pool_device::scale((__gm__ float*)w,(__gm__ float*)total,(__gm__ float*)prob,(__gm__ I*)meta,(__gm__ float*)coeff,(__gm__ float*)amplitude,(__gm__ int32_t*)error,rows,capacity,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
