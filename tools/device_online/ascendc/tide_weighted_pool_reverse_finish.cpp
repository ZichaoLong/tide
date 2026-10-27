#include "weighted_pool_math.h"
extern "C" __global__ __aicore__ void tide_weighted_pool_reverse_finish(GM_ADDR meta,GM_ADDR w,GM_ADDR prob,GM_ADDR total,GM_ADDR dots,GM_ADDR mean_dot,GM_ADDR stat,GM_ADDR connected,GM_ADDR dw,GM_ADDR dv,GM_ADDR dl,GM_ADDR error,int64_t rows,int64_t capacity,int64_t width,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_pool_device::reverse_finish((__gm__ I*)meta,(__gm__ float*)w,(__gm__ float*)prob,(__gm__ float*)total,(__gm__ float*)dots,(__gm__ float*)mean_dot,(__gm__ float*)stat,(__gm__ uint8_t*)connected,(__gm__ float*)dw,(__gm__ float*)dv,(__gm__ float*)dl,(__gm__ int32_t*)error,rows,capacity,width,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
