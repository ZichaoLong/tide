#include "weighted_pool_math.h"
extern "C" __global__ __aicore__ void tide_weighted_pool_reverse_prepare(GM_ADDR meta,GM_ADDR w,GM_ADDR v,GM_ADDR prob,GM_ADDR coeff,GM_ADDR total,GM_ADDR output,GM_ADDR bar,GM_ADDR stat,GM_ADDR roots,GM_ADDR clean_bar,GM_ADDR clean_total,GM_ADDR centered,GM_ADDR value_coeff,GM_ADDR connected,GM_ADDR error,int64_t rows,int64_t capacity,int64_t width,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_pool_device::reverse_prepare((__gm__ I*)meta,(__gm__ float*)w,(__gm__ float*)v,(__gm__ float*)prob,(__gm__ float*)coeff,(__gm__ float*)total,(__gm__ float*)output,(__gm__ float*)bar,(__gm__ float*)stat,(__gm__ uint8_t*)roots,(__gm__ float*)clean_bar,(__gm__ float*)clean_total,(__gm__ float*)centered,(__gm__ float*)value_coeff,(__gm__ uint8_t*)connected,(__gm__ int32_t*)error,rows,capacity,width,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
