#include "weighted_pool_math.h"
extern "C" __global__ __aicore__ void tide_weighted_pool_prepare(GM_ADDR w,GM_ADDR v,GM_ADDR logits,GM_ADDR lengths,GM_ADDR present,GM_ADDR clean_w,GM_ADDR clean_v,GM_ADDR scores,GM_ADDR meta,GM_ADDR error,int64_t rows,int64_t capacity,int64_t width,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_pool_device::prepare((__gm__ float*)w,(__gm__ float*)v,(__gm__ float*)logits,(__gm__ I*)lengths,(__gm__ uint8_t*)present,(__gm__ float*)clean_w,(__gm__ float*)clean_v,(__gm__ float*)scores,(__gm__ I*)meta,(__gm__ int32_t*)error,rows,capacity,width,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
