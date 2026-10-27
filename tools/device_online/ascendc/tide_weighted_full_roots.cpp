#include "weighted_kv_full_math.h"
extern "C" __global__ __aicore__ void tide_weighted_full_roots(GM_ADDR root,GM_ADDR flags,GM_ADDR pool,GM_ADDR query,GM_ADDR window,GM_ADDR error,int64_t rows,int64_t heads,int64_t mode){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_full_device::roots((__gm__ uint8_t*)root,(__gm__ uint8_t*)flags,(__gm__ uint8_t*)pool,(__gm__ uint8_t*)query,(__gm__ uint8_t*)window,(__gm__ int32_t*)error,rows,heads,mode);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
