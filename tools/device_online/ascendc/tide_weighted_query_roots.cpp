#include "weighted_kv_query_math.h"
extern "C" __global__ __aicore__ void tide_weighted_query_roots(GM_ADDR pool,GM_ADDR owners,GM_ADDR query,GM_ADDR keys,GM_ADDR values,GM_ADDR window,GM_ADDR error,int64_t rows,int64_t capacity){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_query_device::roots((__gm__ uint8_t*)pool,(__gm__ I*)owners,(__gm__ uint8_t*)query,(__gm__ uint8_t*)keys,(__gm__ uint8_t*)values,(__gm__ uint8_t*)window,(__gm__ int32_t*)error,rows,capacity);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
