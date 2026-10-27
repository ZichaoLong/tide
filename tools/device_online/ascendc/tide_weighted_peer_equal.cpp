// Replicated optimizer identity includes exact FP32 slots and int64 counters.
// Byte comparison avoids a float projection of counters or NaN equality rules.
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_weighted_peer_equal(GM_ADDR left,GM_ADDR right,
    GM_ADDR error,int64_t bytes){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto e=(__gm__ int32_t*)error;
  if(!*e)for(int64_t i=0;i<bytes;++i)if(((__gm__ uint8_t*)left)[i]!=((__gm__ uint8_t*)right)[i]){*e=25;break;}
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
