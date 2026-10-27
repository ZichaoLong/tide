#include "weighted_kv_full_math.h"
extern "C" __global__ __aicore__ void tide_weighted_full_plan(GM_ADDR active,GM_ADDR owners,GM_ADDR output,GM_ADDR error,int64_t rows,int64_t banks){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_full_device::plan((__gm__ uint8_t*)active,(__gm__ I*)owners,(__gm__ I*)output,(__gm__ int32_t*)error,rows,banks);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
