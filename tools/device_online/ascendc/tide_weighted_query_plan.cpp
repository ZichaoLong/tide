#include "weighted_kv_query_math.h"
extern "C" __global__ __aicore__ void tide_weighted_query_plan(GM_ADDR weights,GM_ADDR lengths,GM_ADDR present,GM_ADDR owners,GM_ADDR input,GM_ADDR clean,GM_ADDR query_owners,GM_ADDR entry_owners,GM_ADDR error,int64_t rows,int64_t capacity,int64_t width,int64_t banks){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_query_device::plan((__gm__ float*)weights,(__gm__ I*)lengths,(__gm__ uint8_t*)present,(__gm__ I*)owners,(__gm__ float*)input,(__gm__ float*)clean,(__gm__ I*)query_owners,(__gm__ I*)entry_owners,(__gm__ int32_t*)error,rows,capacity,width,banks);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
