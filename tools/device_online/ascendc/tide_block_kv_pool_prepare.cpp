#include "block_kv_pool_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_pool_prepare(GM_ADDR weights,GM_ADDR logits,GM_ADDR visible,GM_ADDR present,GM_ADDR clean,GM_ADDR scores,GM_ADDR meta,GM_ADDR error,
    int64_t rows,int64_t entries,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_pool_device::prepare((__gm__ float*)weights,(__gm__ float*)logits,(__gm__ uint8_t*)visible,(__gm__ uint8_t*)present,(__gm__ float*)clean,(__gm__ float*)scores,(__gm__ int64_t*)meta,(__gm__ int32_t*)error,rows,entries,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
