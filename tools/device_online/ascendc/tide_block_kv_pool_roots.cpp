#include "block_kv_pool_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_pool_roots(GM_ADDR meta,GM_ADDR total,GM_ADDR vector_bar,GM_ADDR total_bar,GM_ADDR roots,GM_ADDR clean_bar,GM_ADDR scaled_bar,GM_ADDR stat,GM_ADDR connected,GM_ADDR error,
    int64_t rows,int64_t width,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_pool_device::roots((__gm__ int64_t*)meta,(__gm__ float*)total,(__gm__ float*)vector_bar,(__gm__ float*)total_bar,(__gm__ uint8_t*)roots,(__gm__ float*)clean_bar,(__gm__ float*)scaled_bar,(__gm__ float*)stat,(__gm__ uint8_t*)connected,(__gm__ int32_t*)error,rows,width,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
