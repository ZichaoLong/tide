#include "block_kv_pool_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_pool_finish(GM_ADDR weights,GM_ADDR probability,GM_ADDR total,GM_ADDR dots,GM_ADDR mean_dot,GM_ADDR centered_dot,GM_ADDR anchor_dot,GM_ADDR stat,GM_ADDR visible,GM_ADDR connected,GM_ADDR dw,GM_ADDR dl,GM_ADDR error,
    int64_t rows,int64_t entries,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_pool_device::finish((__gm__ float*)weights,(__gm__ float*)probability,(__gm__ float*)total,(__gm__ float*)dots,(__gm__ float*)mean_dot,(__gm__ float*)centered_dot,(__gm__ float*)anchor_dot,(__gm__ float*)stat,(__gm__ uint8_t*)visible,(__gm__ uint8_t*)connected,(__gm__ float*)dw,(__gm__ float*)dl,(__gm__ int32_t*)error,rows,entries,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
