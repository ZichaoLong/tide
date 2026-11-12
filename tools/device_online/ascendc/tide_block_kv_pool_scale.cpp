#include "block_kv_pool_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_pool_scale(GM_ADDR weights,GM_ADDR total,GM_ADDR probability,GM_ADDR meta,GM_ADDR coefficients,GM_ADDR amplitude,GM_ADDR error,
    int64_t rows,int64_t entries,int64_t attention){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_pool_device::scale((__gm__ float*)weights,(__gm__ float*)total,(__gm__ float*)probability,(__gm__ int64_t*)meta,(__gm__ float*)coefficients,(__gm__ float*)amplitude,(__gm__ int32_t*)error,rows,entries,attention);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
