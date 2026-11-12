#include "block_kv_read_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_read_weight_vjp(GM_ADDR bars,GM_ADDR decay,GM_ADDR visible,GM_ADDR output,GM_ADDR error,int64_t owners,int64_t queries,int64_t entries){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_read_reverse_device::weights((__gm__ float*)bars,(__gm__ float*)decay,(__gm__ uint8_t*)visible,(__gm__ float*)output,(__gm__ int32_t*)error,owners,queries,entries);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
