#include "block_read_selection_math.h"
extern "C" __global__ __aicore__ void tide_block_node_gate(GM_ADDR tasks,GM_ADDR selected,GM_ADDR metadata,
    GM_ADDR active,GM_ADDR emitted,GM_ADDR error,int64_t slots,int64_t capacity,int64_t frames,int64_t nodes){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_selection_device::gate((__gm__ int64_t*)tasks,(__gm__ int64_t*)selected,(__gm__ int64_t*)metadata,
    (__gm__ uint8_t*)active,(__gm__ uint8_t*)emitted,(__gm__ int32_t*)error,slots,capacity,frames,nodes);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
