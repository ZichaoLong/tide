#include "block_kv_read_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_read_roots(GM_ADDR flags,GM_ADDR weights,GM_ADDR present,GM_ADDR snapshot,GM_ADDR query,GM_ADDR keys,GM_ADDR values,GM_ADDR error,int64_t batch,int64_t length,int64_t nodes,int64_t entries,int64_t channels,int64_t kind){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_read_reverse_device::roots((__gm__ uint8_t*)flags,(__gm__ float*)weights,(__gm__ uint8_t*)present,(__gm__ uint8_t*)snapshot,(__gm__ uint8_t*)query,(__gm__ uint8_t*)keys,(__gm__ uint8_t*)values,(__gm__ int32_t*)error,batch,length,nodes,entries,channels,kind);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
