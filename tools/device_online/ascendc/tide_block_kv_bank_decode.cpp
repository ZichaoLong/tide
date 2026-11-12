#include "block_kv_bank_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_bank_decode(GM_ADDR fields,GM_ADDR initial,GM_ADDR initial_i,
    GM_ADDR keys,GM_ADDR values,GM_ADDR summaries,GM_ADDR weights,GM_ADDR mapping,GM_ADDR metadata,GM_ADDR raw,GM_ADDR error,
    int64_t batch,int64_t length,int64_t nodes,int64_t capacity,int64_t entries,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_bank_device::decode((__gm__ int64_t*)fields,(__gm__ float*)initial,(__gm__ int64_t*)initial_i,
    (__gm__ float*)keys,(__gm__ float*)values,(__gm__ float*)summaries,(__gm__ float*)weights,(__gm__ int64_t*)mapping,
    (__gm__ int64_t*)metadata,(__gm__ float*)raw,(__gm__ int32_t*)error,batch,length,nodes,capacity,entries,width);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
