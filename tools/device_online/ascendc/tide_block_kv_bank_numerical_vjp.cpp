#include "block_kv_bank_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_bank_numerical_vjp(GM_ADDR fields,GM_ADDR valid,GM_ADDR keys,GM_ADDR values,GM_ADDR summaries,GM_ADDR weights,GM_ADDR initial,GM_ADDR writes,GM_ADDR error,int64_t batch,int64_t length,int64_t nodes,int64_t capacity,int64_t entries,int64_t width,int64_t write_width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_bank_reverse_device::numerical((__gm__ int64_t*)fields,(__gm__ uint8_t*)valid,(__gm__ float*)keys,(__gm__ float*)values,(__gm__ float*)summaries,(__gm__ float*)weights,(__gm__ float*)initial,(__gm__ float*)writes,(__gm__ int32_t*)error,batch,length,nodes,capacity,entries,width,write_width);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
