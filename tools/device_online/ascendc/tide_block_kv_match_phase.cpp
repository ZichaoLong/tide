#include "block_kv_match_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_match_phase(GM_ADDR ages,GM_ADDR visible,GM_ADDR frequencies,GM_ADDR output,GM_ADDR bias,GM_ADDR error,int64_t count,int64_t pairs,float scale,float slope){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_match_device::phase((__gm__ int64_t*)ages,(__gm__ uint8_t*)visible,(__gm__ float*)frequencies,(__gm__ float*)output,(__gm__ float*)bias,(__gm__ int32_t*)error,count,pairs,scale,slope);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
