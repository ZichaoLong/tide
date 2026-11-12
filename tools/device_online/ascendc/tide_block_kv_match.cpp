#include "block_kv_match_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_match(GM_ADDR query,GM_ADDR keys,GM_ADDR visible,GM_ADDR cosine,GM_ADDR sine,GM_ADDR logits,GM_ADDR bars,GM_ADDR dq,GM_ADDR dk,GM_ADDR error,int64_t owners,int64_t queries,int64_t entries,int64_t width,int64_t pairs,int64_t reverse,float scale){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_match_device::match((__gm__ float*)query,(__gm__ float*)keys,(__gm__ uint8_t*)visible,(__gm__ float*)cosine,(__gm__ float*)sine,(__gm__ float*)logits,(__gm__ float*)bars,(__gm__ float*)dq,(__gm__ float*)dk,(__gm__ int32_t*)error,owners,queries,entries,width,pairs,reverse,scale);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
