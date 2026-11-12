#include "block_kv_sequence_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_sequence(GM_ADDR fields,GM_ADDR initial,GM_ADDR initial_i,
    GM_ADDR write,GM_ADDR write_i,GM_ADDR times,GM_ADDR old,GM_ADDR old_i,GM_ADDR proposal,GM_ADDR proposal_i,
    GM_ADDR final,GM_ADDR final_i,GM_ADDR mapping,GM_ADDR error,
    int64_t batch,int64_t length,int64_t nodes,int64_t width,int64_t write_width,int64_t capacity,int64_t count,int64_t horizon){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_sequence_device::forward((__gm__ int64_t*)fields,(__gm__ float*)initial,(__gm__ int64_t*)initial_i,
    (__gm__ float*)write,(__gm__ int64_t*)write_i,(__gm__ int64_t*)times,(__gm__ float*)old,(__gm__ int64_t*)old_i,
    (__gm__ float*)proposal,(__gm__ int64_t*)proposal_i,(__gm__ float*)final,(__gm__ int64_t*)final_i,
    (__gm__ int64_t*)mapping,(__gm__ int32_t*)error,batch,length,nodes,width,write_width,capacity,count,horizon);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
