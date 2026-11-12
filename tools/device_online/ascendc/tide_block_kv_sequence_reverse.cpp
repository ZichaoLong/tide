#include "block_kv_sequence_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_sequence_reverse(GM_ADDR fields,GM_ADDR write_i,
    GM_ADDR old_i,GM_ADDR proposal_i,GM_ADDR mapping,GM_ADDR old_bar,GM_ADDR old_roots,
    GM_ADDR proposal_bar,GM_ADDR proposal_roots,GM_ADDR final_bar,GM_ADDR final_roots,
    GM_ADDR initial_bar,GM_ADDR initial_roots,GM_ADDR write_bar,GM_ADDR write_roots,GM_ADDR error,
    int64_t batch,int64_t length,int64_t nodes,int64_t width,int64_t write_width,int64_t capacity,int64_t count){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_sequence_device::reverse((__gm__ int64_t*)fields,(__gm__ int64_t*)write_i,
    (__gm__ int64_t*)old_i,(__gm__ int64_t*)proposal_i,(__gm__ int64_t*)mapping,
    (__gm__ float*)old_bar,(__gm__ uint8_t*)old_roots,(__gm__ float*)proposal_bar,(__gm__ uint8_t*)proposal_roots,
    (__gm__ float*)final_bar,(__gm__ uint8_t*)final_roots,(__gm__ float*)initial_bar,(__gm__ uint8_t*)initial_roots,
    (__gm__ float*)write_bar,(__gm__ uint8_t*)write_roots,(__gm__ int32_t*)error,
    batch,length,nodes,width,write_width,capacity,count);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
