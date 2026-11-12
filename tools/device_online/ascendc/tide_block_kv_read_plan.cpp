#include "block_kv_read_plan_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_read_plan(GM_ADDR mapping,GM_ADDR meta,GM_ADDR bank_weights,GM_ADDR stamps,
    GM_ADDR times,GM_ADDR input,GM_ADDR owners,GM_ADDR present,GM_ADDR weights,GM_ADDR decay,GM_ADDR visible,GM_ADDR ages,GM_ADDR selected,
    GM_ADDR qowners,GM_ADDR eowners,GM_ADDR fowners,GM_ADDR clean,GM_ADDR error,
    int64_t batch,int64_t length,int64_t nodes,int64_t capacity,int64_t entries,int64_t width,int64_t banks,int64_t horizon,int64_t query,int64_t full,float beta){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_read_device::plan((__gm__ int64_t*)mapping,(__gm__ int64_t*)meta,(__gm__ float*)bank_weights,(__gm__ int64_t*)stamps,
    (__gm__ int64_t*)times,(__gm__ float*)input,(__gm__ int64_t*)owners,(__gm__ uint8_t*)present,(__gm__ float*)weights,(__gm__ float*)decay,
    (__gm__ uint8_t*)visible,(__gm__ int64_t*)ages,(__gm__ uint8_t*)selected,(__gm__ int64_t*)qowners,(__gm__ int64_t*)eowners,(__gm__ int64_t*)fowners,
    (__gm__ float*)clean,(__gm__ int32_t*)error,batch,length,nodes,capacity,entries,width,banks,horizon,query,full,beta);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
