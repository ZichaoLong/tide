#include "block_kv_bank_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_bank_plan(GM_ADDR fields,GM_ADDR initial,GM_ADDR initial_i,
    GM_ADDR write,GM_ADDR write_i,GM_ADDR times,GM_ADDR keys,GM_ADDR values,GM_ADDR summaries,GM_ADDR weights,
    GM_ADDR stamps,GM_ADDR valid,GM_ADDR written,GM_ADDR old_map,GM_ADDR proposal_map,GM_ADDR old_i,GM_ADDR proposal_i,GM_ADDR error,
    int64_t batch,int64_t length,int64_t nodes,int64_t capacity,int64_t d,int64_t ds,int64_t width,int64_t write_width,int64_t horizon,int64_t bound){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_bank_device::plan((__gm__ int64_t*)fields,(__gm__ float*)initial,(__gm__ int64_t*)initial_i,
    (__gm__ float*)write,(__gm__ int64_t*)write_i,(__gm__ int64_t*)times,(__gm__ float*)keys,(__gm__ float*)values,
    (__gm__ float*)summaries,(__gm__ float*)weights,(__gm__ int64_t*)stamps,(__gm__ uint8_t*)valid,(__gm__ uint8_t*)written,
    (__gm__ int64_t*)old_map,(__gm__ int64_t*)proposal_map,(__gm__ int64_t*)old_i,(__gm__ int64_t*)proposal_i,
    (__gm__ int32_t*)error,batch,length,nodes,capacity,d,ds,width,write_width,horizon,bound);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
