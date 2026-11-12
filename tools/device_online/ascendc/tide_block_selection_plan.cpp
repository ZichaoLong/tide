#include "block_read_selection_math.h"
extern "C" __global__ __aicore__ void tide_block_selection_plan(GM_ADDR present,GM_ADDR times,GM_ADDR source_owners,
    GM_ADDR offsets,GM_ADDR members,GM_ADDR owners,GM_ADDR saved_present,GM_ADDR saved_times,
    GM_ADDR order,GM_ADDR inverse,GM_ADDR ids,GM_ADDR region_times,GM_ADDR lengths,GM_ADDR node_region_row,GM_ADDR error,
    int64_t frames,int64_t nodes,int64_t regions,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  using I=int64_t;tide_block_selection_device::plan((__gm__ uint8_t*)present,(__gm__ I*)times,(__gm__ I*)source_owners,
    (__gm__ I*)offsets,(__gm__ I*)members,(__gm__ I*)owners,(__gm__ uint8_t*)saved_present,(__gm__ I*)saved_times,
    (__gm__ I*)order,(__gm__ I*)inverse,(__gm__ I*)ids,(__gm__ I*)region_times,(__gm__ I*)lengths,(__gm__ I*)node_region_row,
    (__gm__ int32_t*)error,frames,nodes,regions,width);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
