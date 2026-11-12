#include "block_read_selection_math.h"
extern "C" __global__ __aicore__ void tide_block_selection_flags(GM_ADDR first,GM_ADDR mapping,GM_ADDR second,
    GM_ADDR output,GM_ADDR error,int64_t command,int64_t rows,int64_t regions){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(command==0)tide_block_selection_device::flags((__gm__ uint8_t*)first,(__gm__ int64_t*)mapping,
    (__gm__ uint8_t*)second,(__gm__ uint8_t*)output,(__gm__ int32_t*)error,rows,regions);
  else tide_block_selection_device::merge((__gm__ uint8_t*)first,(__gm__ uint8_t*)second,(__gm__ uint8_t*)output,(__gm__ int32_t*)error,rows);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
