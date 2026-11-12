#include "block_read_selection_math.h"
extern "C" __global__ __aicore__ void tide_block_selection_roots(GM_ADDR order,GM_ADDR bars,GM_ADDR connected,
    GM_ADDR packed,GM_ADDR row_roots,GM_ADDR error,int64_t rows,int64_t width,int64_t input_rows){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_selection_device::roots((__gm__ int64_t*)order,(__gm__ float*)bars,(__gm__ uint8_t*)connected,
    (__gm__ float*)packed,(__gm__ uint8_t*)row_roots,(__gm__ int32_t*)error,rows,width,input_rows);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
