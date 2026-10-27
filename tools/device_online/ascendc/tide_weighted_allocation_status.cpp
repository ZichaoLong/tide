#include "kernel_operator.h"
// Separate all-row decision; no racing writes to the shared sticky error.
extern "C" __global__ __aicore__ void tide_weighted_allocation_status(GM_ADDR row_errors,GM_ADDR error,int64_t rows){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)row_errors);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;
  if(!status[0])for(int64_t row=0;row<rows;++row)if(((__gm__ int32_t*)row_errors)[row]){
    status[0]=((__gm__ int32_t*)row_errors)[row];break;
  }
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
