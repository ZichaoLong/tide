// Sending support is decided before payload storage. Never silently lose a
// positive coefficient when rounding (c,G); inactive poison is not a value.
#include "weighted_allocation_math.h"
extern "C" __global__ __aicore__ void tide_weighted_payload_guard(GM_ADDR source,GM_ADDR rounded,
    GM_ADDR present,GM_ADDR error,int64_t rows,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto e=(__gm__ int32_t*)error;if(*e)return;
  for(I row=0;row<rows;++row)if(((__gm__ uint8_t*)present)[row]){
    const auto before=(__gm__ float*)source+row*width,after=(__gm__ float*)rounded+row*width;
    for(I j=0;j<width;++j)if(!finite(before[j])||!finite(after[j]))*e=24;
    if(before[0]<0||after[0]<0||(before[0]>0&&after[0]<=0))*e=24;
  }
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
