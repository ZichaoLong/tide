// Effective source matrices: mask absent adjoints before vendor products.
#include "weighted_allocation_math.h"
extern "C" __global__ __aicore__ void tide_weighted_source_guard(GM_ADDR input,GM_ADDR roots,
    GM_ADDR output,GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,int64_t stage){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto e=(__gm__ int32_t*)row_errors+row;*e=0;
    const bool root=stage==0||((__gm__ uint8_t*)roots)[row];
    for(I j=0;j<width;++j){const float value=root?((__gm__ float*)input)[row*width+j]:0;
      if(!finite(value))*e=stage==0?20:23;
      ((__gm__ float*)output)[row*width+j]=value;}
  }
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
