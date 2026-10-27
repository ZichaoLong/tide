#include "weighted_allocation_math.h"
extern "C" __global__ __aicore__ void tide_weighted_region_result(GM_ADDR lengths,GM_ADDR times,GM_ADDR controls,
    GM_ADDR active,GM_ADDR history_times,GM_ADDR history_present,GM_ADDR error,int64_t rows,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    const I n=((__gm__ I*)lengths)[row];
    ((__gm__ I*)history_times)[row]=n?((__gm__ I*)times)[row]:-1;
    ((__gm__ uint8_t*)history_present)[row]=n>0;
    for(I j=0;j<width;++j)((__gm__ uint8_t*)active)[row*width+j]=j<n&&((__gm__ float*)controls)[row*width+j]>0;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
