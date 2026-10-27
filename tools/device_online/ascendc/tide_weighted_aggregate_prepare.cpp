#include "weighted_aggregate_math.h"
extern "C" __global__ __aicore__ void tide_weighted_aggregate_prepare(GM_ADDR payload,GM_ADDR lengths,
    GM_ADDR owners,GM_ADDR clean,GM_ADDR active_owners,GM_ADDR totals,GM_ADDR row_errors,GM_ADDR error,
    int64_t rows,int64_t width,int64_t d,float alpha){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_aggregate_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto e=(__gm__ int32_t*)row_errors+row;*e=0;const I n=((__gm__ I*)lengths)[row];
    if(n<0||n>width){*e=2;continue;}
    *e=prepare((__gm__ float*)payload+row*width*(d+1),(__gm__ I*)owners+row*width,n,width,d,alpha,
      (__gm__ float*)clean+row*width*d,(__gm__ I*)active_owners+row*width,(__gm__ float*)totals+row);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
