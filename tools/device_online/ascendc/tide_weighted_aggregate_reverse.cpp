#include "weighted_aggregate_math.h"
extern "C" __global__ __aicore__ void tide_weighted_aggregate_reverse(GM_ADDR payload,GM_ADDR lengths,
    GM_ADDR values,GM_ADDR output,GM_ADDR gradient,GM_ADDR roots,GM_ADDR input_gradient,
    GM_ADDR transform_gradient,GM_ADDR connected,GM_ADDR source_roots,GM_ADDR row_errors,GM_ADDR error,
    int64_t rows,int64_t width,int64_t d,float alpha){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_aggregate_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto e=(__gm__ int32_t*)row_errors+row;*e=0;const I n=((__gm__ I*)lengths)[row];
    if(n<0||n>width){*e=2;continue;}
    *e=reverse_mean((__gm__ float*)payload+row*width*(d+1),(__gm__ float*)values+row*width*d,
      (__gm__ float*)output+row*(d+1),(__gm__ float*)gradient+row*(d+1),n,width,d,alpha,
      ((__gm__ uint8_t*)roots)[row],(__gm__ float*)input_gradient+row*width*(d+1),
      (__gm__ float*)transform_gradient+row*width*d,(__gm__ uint8_t*)connected+row*width,(__gm__ uint8_t*)source_roots+row*width);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
