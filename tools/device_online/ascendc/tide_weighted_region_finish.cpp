#include "weighted_region_prepare.h"
extern "C" __global__ __aicore__ void tide_weighted_region_finish(GM_ADDR local_score,GM_ADDR keys,
    GM_ADDR lengths,GM_ADDR statistics,GM_ADDR context_values,GM_ADDR query_values,GM_ADDR scores,
    GM_ADDR bounds,GM_ADDR parameters,GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,
    int64_t qwidth,int64_t mode,int64_t context,int64_t attention,int64_t k,float fixed,float tau,
    float alpha_local,float alpha_query,float inverse_sqrt){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_region_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;Math math;math.init();
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto s=(__gm__ float*)scores+row*width,b=(__gm__ float*)bounds+row*4,p=(__gm__ float*)parameters+row*2;
    auto e=(__gm__ int32_t*)row_errors+row;*e=0;for(I j=0;j<width;++j)s[j]=0;
    for(I j=0;j<4;++j)b[j]=0;for(I j=0;j<2;++j)p[j]=0;
    const I n=((__gm__ I*)lengths)[row];if(n<0||n>width){*e=2;continue;}if(!n)continue;
    *e=finish(math,(__gm__ float*)local_score+row*width,(__gm__ float*)keys+row*width*qwidth,n,qwidth,
      mode,context,attention,k,fixed,tau,alpha_local,alpha_query,inverse_sqrt,
      (__gm__ float*)statistics+row*Statistics,(__gm__ float*)context_values+row,(__gm__ float*)query_values+row*qwidth,s,b,p);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
