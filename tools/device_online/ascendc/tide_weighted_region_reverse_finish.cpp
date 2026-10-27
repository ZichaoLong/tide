#include "weighted_region_reverse.h"
extern "C" __global__ __aicore__ void tide_weighted_region_reverse_finish(GM_ADDR lengths,GM_ADDR statistics,
    GM_ADDR context_values,GM_ADDR query_values,GM_ADDR keys,GM_ADDR vectors,GM_ADDR scalars,GM_ADDR connected,
    GM_ADDR received_grad,GM_ADDR caps_grad,GM_ADDR score_grad,GM_ADDR key_grad,GM_ADDR context_grad,GM_ADDR query_grad,
    GM_ADDR field_roots,GM_ADDR context_roots,GM_ADDR query_roots,GM_ADDR row_errors,GM_ADDR error,
    int64_t rows,int64_t width,int64_t qwidth,int64_t mode,int64_t context,int64_t attention,int64_t query_received,
    int64_t k,float fixed,float alpha_local,float alpha_query,float inverse_sqrt){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_region_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;Math math;math.init();
  const bool allocated=mode==Total||mode==Threshold||mode==Constrained;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto gw=(__gm__ float*)received_grad+row*width,gm=(__gm__ float*)caps_grad+row*width;
    auto gs=(__gm__ float*)score_grad+row*width,gkey=(__gm__ float*)key_grad+row*width*qwidth;
    auto gh=(__gm__ float*)context_grad+row,gq=(__gm__ float*)query_grad+row*qwidth;
    auto flags=(__gm__ uint8_t*)field_roots+row*Fields;
    auto hr=(__gm__ uint8_t*)context_roots+row,qr=(__gm__ uint8_t*)query_roots+row;
    for(I j=0;j<width;++j){gw[j]=0;gm[j]=0;gs[j]=0;for(I d=0;d<qwidth;++d)gkey[j*qwidth+d]=0;}
    *gh=0;for(I d=0;d<qwidth;++d)gq[d]=0;for(I j=0;j<Fields;++j)flags[j]=0;*hr=0;*qr=0;
    auto e=(__gm__ int32_t*)row_errors+row;*e=0;
    const I n=((__gm__ I*)lengths)[row];if(n<0||n>width){*e=2;continue;}if(!n)continue;
    *e=reverse_finish(n,width,qwidth,mode,context,attention,query_received,k,fixed,alpha_local,alpha_query,inverse_sqrt,
      (__gm__ float*)statistics+row*Statistics,(__gm__ float*)context_values+row,(__gm__ float*)query_values+row*qwidth,
      (__gm__ float*)keys+row*width*qwidth,(__gm__ float*)vectors+row*width*(allocated?2:3),
      (__gm__ float*)scalars+row*(allocated?4:2),(__gm__ uint8_t*)connected+row*(allocated?6:5),
      gw,gm,gs,gkey,gh,gq,flags,hr,qr,math);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
