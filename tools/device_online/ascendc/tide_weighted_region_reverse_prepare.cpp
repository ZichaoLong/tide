#include "weighted_region_reverse.h"
extern "C" __global__ __aicore__ void tide_weighted_region_reverse_prepare(GM_ADDR lengths,GM_ADDR received,
    GM_ADDR caps,GM_ADDR evidence,GM_ADDR current,GM_ADDR statistics,GM_ADDR context_input,GM_ADDR query_input,
    GM_ADDR context_input_grad,GM_ADDR query_input_grad,GM_ADDR context_roots,GM_ADDR query_roots,
    GM_ADDR received_grad,GM_ADDR caps_grad,GM_ADDR evidence_grad,GM_ADDR current_grad,GM_ADDR row_errors,GM_ADDR error,
    int64_t rows,int64_t width,int64_t ewidth,int64_t qwidth,int64_t query_received){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_region_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto ge=(__gm__ float*)evidence_grad+row*width*ewidth,gx=(__gm__ float*)current_grad+row*width*qwidth;
    for(I j=0;j<width*ewidth;++j)ge[j]=0;for(I j=0;j<width*qwidth;++j)gx[j]=0;
    auto e=(__gm__ int32_t*)row_errors+row;*e=0;
    const I n=((__gm__ I*)lengths)[row];if(n<0||n>width){*e=2;continue;}if(!n)continue;
    *e=reverse_prepare(n,ewidth,qwidth,((__gm__ uint8_t*)context_roots)[row],((__gm__ uint8_t*)query_roots)[row],query_received,
      (__gm__ float*)received+row*width,(__gm__ float*)caps+row*width,(__gm__ float*)evidence+row*width*ewidth,
      (__gm__ float*)current+row*width*qwidth,(__gm__ float*)statistics+row*Statistics,
      (__gm__ float*)context_input+row*(ewidth+2),(__gm__ float*)query_input+row*qwidth,
      (__gm__ float*)context_input_grad+row*(ewidth+2),(__gm__ float*)query_input_grad+row*qwidth,
      (__gm__ float*)received_grad+row*width,(__gm__ float*)caps_grad+row*width,ge,gx);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
