#include "weighted_allocation_math.h"
extern "C" __global__ __aicore__ void tide_weighted_sum_roots(GM_ADDR pool_roots,GM_ADDR window_roots,GM_ADDR error,int64_t rows){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(!((__gm__ int32_t*)error)[0]&&AscendC::GetBlockIdx()==0){auto p=(__gm__ uint8_t*)pool_roots;auto w=(__gm__ uint8_t*)window_roots;
    for(I r=0;r<rows;++r){w[5*r]=w[5*r+1]=0;w[5*r+2]=p[6*r+1]||p[6*r+4];w[5*r+3]=p[6*r];w[5*r+4]=p[6*r+3];}}
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
