#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_weighted_owner_adopt(GM_ADDR target,GM_ADDR present,GM_ADDR times,
    GM_ADDR error,int64_t samples,int64_t history){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)target);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto out=(__gm__ I*)target,clock=(__gm__ I*)times;auto live=(__gm__ uint8_t*)present;
  if(!((__gm__ int32_t*)error)[0])for(I b=0;b<samples;++b)if(live[b]){
    if(history){out[b*2]=clock[b];out[b*2+1]=1;}else out[b]=1;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
