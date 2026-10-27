#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_weighted_raw_adopt(GM_ADDR source,GM_ADDR source_integer,
    GM_ADDR target,GM_ADDR target_integer,GM_ADDR error,int64_t nf,int64_t ni){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)source_integer);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(!((__gm__ int32_t*)error)[0]&&AscendC::GetBlockIdx()==0){
    for(I i=0;i<nf;++i)((__gm__ float*)target)[i]=((__gm__ float*)source)[i];
    for(I i=0;i<ni;++i)((__gm__ I*)target_integer)[i]=((__gm__ I*)source_integer)[i];
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
