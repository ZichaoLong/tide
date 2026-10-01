#include "kernel_operator.h"
#include "../optimizer_layout.h"
// Runs after the numerical commit kernel. A simultaneous counter write would
// race SGD's first-use momentum decision on another vector core.
extern "C" __global__ __aicore__ void tide_optimizer_commit(GM_ADDR table,GM_ADDR connected,
    GM_ADDR steps,GM_ADDR corrections,GM_ADDR next_steps,GM_ADDR next_corrections,
    GM_ADDR error,int64_t count,int64_t kind) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_device;using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)table);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto t=(__gm__ I*)table;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<count;++i)
    if(t[i*OWNER_FIELDS+OFFSET]>=0&&t[i*OWNER_FIELDS+GROUP]>=0&&((__gm__ uint8_t*)connected)[i]) {
      ((__gm__ I*)steps)[i]=((__gm__ I*)next_steps)[i];
      if(kind){((__gm__ float*)corrections)[i*2]=((__gm__ float*)next_corrections)[i*2];
        ((__gm__ float*)corrections)[i*2+1]=((__gm__ float*)next_corrections)[i*2+1];}
    }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
