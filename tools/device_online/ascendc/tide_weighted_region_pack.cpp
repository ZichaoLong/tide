#include "kernel_operator.h"
// Compact actual candidates in canonical node-ID order. Each row is a complete
// sample/region/time frame; all members share its denominator and allocation.
extern "C" __global__ __aicore__ void tide_weighted_region_pack(GM_ADDR present,GM_ADDR time,GM_ADDR nodes,
    GM_ADDR order,GM_ADDR ids,GM_ADDR times,GM_ADDR lengths,GM_ADDR inverse,GM_ADDR error,int64_t samples,int64_t members){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)order);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto live=(__gm__ uint8_t*)present;auto t=(__gm__ I*)time,node=(__gm__ I*)nodes;
  auto ix=(__gm__ I*)order,id=(__gm__ I*)ids,clock=(__gm__ I*)times,n=(__gm__ I*)lengths,back=(__gm__ I*)inverse;
  auto status=(__gm__ int32_t*)error;
  for(I i=0;i<samples*members;++i){ix[i]=back[i]=samples*members;id[i]=0;}
  for(I b=0;b<samples;++b){n[b]=0;clock[b]=0;if(status[0])continue;
    for(I j=0;j<members;++j)if(live[b*members+j]){
      I k=n[b];if(k&&clock[b]!=t[b*members+j]){status[0]=2;break;}
      clock[b]=t[b*members+j];ix[b*members+k]=b*members+j;id[b*members+k]=node[j];back[b*members+j]=b*members+k;++n[b];
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
