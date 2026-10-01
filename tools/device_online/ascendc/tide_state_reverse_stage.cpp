#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_reverse_stage(GM_ADDR metadata,GM_ADDR count,GM_ADDR event_rows,GM_ADDR global_range,
    GM_ADDR stage_meta,GM_ADDR stage_count,GM_ADDR local_range,GM_ADDR local_rows,GM_ADDR source_rows,GM_ADDR destinations,GM_ADDR error,
    int64_t capacity,int64_t global_capacity) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto m=(__gm__ I*)metadata,e=(__gm__ I*)event_rows,g=(__gm__ I*)global_range;
  auto out=(__gm__ I*)stage_meta,n=(__gm__ I*)stage_count,r=(__gm__ I*)local_range,local=(__gm__ I*)local_rows;
  auto src=(__gm__ I*)source_rows,dst=(__gm__ I*)destinations;const I size=((__gm__ I*)count)[0];n[0]=0;r[0]=r[1]=0;
  for(I i=0;i<capacity;++i){local[i]=capacity;src[i]=global_capacity;dst[i]=global_capacity+i;for(I j=0;j<13;++j)out[i*13+j]=0;}
  if(status[0]==0&&(size<0||size>capacity||g[0]<0||g[1]<g[0]||g[1]>global_capacity))status[0]=2;
  if(!status[0]) {
    I first=0;while(first<size&&e[first]<g[0])++first;I stop=first;while(stop<size&&e[stop]<g[1])++stop;
    r[0]=first;r[1]=stop;n[0]=stop-first;
    for(I i=first;i<stop;++i){const I row=i-first;local[row]=i;src[row]=dst[row]=e[i]-g[0];
      for(I j=0;j<13;++j)out[row*13+j]=m[i*13+j];}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
