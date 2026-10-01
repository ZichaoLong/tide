#include "kernel_operator.h"
// Bool IndexCopy dispatches to AiCPU on this stack. Merge all connection flags,
// the sticky status and exact chunk count with one device metadata writer.
extern "C" __global__ __aicore__ void tide_full_reverse_merge(GM_ADDR destinations,GM_ADDR nodes,
    GM_ADDR content_on,GM_ADDR comparison_on,GM_ADDR parameter_on,GM_ADDR chunks,GM_ADDR source_error,
    GM_ADDR output_content,GM_ADDR output_comparison,GM_ADDR output_parameters,GM_ADDR output_chunks,GM_ADDR error,
    int64_t capacity,int64_t local_nodes,int64_t total_nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)destinations);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,source=(__gm__ int32_t*)source_error;
  auto dst=(__gm__ I*)destinations,ids=(__gm__ I*)nodes,part=(__gm__ I*)chunks,total=(__gm__ I*)output_chunks;
  if(source[0]&&!status[0])status[0]=source[0];
  if(!status[0]) {
    if(part[0]<0||total[0]<0||part[0]>I(0x7fffffffffffffffLL)-total[0])status[0]=2;
    for(I i=0;i<capacity&&!status[0];++i)if(dst[i]<0||dst[i]>=2*capacity)status[0]=2;
    for(I i=0;i<local_nodes&&!status[0];++i)if(ids[i]<0||ids[i]>=total_nodes)status[0]=2;
    if(!status[0]) {
      for(I i=0;i<capacity;++i)if(dst[i]<capacity) {
        ((__gm__ uint8_t*)output_content)[dst[i]]=((__gm__ uint8_t*)content_on)[i];
        ((__gm__ uint8_t*)output_comparison)[dst[i]]=((__gm__ uint8_t*)comparison_on)[i];
      }
      for(I i=0;i<local_nodes;++i)((__gm__ uint8_t*)output_parameters)[ids[i]]=((__gm__ uint8_t*)parameter_on)[i];
      total[0]+=part[0];
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
