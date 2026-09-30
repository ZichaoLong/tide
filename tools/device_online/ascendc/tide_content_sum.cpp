#include "kernel_operator.h"
namespace {using I=int64_t;}
// Scalar numerical option. The same complete metadata preflight and projected
// source order feed both scalar and vector implementations.
extern "C" __global__ __aicore__ void tide_content_sum(GM_ADDR values,GM_ADDR offsets,
    GM_ADDR lengths,GM_ADDR keys,GM_ADDR order,GM_ADDR scales,GM_ADDR content,
    GM_ADDR weighted,GM_ADDR error,int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)keys);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto o=(__gm__ I*)offsets,k=(__gm__ I*)keys,ordered=(__gm__ I*)order;
  auto x=(__gm__ float*)values,w=(__gm__ float*)scales,h=(__gm__ float*)content,z=(__gm__ float*)weighted;
  const I count=((__gm__ I*)lengths)[1];
  for(I i=0;i<count;++i) {
    const I first=o[i],end=o[i+1];
    for(I pos=first;pos<end;++pos) {
      const I a=ordered[pos];
      for(I j=0;j<width;++j)z[a*width+j]=x[a*width+j]*w[k[a]];
    }
    for(I j=0;j<width;++j) {
      float sum=z[ordered[first]*width+j];
      for(I pos=first+1;pos<end;++pos)sum=sum+z[ordered[pos]*width+j];
      h[i*width+j]=sum;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
