#include "kernel_operator.h"
namespace {
using I=int64_t;
// Scalar numerical option. The same complete metadata preflight and projected
// source order feed both scalar and vector implementations.
template<class T>
__aicore__ inline void sum(GM_ADDR values,GM_ADDR offsets,
    GM_ADDR lengths,GM_ADDR keys,GM_ADDR order,GM_ADDR scales,GM_ADDR content,
    GM_ADDR weighted,GM_ADDR error,int64_t width) {
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)keys);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto o=(__gm__ I*)offsets,k=(__gm__ I*)keys,ordered=(__gm__ I*)order;
  auto x=(__gm__ T*)values,w=(__gm__ T*)scales,h=(__gm__ T*)content,z=(__gm__ T*)weighted;
  const I count=((__gm__ I*)lengths)[1];
  for(I i=0;i<count;++i) {
    const I first=o[i],end=o[i+1];
    for(I j=0;j<width;++j) {
      float total=0.f;
      for(I pos=first;pos<end;++pos) {
        const I a=ordered[pos];const float contribution=float(x[a*width+j])*float(w[k[a]]);
        z[a*width+j]=T(contribution);total=pos==first?contribution:total+contribution;
      }
      h[i*width+j]=T(total);
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
}
extern "C" __global__ __aicore__ void tide_content_sum(GM_ADDR values,GM_ADDR offsets,
    GM_ADDR lengths,GM_ADDR keys,GM_ADDR order,GM_ADDR scales,GM_ADDR content,
    GM_ADDR weighted,GM_ADDR error,int64_t width,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16)sum<half>(values,offsets,lengths,keys,order,scales,content,weighted,error,width);
  else sum<float>(values,offsets,lengths,keys,order,scales,content,weighted,error,width);
}
