#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_vjp_pool(GM_ADDR counts,GM_ADDR slots,GM_ADDR kinds,
    GM_ADDR lengths,GM_ADDR weights,GM_ADDR on,GM_ADDR logits,GM_ADDR probabilities,GM_ADDR coefficients,
    GM_ADDR partials,GM_ADDR gradient,GM_ADDR pool_scale,GM_ADDR error,int64_t batch,int64_t sources,int64_t domain,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)counts);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto c=(__gm__ I*)counts,s=(__gm__ I*)slots,k=(__gm__ I*)kinds,n=(__gm__ I*)lengths;
  auto w=(__gm__ float*)weights,l=(__gm__ float*)logits,p=(__gm__ float*)probabilities;
  auto a=(__gm__ float*)coefficients,dp=(__gm__ float*)partials,g=(__gm__ float*)gradient;
  const float minus_inf=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000);
  for(I b=0;b<batch;++b) {
    const bool active=((__gm__ uint8_t*)on)[b];
    if(mode==0) {
      ((__gm__ float*)pool_scale)[b]=active&&k[b]==1?1.f/float(c[b]):1.f;
      for(I j=0;j<domain;++j)l[b*domain+j]=minus_inf;
      if(!active||k[b]<3)l[b*domain]=0.f;
      else if(k[b]==4)for(I j=0;j<n[b];++j)l[b*domain+j]=w[b*domain+j];
      else for(I j=0;j<c[b];++j)l[b*domain+s[b*sources+j]]=w[b*domain+s[b*sources+j]];
    }else if(mode==1) {
      for(I j=0;j<sources;++j) {
        float x=0;
        if(active&&j<c[b]) {
          const I at=b*domain+s[b*sources+j];
          x=k[b]==0?1.f:k[b]==1?1.f/float(c[b]):k[b]==2?w[at]:p[at];
        }
        a[b*sources+j]=x;
      }
    }else {
      float dot=0;
      if(active&&k[b]>=3)for(I j=0;j<n[b];++j)dot+=p[b*domain+j]*dp[b*domain+j];
      for(I j=0;j<domain;++j) {
        const I at=b*domain+j;
        g[at]=!active||k[b]<2||j>=n[b]?0.f:k[b]==2?dp[at]:p[at]*(dp[at]-dot);
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
