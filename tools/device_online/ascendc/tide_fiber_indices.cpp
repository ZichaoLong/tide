#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_indices(GM_ADDR events,GM_ADDR tokens,GM_ADDR ids,
    GM_ADDR bias,GM_ADDR indices,GM_ADDR additive,GM_ADDR error,int64_t capacity,int64_t owners,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)ids);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto e=(__gm__ I*)events,t=(__gm__ I*)tokens,index=(__gm__ I*)ids,out=(__gm__ I*)indices;
  auto b=(__gm__ float*)bias,a=(__gm__ float*)additive;
  const float minus_inf=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000);
  for(I i=AscendC::GetBlockIdx();i<chunk*capacity;i+=AscendC::GetBlockNum()) {
    const I query=i/capacity,k=i%capacity,token=index[query];
    I source=owners*capacity;float value=minus_inf;
    if(token>=0) {
      const I event=t[token*4+1],length=e[event*7+4],owner=e[event*7+1];
      if(k<length){source=owner*capacity+k;value=b[event*capacity+k];}
    }else if(k==0)value=0.f; // Padding has its own finite denominator and zero KV.
    out[i]=source;a[i]=value;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
