#include "kernel_operator.h"
#include "event_sequence.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_event_indices(GM_ADDR events,GM_ADDR ids,GM_ADDR indices,
    GM_ADDR additive,GM_ADDR error,int64_t heads,int64_t kv_heads,int64_t capacity,int64_t owners,int64_t rows,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)ids);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto e=(__gm__ I*)events,index=(__gm__ I*)ids,out=(__gm__ I*)indices;auto mask=(__gm__ float*)additive;
  const float minus_inf=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000);
  for(I i=AscendC::GetBlockIdx();i<chunk*heads*capacity;i+=AscendC::GetBlockNum()) {
    const I row=i/(heads*capacity),head=(i/capacity)%heads,k=i%capacity,event=index[row];
    I source=(owners*capacity+rows)*kv_heads;float value=minus_inf;
    if(event>=0&&k<e[event*7+4]){source=tide_device::event_key_row(e,event,k,capacity,owners)*kv_heads+head/(heads/kv_heads);value=0.f;}
    else if(event<0&&k==0)value=0.f;
    out[i]=source;if(head==0)mask[row*capacity+k]=value;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
