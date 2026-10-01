#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_vjp_state(GM_ADDR proposal,GM_ADDR on,GM_ADDR projection,
    GM_ADDR key_root,GM_ADDR value_root,GM_ADDR bias_root,GM_ADDR key_on,GM_ADDR value_on,GM_ADDR bias_on,
    GM_ADDR lengths,GM_ADDR old_lengths,GM_ADDR ticks,GM_ADDR parameter_on,
    GM_ADDR safe_root,GM_ADDR safe_projection,GM_ADDR key,GM_ADDR value,GM_ADDR bias,GM_ADDR decay,
    GM_ADDR error,int64_t batch,int64_t width,int64_t heads,int64_t capacity,int64_t mode,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I d=width/heads,dt=(d+255)/256,wt=(width+255)/256;
  auto active=(__gm__ uint8_t*)on;auto n=(__gm__ I*)(mode?old_lengths:lengths);
  tide_device::FiberVector op;op.init();auto x=op.x();
  if(mode==0) {
    for(I task=AscendC::GetBlockIdx();task<batch*wt;task+=AscendC::GetBlockNum()) {
      const I b=task/wt,start=(task%wt)*256;const uint32_t size=width-start<256?width-start:256;
      if(active[b])op.load(x,(__gm__ float*)proposal,b*width+start,size);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)safe_root,b*width+start,size);
    }
    for(I task=AscendC::GetBlockIdx();task<batch*width*wt;task+=AscendC::GetBlockNum()) {
      const I b=task/width/wt,row=(task/wt)%width,start=(task%wt)*256;
      const uint32_t size=width-start<256?width-start:256;
      if(active[b]) {
        const I at=(b*width+row)*width+start;
        if(fp16)op.load(x,(__gm__ half*)projection,at,size);else op.load(x,(__gm__ float*)projection,at,size);
      }
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)safe_projection,(b*width+row)*width+start,size);
    }
  }
  for(I task=AscendC::GetBlockIdx();task<batch*heads*capacity*dt;task+=AscendC::GetBlockNum()) {
    const I start=(task%dt)*256,k=(task/dt)%capacity,b=task/dt/capacity/heads;
    const I at=(task/dt)*d+start;const uint32_t size=d-start<256?d-start:256;
    if(mode&&k<n[b])continue;
    for(I which=0;which<2;++which) {
      if(!mode&&k<n[b]&&((__gm__ uint8_t*)(which?value_on:key_on))[b])
        op.load(x,(__gm__ float*)(which?value_root:key_root),at,size);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)(which?value:key),at,size);
    }
  }
  // Each owner exclusively writes its bias row and scalar decay adjoint.
  // Preserve repeated-subtraction VJP accumulation rather than rounding one
  // multiplication by a potentially large logical tick count.
  if(AscendC::GetBlockIdx()==0)for(I b=0;b<batch;++b) {
    auto out=(__gm__ float*)bias;float sum=0;
    for(I k=0;k<capacity;++k) {
      const I at=b*capacity+k;
      if(!mode)out[at]=k<n[b]&&((__gm__ uint8_t*)bias_on)[b]?((__gm__ float*)bias_root)[at]:0.f;
      else if(k<n[b])sum+=out[at];else out[at]=0.f;
    }
    if(mode) {
      float total=0;
      if(((__gm__ uint8_t*)parameter_on)[b*6+4])for(I tick=0;tick<((__gm__ I*)ticks)[b];++tick)total-=sum;
      ((__gm__ float*)decay)[b]=total;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
