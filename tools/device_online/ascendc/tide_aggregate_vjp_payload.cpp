#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_aggregate_vjp_payload(GM_ADDR ids,GM_ADDR message_ids,GM_ADDR links,
    GM_ADDR source_scales,GM_ADDR fiber_values,GM_ADDR content_gradient,GM_ADDR packed_gradient,GM_ADDR weighted,
    GM_ADDR probabilities,GM_ADDR messages,GM_ADDR scale_partials,GM_ADDR error,
    int64_t chunk,int64_t slots,int64_t width,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)ids);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  tide_device::FiberVector v;v.init();auto x=v.x(),y=v.y();const I tiles=(width+255)/256;
  auto mi=(__gm__ I*)message_ids,link=(__gm__ I*)links;auto scales=(__gm__ float*)source_scales;
  for(I task=AscendC::GetBlockIdx();task<chunk*slots*tiles;task+=AscendC::GetBlockNum()) {
    const I fiber=task/tiles,row=fiber/slots,slot=fiber%slots,start=(task%tiles)*256;
    const I message=mi[fiber],event=((__gm__ I*)ids)[row];const uint32_t size=width-start<256?width-start:256;
    if(mode==0) {
      if(slot==0) {
        if(event>=0)v.load(x,(__gm__ float*)content_gradient,event*width+start,size);
        else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        v.save(x,(__gm__ float*)packed_gradient,row*width+start,size);
      }
      if(message>=0){v.load(x,(__gm__ float*)fiber_values,message*width+start,size);
        const float scale=scales[link[message*4+2]];
        AscendC::Muls(x,x,scale,size);AscendC::PipeBarrier<PIPE_V>();}
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      v.save(x,(__gm__ float*)weighted,fiber*width+start,size);
    } else if(message>=0) {
      v.load(x,(__gm__ float*)packed_gradient,row*width+start,size);
      const float probability=((__gm__ float*)probabilities)[fiber],scale=scales[link[message*4+2]];
      AscendC::Muls(x,x,probability,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(y,x,scale,size);AscendC::PipeBarrier<PIPE_V>();
      v.save(y,(__gm__ float*)messages,message*width+start,size);
      v.load(y,(__gm__ float*)fiber_values,message*width+start,size);AscendC::Mul(y,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      v.save(y,(__gm__ float*)scale_partials,message*width+start,size);
    }
  }
}
