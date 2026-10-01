#include "fiber_vector.h"
namespace {using I=int64_t;}
// One event/feature tile owns its producer reduction. Disconnected and absent
// slots are never evaluated; parameter connectivity is independent of value.
extern "C" __global__ __aicore__ void tide_emission_vjp_payload(GM_ADDR messages,GM_ADDR heads,GM_ADDR next,
    GM_ADDR mapping,GM_ADDR scales,GM_ADDR connected,GM_ADDR gradients,GM_ADDR range,GM_ADDR upstream,
    GM_ADDR projected,GM_ADDR full_gradient,GM_ADDR error,int64_t width,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)messages);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto msg=(__gm__ I*)messages,h=(__gm__ I*)heads,n=(__gm__ I*)next,map=(__gm__ I*)mapping,r=(__gm__ I*)range;
  const I first=r[0],rows=r[1]-first,tiles=(width+255)/256;
  tide_device::FiberVector vector;vector.init();auto x=vector.x(),y=vector.y();
  for(I task=AscendC::GetBlockIdx();task<rows*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
    AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    for(I m=h[first+row];m>=0;m=n[m])if(((__gm__ uint8_t*)connected)[m]) {
      if(mode==0) {
        const float scale=((__gm__ float*)scales)[msg[m*4+3]];
        vector.load(y,(__gm__ float*)gradients,m*width+start,size);
        AscendC::Muls(y,y,scale,size);AscendC::PipeBarrier<PIPE_V>();
        vector.save(y,(__gm__ float*)upstream,m*width+start,size);
        if(map[m]<0)vector.save(y,(__gm__ float*)projected,m*width+start,size);
      } else {
        vector.load(y,(__gm__ float*)projected,m*width+start,size);
        AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      }
    }
    if(mode==1)vector.save(x,(__gm__ float*)full_gradient,row*width+start,size);
  }
}
