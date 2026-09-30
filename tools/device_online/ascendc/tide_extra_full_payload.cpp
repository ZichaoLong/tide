#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_extra_full_payload(GM_ADDR values,GM_ADDR upstream,GM_ADDR sources,
    GM_ADDR comparison,GM_ADDR gradient,GM_ADDR content,GM_ADDR error,int64_t capacity,int64_t width,int64_t chunk,int64_t residual) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)sources);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  tide_device::FiberVector v;v.init();auto x=v.x();const I tiles=(width+255)/256;
  for(I task=AscendC::GetBlockIdx();task<chunk*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256,source=((__gm__ I*)sources)[row];
    const uint32_t size=width-start<256?width-start:256;
    if(source==capacity) {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
      v.save(x,(__gm__ float*)comparison,row*width+start,size);v.save(x,(__gm__ float*)gradient,row*width+start,size);continue;}
    v.load(x,(__gm__ float*)values,source*(5*width+2)+3*width+start,size);v.save(x,(__gm__ float*)comparison,row*width+start,size);
    v.load(x,(__gm__ float*)upstream,source*width+start,size);v.save(x,(__gm__ float*)gradient,row*width+start,size);
    if(!residual){AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();v.save(x,(__gm__ float*)content,source*width+start,size);}
  }
}
