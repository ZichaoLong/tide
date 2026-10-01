#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_control_read_reduce(GM_ADDR head,GM_ADDR next,GM_ADDR connected,
    GM_ADDR partials,GM_ADDR output,GM_ADDR error,int64_t nodes,int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)head);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I tiles=(width+255)/256;tide_device::FiberVector v;v.init();auto x=v.x(),y=v.y();
  for(I task=AscendC::GetBlockIdx();task<nodes*tiles;task+=AscendC::GetBlockNum()) {
    const I node=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
    AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    for(I row=((__gm__ I*)head)[node];row>=0;row=((__gm__ I*)next)[row])if(((__gm__ uint8_t*)connected)[row]) {
      v.load(y,(__gm__ float*)partials,row*width+start,size);AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();}
    v.save(x,(__gm__ float*)output,node*width+start,size);
  }
}
