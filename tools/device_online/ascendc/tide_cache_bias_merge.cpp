#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_cache_bias_merge(GM_ADDR lengths,GM_ADDR left,GM_ADDR left_on,
    GM_ADDR right,GM_ADDR right_on,GM_ADDR output,GM_ADDR connected,GM_ADDR error,
    int64_t owners,int64_t capacity,int64_t combine,int64_t phase) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto len=(__gm__ I*)lengths;auto a=(__gm__ uint8_t*)left_on,b=(__gm__ uint8_t*)right_on;
  if(phase==0) {
    if(AscendC::GetBlockIdx()!=0)return;
    for(I owner=0;owner<owners;++owner)((__gm__ uint8_t*)connected)[owner]=a[owner]||(combine&&b[owner]);
  }else {
    const I tiles=(capacity+255)/256;tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
    for(I task=AscendC::GetBlockIdx();task<owners*tiles;task+=AscendC::GetBlockNum()) {
      const I owner=task/tiles,start=(task%tiles)*256,at=owner*capacity+start;
      const uint32_t size=capacity-start<256?capacity-start:256;
      const uint32_t valid=len[owner]<=start?0:len[owner]-start<size?len[owner]-start:size;
      AscendC::Duplicate(x,0.f,size);AscendC::Duplicate(y,0.f,size);AscendC::PipeBarrier<PIPE_V>();
      if(valid&&a[owner])op.load(x,(__gm__ float*)left,at,valid);
      if(valid&&combine&&b[owner])op.load(y,(__gm__ float*)right,at,valid);
      AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();op.save(x,(__gm__ float*)output,at,size);
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
