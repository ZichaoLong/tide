#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_attention_merge(GM_ADDR partial,GM_ADDR normalization,
    GM_ADDR sum,GM_ADDR error,int64_t rows,int64_t width,int64_t normalize) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)normalization);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();const I tiles=(width+255)/256;
  auto stat=(__gm__ float*)normalization;
  for(I task=AscendC::GetBlockIdx();task<rows*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
    op.load(x,(__gm__ float*)sum,row*width+start,size);
    if(normalize) {
      const float denominator=stat[row*8+1];
      if(denominator==0) {
        const float empty=stat[row*8+3]==0?0.f:AscendC::GetScalarBitcodeValue<uint32_t,float>(0x7fc00000);
        AscendC::Duplicate(x,empty,size);
      }
      else AscendC::Muls(x,x,1.f/denominator,size);
      AscendC::PipeBarrier<PIPE_V>();
    }else {
      const float scale=stat[row*8+2];AscendC::Muls(x,x,scale,size);AscendC::PipeBarrier<PIPE_V>();
      op.load(y,(__gm__ float*)partial,row*width+start,size);AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
    }
    op.save(x,(__gm__ float*)sum,row*width+start,size);
  }
}
