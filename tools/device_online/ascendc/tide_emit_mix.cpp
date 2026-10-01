#include "fiber_vector.h"
namespace {
template<class T>
__aicore__ inline void emit_mix(GM_ADDR coordinates,GM_ADDR valid,GM_ADDR identities,GM_ADDR content,
    GM_ADDR fresh,GM_ADDR controls,GM_ADDR output,GM_ADDR error,int64_t rows,int64_t width,int64_t fp16) {
  using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)coordinates);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I tiles=(width+255)/256;tide_device::FiberVector v;v.init();auto x=v.x(),y=v.y();
  for(I task=AscendC::GetBlockIdx();task<rows*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;if(!((__gm__ uint8_t*)valid)[row])continue;
    const I node=((__gm__ I*)coordinates)[row*4+1];if(((__gm__ I*)identities)[node]<0)continue;
    const uint32_t size=width-start<256?width-start:256;const float p=float(T(((__gm__ float*)controls)[row]));
    v.load(x,(__gm__ T*)fresh,row*width+start,size);v.load(y,(__gm__ T*)content,row*width+start,size);
    AscendC::Sub(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
    if(fp16) {
      auto rounded=y.ReinterpretCast<half>();
      AscendC::Cast(rounded,x,AscendC::RoundMode::CAST_RINT,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(x,rounded,AscendC::RoundMode::CAST_NONE,size);AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Muls(x,x,p,size);AscendC::PipeBarrier<PIPE_V>();
    if(fp16) {
      auto rounded=y.ReinterpretCast<half>();
      AscendC::Cast(rounded,x,AscendC::RoundMode::CAST_RINT,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(x,rounded,AscendC::RoundMode::CAST_NONE,size);AscendC::PipeBarrier<PIPE_V>();
      v.load(y,(__gm__ T*)content,row*width+start,size);
    }
    AscendC::Add(x,y,x,size);AscendC::PipeBarrier<PIPE_V>();v.save(x,(__gm__ T*)output,row*width+start,size);
  }
}
}
extern "C" __global__ __aicore__ void tide_emit_mix(GM_ADDR coordinates,GM_ADDR valid,GM_ADDR identities,GM_ADDR content,
    GM_ADDR fresh,GM_ADDR controls,GM_ADDR output,GM_ADDR error,int64_t rows,int64_t width,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16)emit_mix<half>(coordinates,valid,identities,content,fresh,controls,output,error,rows,width,fp16);
  else emit_mix<float>(coordinates,valid,identities,content,fresh,controls,output,error,rows,width,fp16);
}
